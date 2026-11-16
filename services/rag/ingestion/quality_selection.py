"""Auditable, lossless quality partitioning for a Drive chunk batch."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


INTENTIONALLY_EXCLUDED = {
    "1WoUidKMTBMNMvfakro3tSSYkZHLcWGfn": "Intentionally excluded: extracted content was reported as gibberish and unsuitable for the legal corpus.",
    "1tyy1nrrsIR-ue8lLIddinJF1zPPxKKVU": "Intentionally excluded: extracted content was reported as gibberish and unsuitable for the legal corpus; this is an irrelevant test document.",
}

QUARANTINE_WARNING_CODES = {
    "toc_or_front_matter_classified",
    "classified_before_body",
    "classified_without_detected_body",
    "section_metadata_text_mismatch",
    "multiple_section_heading_candidates",
    "subsection_marker_absent",
    "chapter_title_inconsistent",
}


class QualitySelectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Decision:
    status: str
    reasons: tuple[str, ...]
    review_flags: tuple[str, ...] = ()


def _toc_only_text(text: str) -> bool:
    """Conservative TOC-only check; mixed provision text is never discarded."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    marker = any(re.fullmatch(r"(?:arrangement of sections|contents|table of contents)", x, re.I) for x in lines)
    if not marker:
        return False
    listing_lines = sum(
        bool(re.search(r"(?:\.{2,}|\s{2,}|\s+\d{1,4})\s*$", line))
        or bool(re.match(r"^(?:chapter|part)\s+[ivxlcdm\d]+\b", line, re.I))
        for line in lines
    )
    has_provision_body = bool(re.search(r"(?m)^\s*\d{1,4}\s*[.．)]\s+[A-Z]", text)) and bool(
        re.search(r"(?m)^\s*\([0-9a-zA-Z]+\)\s+", text)
    )
    return listing_lines / len(lines) >= 0.65 and not has_provision_body


def decide_chunk(
    chunk: dict[str, Any],
    *,
    body_start_page: int | None,
    ocr_review_pages: set[int] | None = None,
    warning_codes: set[str] | None = None,
    same_text_same_span_duplicate: bool = False,
) -> Decision:
    text = chunk.get("text")
    if not isinstance(text, str) or not text.strip():
        return Decision("rejected", ("empty_or_whitespace_chunk",))

    warnings = warning_codes or set()
    quarantine = []
    flags = []
    start_page, end_page = chunk.get("start_page"), chunk.get("end_page")
    if body_start_page is not None and isinstance(start_page, int) and start_page < body_start_page:
        quarantine.append("front_matter_or_toc_before_detected_body")
    if _toc_only_text(text):
        quarantine.append("toc_only_chunk")
    if warnings & QUARANTINE_WARNING_CODES:
        quarantine.extend(sorted(warnings & QUARANTINE_WARNING_CODES))
    if same_text_same_span_duplicate:
        quarantine.append("exact_duplicate_text_same_source_span_review")
    if ocr_review_pages and isinstance(start_page, int) and isinstance(end_page, int):
        if any(start_page <= page <= end_page for page in ocr_review_pages):
            quarantine.append("overlaps_low_text_ocr_review_page")

    if "repeated_section_heading" in warnings:
        # The source audit showed these include real repeated section chunks,
        # amendment notes, cross-references and schedules. Preserve them.
        flags.append("repeated_heading_warning_preserved")
    if "short_classified_chunk" in warnings:
        flags.append("short_chunk_preserved_for_legal_review")
    if chunk.get("classification_status") == "unclassified":
        flags.append("unclassified_structure_preserved")

    if quarantine:
        return Decision("quarantined", tuple(dict.fromkeys(quarantine)), tuple(flags))
    return Decision("accepted", (), tuple(flags))


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise QualitySelectionError(f"Could not read JSON artifact {path.name}: {type(exc).__name__}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_quality_selection(
    output_dir: str | Path,
    *,
    selection_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Partition all non-excluded manifest chunks and verify exact ordering.

    Original files are read-only. Every chunk appears in exactly one derived
    accepted/rejected/quarantined artifact.
    """
    source_dir = Path(output_dir)
    target = Path(selection_dir) if selection_dir else source_dir / "quality_selection"
    batch = _read_json(source_dir / "drive_batch_report.json")
    audit = _read_json(source_dir / "drive_validation_results.json")
    batch_docs = {row["drive_file_id"]: row for row in batch.get("documents", [])}
    audit_docs = {row["drive_file_id"]: row for row in audit.get("documents", [])}
    expected = set(batch_docs)
    if not set(INTENTIONALLY_EXCLUDED).issubset(expected):
        raise QualitySelectionError("The manifest does not contain both recorded intentional exclusions.")
    eligible = expected - set(INTENTIONALLY_EXCLUDED)
    if set(audit_docs) != expected:
        raise QualitySelectionError("Source-validation document IDs do not exactly match the batch manifest.")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    per_document = []
    reconstruction_checks = 0

    for file_id, document in batch_docs.items():
        if file_id in INTENTIONALLY_EXCLUDED:
            continue
        audit_doc = audit_docs[file_id]
        chunk_file = document.get("chunk_file")
        if not chunk_file or not (source_dir / chunk_file).is_file():
            raise QualitySelectionError(f"Missing non-excluded chunk artifact for Drive file ID {file_id}.")
        if audit_doc.get("exact_reconstruction") is not True or audit_doc.get("page_provenance_checked") is not True:
            raise QualitySelectionError(f"Source reconstruction/provenance did not pass for {file_id}.")
        chunks = _read_json(source_dir / chunk_file)
        if not isinstance(chunks, list) or len(chunks) != document.get("chunk_count"):
            raise QualitySelectionError(f"Chunk count or JSON shape differs from manifest for {file_id}.")
        if len({c.get("chunk_index") for c in chunks}) != len(chunks):
            raise QualitySelectionError(f"Duplicate chunk_index values found for {file_id}.")
        if any(c.get("document_id") != file_id for c in chunks):
            raise QualitySelectionError(f"Chunk source IDs do not match the manifest for {file_id}.")

        warning_by_index: dict[int, set[str]] = defaultdict(set)
        for issue in audit_doc.get("suspicious_chunks", []):
            for reason in issue.get("reasons", []):
                index = issue.get("chunk_index")
                if isinstance(index, int):
                    warning_by_index[index].add(reason.get("code", "unknown_warning"))

        text_spans: dict[tuple[str, int | None, int | None], int] = {}
        duplicate_flags = set()
        for chunk in chunks:
            key = (chunk.get("text", ""), chunk.get("start_page"), chunk.get("end_page"))
            if key in text_spans:
                duplicate_flags.add(chunk.get("chunk_index"))
            else:
                text_spans[key] = chunk.get("chunk_index")

        ocr_pages = {
            value if isinstance(value, int) else value.get("page_number")
            for value in document.get("ocr_review_pages", [])
        }
        ocr_pages.discard(None)
        source = {
            "source_drive_file_id": file_id,
            "document_title": document.get("drive_name"),
            "original_chunk_file": chunk_file,
        }
        original_text = "".join(str(c.get("text", "")) for c in sorted(chunks, key=lambda x: x["chunk_index"]))
        selected_text = []
        status_counts = Counter()
        per_doc_reasons = Counter()
        for chunk in chunks:
            index = chunk["chunk_index"]
            decision = decide_chunk(
                chunk,
                body_start_page=document.get("body_start_page"),
                ocr_review_pages=ocr_pages,
                warning_codes=warning_by_index.get(index, set()),
                same_text_same_span_duplicate=index in duplicate_flags,
            )
            entry = {**chunk, **source, "selection_status": decision.status}
            if decision.reasons:
                entry["selection_reasons"] = list(decision.reasons)
                per_doc_reasons.update(decision.reasons)
            if decision.review_flags:
                entry["review_flags"] = list(decision.review_flags)
            {"accepted": accepted, "rejected": rejected, "quarantined": quarantined}[decision.status].append(entry)
            status_counts[decision.status] += 1
            selected_text.append(entry["text"])
        combined = "".join(selected_text)
        if combined != original_text:
            raise QualitySelectionError(f"Selection altered or reordered source text for {file_id}.")
        reconstruction_checks += 1
        per_document.append({
            **source,
            "pages": document.get("pdf_pages"),
            "chunks": len(chunks),
            "accepted": status_counts["accepted"],
            "rejected": status_counts["rejected"],
            "quarantined": status_counts["quarantined"],
            "exact_partition_reconstruction": True,
            "reason_counts": dict(per_doc_reasons),
        })

    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "accepted_chunks.json", {"chunks": accepted})
    _write_json(target / "rejected_chunks.json", {"chunks": rejected})
    _write_json(target / "quarantined_chunks.json", {"chunks": quarantined})
    excluded_docs = [
        {
            "source_drive_file_id": file_id,
            "document_title": batch_docs[file_id].get("drive_name"),
            "reason": reason,
            "chunk_artifact_expected": False,
        }
        for file_id, reason in INTENTIONALLY_EXCLUDED.items()
    ]
    reason_counts = Counter()
    for rows in (rejected, quarantined):
        for row in rows:
            reason_counts.update(row.get("selection_reasons", []))
    report = {
        "source_pdf_count_in_manifest": len(expected),
        "intentionally_excluded_pdf_count": len(excluded_docs),
        "eligible_pdf_count": len(eligible),
        "eligible_pdf_count_source_verified": reconstruction_checks,
        "accepted_chunk_count": len(accepted),
        "rejected_chunk_count": len(rejected),
        "quarantined_chunk_count": len(quarantined),
        "partitioned_chunk_count": len(accepted) + len(rejected) + len(quarantined),
        "all_partitions_preserve_original_chunk_text_and_order": reconstruction_checks == len(eligible),
        "ocr_performed": False,
        "ocr_note": "Low-text pages are quarantined for review; OCR has not been run.",
        "selection_policy": {
            "whitespace_only": "rejected from embedding; preserved in rejected artifact",
            "front_matter_toc_and_uncertain_metadata": "quarantined, not discarded",
            "repeated_section_numbers_and_headings": "retained; flagged only because repetitions may be amendments or references",
            "missing_structural_metadata": "does not by itself cause rejection or quarantine",
            "duplicate_text": "only exact text repeated on the same source page span is quarantined; repeated text on different pages is retained",
        },
        "selection_reason_counts": dict(reason_counts),
        "intentionally_excluded_documents": excluded_docs,
        "documents": per_document,
    }
    _write_json(target / "selection_report.json", report)
    lines = [
        "# Legal corpus quality selection",
        "",
        f"- PDFs in source manifest: {len(expected)}",
        f"- Intentionally excluded PDFs: {len(excluded_docs)}",
        f"- Eligible PDFs source-verified: {reconstruction_checks}/{len(eligible)}",
        f"- Accepted chunks: {len(accepted)}",
        f"- Rejected chunks: {len(rejected)}",
        f"- Quarantined chunks: {len(quarantined)}",
        f"- Text and order partition checks: {reconstruction_checks}/{len(eligible)}",
        "- OCR performed: no",
        "",
        "## Intentional document exclusions",
        "",
    ]
    lines.extend(f"- `{row['document_title']}` ({row['source_drive_file_id']}): {row['reason']}" for row in excluded_docs)
    lines += ["", "## Selection reasons", ""]
    lines.extend(f"- `{reason}`: {count}" for reason, count in sorted(reason_counts.items()))
    lines += [
        "",
        "Repeated section-like headings, short legal text, and missing structural metadata were not used as grounds to discard chunks. Quarantined content remains in its separate output with original text and provenance. Rejected whitespace-only chunks are also retained in the rejected output. Original chunk files were read-only.",
        "",
    ]
    (target / "selection_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="services/rag/processed/legal_chunking")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run_quality_selection(args.input_dir, selection_dir=args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "documents" and k != "selection_policy"}, ensure_ascii=False, indent=2))
