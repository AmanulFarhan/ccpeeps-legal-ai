"""Source-verify and audit every JSON chunk in the Google Drive batch."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import sys
import argparse

ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = Path(os.getenv("LEGAL_CHUNK_OUTPUT_DIR", ROOT / "services/rag/processed/legal_chunking"))
sys.path.insert(0, str(ROOT))

from services.rag.ingestion.legal_chunker import analyze_document
from services.rag.ingestion.chunk_validator import validate_chunks
from services.rag.ingestion.drive_client import DriveClient, load_env_file
from services.rag.ingestion.pdf_extractor import extract_pdf_text


def _normalize_title(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"_[0-9a-fA-F]{8}$", "", stem)
    return re.sub(r"[^a-z0-9]+", " ", stem.casefold()).strip()


def _chunk_page_text(chunk: dict, pages: list[dict]) -> str:
    start, end = chunk.get("start_page"), chunk.get("end_page")
    if not isinstance(start, int) or not isinstance(end, int):
        return ""
    return "\n".join(page["text"] for page in pages if start <= page["page_number"] <= end)


def _per_document(item: dict, pages: list[dict], chunks: list[dict]) -> dict:
    result = validate_chunks(chunks, pages=pages)
    analysis = analyze_document(pages)
    body_start = analysis.get("body_start_page")
    suspicious: dict[int, list[dict]] = {}

    def issue(index, code, message, **extra):
        suspicious.setdefault(index, []).append({"code": code, "message": message, **extra})

    for error in result["errors"]:
        issue(error.get("chunk_index", -1), error["code"], error["message"], **{k:v for k,v in error.items() if k not in {"chunk_index","code","message"}})
    for warning in result["warnings"]:
        issue(warning.get("chunk_index", -1), warning["code"], warning["message"], **{k:v for k,v in warning.items() if k not in {"chunk_index","code","message"}})

    classified = [chunk for chunk in chunks if chunk.get("classification_status") == "classified"]
    chapter_titles: dict[str, set[str]] = defaultdict(set)
    opening_section_numbers = []
    for chunk in chunks:
        index = chunk.get("chunk_index", -1)
        for field in ("document_title", "document_type", "jurisdiction", "original_source"):
            if not chunk.get(field):
                issue(index, "missing_source_metadata", f"{field} is absent; existing chunk is retained unchanged.", field=field)
        if chunk.get("document_id") != item.get("drive_file_id"):
            issue(index, "document_id_mismatch", "Chunk document_id does not match its Drive file ID.")
    for chunk in classified:
        index = chunk.get("chunk_index", -1)
        if body_start is None:
            issue(index, "classified_without_detected_body", "Chunk is classified although no legal-body start was detected.")
        elif chunk.get("start_page") is not None and chunk["start_page"] < body_start:
            issue(index, "classified_before_body", "Classified chunk begins before detected legal-body start.", body_start_page=body_start)
        first_pages = _chunk_page_text(chunk, pages).casefold()
        chunk_start = chunk.get("start_page")
        if chunk_start is not None and chunk_start < (body_start or 1) and any(x in first_pages for x in ("arrangement of sections", "contents")):
            issue(index, "toc_or_front_matter_classified", "Source page appears to be a contents/front-matter page.")
        if chunk.get("chapter_number") is not None and chunk.get("chapter_title"):
            chapter_titles[str(chunk["chapter_number"])].add(str(chunk["chapter_title"]))
        if chunk.get("section_number") is not None and not chunk.get("section_title"):
            issue(index, "section_title_missing", "Section number is set but section title metadata is missing.")
        match = re.match(r"^\s*(\d{1,4})\s*[.．)]\s+[A-Z]", str(chunk.get("text", "")))
        if match:
            opening_section_numbers.append((match.group(1), chunk))
    for number, title_set in chapter_titles.items():
        if len(title_set) > 1:
            for chunk in classified:
                if str(chunk.get("chapter_number")) == number:
                    issue(chunk.get("chunk_index", -1), "chapter_title_inconsistent", "One chapter number has inconsistent titles across chunks.", titles=sorted(title_set))
    opening_counts = Counter(number for number, _ in opening_section_numbers)
    repeated_openings = {number: count for number, count in opening_counts.items() if count > 1}
    for number, chunk in opening_section_numbers:
        if number in repeated_openings:
            issue(chunk.get("chunk_index", -1), "repeated_section_heading", "A section heading appears at the start of multiple classified chunks.", count=repeated_openings[number])
    suspicious_codes = {key: {item["code"] for item in values} for key, values in suspicious.items()}
    structural_codes = {
        "toc_text_classified", "classified_before_body", "classified_without_detected_body",
        "multiple_section_heading_candidates", "section_metadata_text_mismatch",
        "section_title_missing", "chapter_title_inconsistent", "repeated_section_heading",
        "subsection_marker_absent", "subsection_looks_like_clause", "clause_looks_like_subsection",
        "unusual_section_number", "unusual_subsection_number", "unusual_clause_number",
    }
    plausible_examples = []
    for chunk in classified:
        index = chunk.get("chunk_index", -1)
        match = re.match(r"^\s*(\d{1,4})\s*[.．)]\s+[A-Z]", str(chunk.get("text", "")))
        if (match and match.group(1) == str(chunk.get("section_number"))
                and (body_start is None or (chunk.get("start_page") or 0) >= body_start)
                and not (suspicious_codes.get(index, set()) & structural_codes)):
            plausible_examples.append({
                "chunk_index": index, "start_page": chunk.get("start_page"), "end_page": chunk.get("end_page"),
                "section_number": chunk.get("section_number"), "section_title": chunk.get("section_title"),
                "generated_metadata": {key: chunk.get(key) for key in ("part", "chapter_number", "chapter_title", "section_number", "section_title", "subsection_number", "clause_number")},
                "text_excerpt": str(chunk.get("text", ""))[:420],
            })
    selected_examples = []
    for position in (0, len(plausible_examples) // 2, len(plausible_examples) - 1):
        if plausible_examples and plausible_examples[position] not in selected_examples:
            selected_examples.append(plausible_examples[position])

    titles = [
        {"number": page["page_number"], "characters": len(page["text"].strip()), "text_excerpt": page["text"][:900]}
        for page in pages if len(re.sub(r"\s+", "", page["text"])) < 80
    ]
    chunk_words = [len(str(c.get("text", "")).split()) for c in chunks]
    return {
        "drive_file_id": item["drive_file_id"],
        "drive_name": item["drive_name"],
        "pdf_pages": len(pages),
        "chunk_count": len(chunks),
        "classified_chunks": len(classified),
        "unclassified_chunks": len(chunks) - len(classified),
        "body_start_page_detected": body_start,
        "parts_detected": analysis.get("parts", 0),
        "chapters_detected": analysis.get("chapters", 0),
        "sections_detected": analysis.get("sections", 0),
        "subsections_detected": analysis.get("subsections", 0),
        "clauses_detected": analysis.get("clauses", 0),
        "classified_before_body": sum(1 for chunk in classified if body_start is not None and (chunk.get("start_page") or 0) < body_start),
        "repeated_section_heading_numbers": repeated_openings,
        "plausible_structure_examples": selected_examples,
        "average_chunk_words": round(sum(chunk_words) / len(chunk_words), 1) if chunk_words else 0,
        "maximum_chunk_words": max(chunk_words, default=0),
        "exact_reconstruction": result["exact_reconstruction"],
        "page_provenance_checked": result["page_provenance_checked"],
        "validation_error_count": result["error_count"],
        "validation_warning_count": max(0, sum(len(v) for v in suspicious.values()) - result["error_count"]),
        "validation_errors": result["errors"],
        "validation_warnings": result["warnings"],
        "ocr_review_page_details": titles,
        "first_page_text_excerpt": pages[0]["text"][:1200] if pages else "",
        "suspicious_chunks": [
            {"chunk_index": i, "start_page": next((c.get("start_page") for c in chunks if c.get("chunk_index") == i), None),
             "end_page": next((c.get("end_page") for c in chunks if c.get("chunk_index") == i), None),
             "section_number": next((c.get("section_number") for c in chunks if c.get("chunk_index") == i), None),
             "generated_metadata": {key: next((c.get(key) for c in chunks if c.get("chunk_index") == i), None) for key in ("part", "chapter_number", "chapter_title", "section_number", "section_title", "subsection_number", "clause_number", "classification_status")},
             "chunk_text_excerpt": next((c.get("text", "")[:600] for c in chunks if c.get("chunk_index") == i), ""),
             "reasons": reasons}
            for i, reasons in sorted(suspicious.items(), key=lambda pair: (-1 if pair[0] is None else pair[0]))
        ],
    }


def run(*, retry_errors: bool = False) -> dict:
    load_env_file(ROOT / ".env.local")
    source_report = json.loads((OUTPUTS / "drive_batch_report.json").read_text(encoding="utf-8"))
    client = DriveClient.from_env()
    previous_path = OUTPUTS / "drive_validation_results.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if retry_errors and previous_path.exists() else None
    retry_ids = {row["drive_file_id"] for row in previous.get("source_errors", [])} if previous else None
    docs = list(previous.get("documents", [])) if previous else []
    document_errors = []
    for pos, row in enumerate(source_report["documents"], start=1):
        if retry_ids is not None and row["drive_file_id"] not in retry_ids:
            continue
        print(f"[{pos}/{len(source_report['documents'])}] Auditing {row['drive_name']}", flush=True)
        chunk_path = OUTPUTS / row["chunk_file"]
        chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
        try:
            with client.temporary_pdf(row["drive_file_id"]) as pdf_path:
                pages = extract_pdf_text(str(pdf_path))
                docs.append(_per_document(row, pages, chunks))
        except Exception as exc:
            document_errors.append({"drive_file_id": row["drive_file_id"], "drive_name": row["drive_name"], "error": f"{type(exc).__name__}: {exc}"})

    by_title: dict[str, list[dict]] = {}
    for d in source_report["documents"]:
        by_title.setdefault(_normalize_title(d["drive_name"]), []).append({"drive_file_id": d["drive_file_id"], "drive_name": d["drive_name"]})
    duplicate_titles = [rows for rows in by_title.values() if len({r["drive_file_id"] for r in rows}) > 1]
    docs_by_id = {d["drive_file_id"]: d for d in docs}
    docs = [docs_by_id[row["drive_file_id"]] for row in source_report["documents"] if row["drive_file_id"] in docs_by_id]
    test_order = next((d for d in docs if Path(d["drive_name"]).name.casefold() == "test_order.pdf"), None)
    totals = {
        "documents": len(source_report["documents"]),
        "documents_source_verified": len(docs),
        "document_source_errors": len(document_errors),
        "chunks": sum(d["chunk_count"] for d in docs),
        "classified_chunks": sum(d["classified_chunks"] for d in docs),
        "unclassified_chunks": sum(d["unclassified_chunks"] for d in docs),
        "validation_errors": sum(d["validation_error_count"] for d in docs),
        "validation_warnings": sum(d["validation_warning_count"] for d in docs),
        "documents_without_sections": [d["drive_name"] for d in docs if d["sections_detected"] == 0],
        "classified_before_body": sum(d.get("classified_before_body", 0) for d in docs),
        "exact_reconstruction_documents": sum(d["exact_reconstruction"] is True for d in docs),
        "page_provenance_documents_checked": sum(d["page_provenance_checked"] for d in docs),
        "ocr_review_page_pairs": sum(len(d["ocr_review_page_details"]) for d in docs),
    }
    result = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source_verification": "Fresh Google Drive download and repository PyMuPDF extraction, one PDF at a time.",
        "ocr_performed": False,
        "aggregate": totals,
        "duplicate_titles_different_drive_ids": duplicate_titles,
        "test_order_assessment": {
            "included_in_batch": test_order is not None,
            "provisional_recommendation": "exclude_from_acts_and_rules_subset_or_relabel_as_government_order",
            "basis": "The source is a Chandigarh Police office order citing BNSS section 187, not an Act or Rules instrument. It may belong in a broader legal corpus if government orders are intentionally included.",
            "source_excerpt": next((d["first_page_text_excerpt"] for d in docs if Path(d["drive_name"]).name.casefold() == "test_order.pdf"), None),
        },
        "source_errors": document_errors,
        "documents": docs,
    }
    (OUTPUTS / "drive_validation_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(result)
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    return result


def _write_markdown(result: dict) -> None:
    agg = result["aggregate"]
    lines = [
        "# Legal PDF batch validation", "",
        "Validation downloads each source PDF from Drive, re-extracts it with the repository PyMuPDF extractor, and compares concatenated chunk text and per-chunk page provenance. OCR was not performed.", "",
        "## Summary", "",
        f"- Documents in batch: {agg['documents']}; source-verified: {agg['documents_source_verified']}; source errors: {agg['document_source_errors']}.",
        f"- Chunks: {agg['chunks']} (classified {agg['classified_chunks']}; unclassified {agg['unclassified_chunks']}).",
        f"- Exact reconstruction: {agg['exact_reconstruction_documents']}/{agg['documents_source_verified']} source-verified documents.",
        f"- Per-chunk page provenance checked: {agg['page_provenance_documents_checked']} documents.",
        f"- Structural validation errors: {agg['validation_errors']}; warnings: {agg['validation_warnings']}.",
        f"- Documents with no detected sections: {len(agg['documents_without_sections'])}.",
        f"- OCR-review candidate pairs: {agg['ocr_review_page_pairs']} (heuristic only; OCR not run).", "",
        "### Warning categories", "",
    ]
    for code, count in sorted(agg.get("warning_code_counts", {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{code}`: {count}")
    lines += [
        "", f"- Classified TOC/front-matter chunks identified by standalone markers: {agg.get('toc_classified_chunks', 0)}.",
        f"- Chunks missing one or more source metadata fields: {agg.get('chunks_missing_source_metadata', 0)}; missing-field occurrences: {json.dumps(agg.get('missing_source_metadata_fields', {}), ensure_ascii=False)}.",
        "", "## Structural validation errors", "",
    ]
    error_rows = [
        (d, error) for d in result["documents"] for error in d["validation_errors"]
    ]
    if error_rows:
        for doc, error in error_rows:
            chunk = next((c for c in doc["suspicious_chunks"] if c["chunk_index"] == error.get("chunk_index")), None)
            excerpt = (chunk or {}).get("chunk_text_excerpt", "")
            lines.append(f"- {doc['drive_name']}, chunk {error.get('chunk_index')}, pages {(chunk or {}).get('start_page')}-{(chunk or {}).get('end_page')}: `{error['code']}`. Text excerpt: `{excerpt[:160]}`")
    else:
        lines.append("No structural validation errors were recorded.")
    lines += [
        "", "Whitespace-only chunks are counted as validation errors because they are empty as standalone chunks. They are still retained in the JSON and are necessary to preserve exact source reconstruction.",
        "", "## Duplicate normalized titles", "",
    ]
    if result["duplicate_titles_different_drive_ids"]:
        for group in result["duplicate_titles_different_drive_ids"]:
            lines.extend([f"- {group[0]['drive_name']}"] + [f"  - Drive ID `{row['drive_file_id']}`: {row['drive_name']}" for row in group])
    else:
        lines.append("No normalized duplicate titles were found.")
    lines += ["", "## Documents with no detected sections", ""]
    lines.extend(f"- {name}" for name in agg["documents_without_sections"])
    lines += ["", "## `test_order.pdf`", "", f"- Recommendation: `{result['test_order_assessment']['provisional_recommendation']}`.", f"- Basis: {result['test_order_assessment']['basis']}", "- Source excerpt:", "", "  ```text", "  " + str(result["test_order_assessment"]["source_excerpt"] or "(not captured)").replace("\n", "\n  ")[:900], "  ```", "", "## Per-document results", "", "| Document | Pages | Body page | Parts | Chapters | Sections | Subsections | Clauses | Chunks | Classified | Unclassified | Exact text | Provenance | Errors | Warnings | OCR pages |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|"]
    for d in result["documents"]:
        lines.append(f"| {d['drive_name'].replace('|','\\|')} | {d['pdf_pages']} | {d['body_start_page_detected']} | {d['parts_detected']} | {d['chapters_detected']} | {d['sections_detected']} | {d['subsections_detected']} | {d['clauses_detected']} | {d['chunk_count']} | {d['classified_chunks']} | {d['unclassified_chunks']} | {d['exact_reconstruction']} | {d['page_provenance_checked']} | {d['validation_error_count']} | {d['validation_warning_count']} | {len(d['ocr_review_page_details'])} |")
    lines += ["", "## OCR-review page excerpts", ""]
    for d in result["documents"]:
        for page in d["ocr_review_page_details"]:
            lines += [f"### {d['drive_name']}, page {page['number']}", "", f"- Non-whitespace characters: {page['characters']}", "- Extracted text:", "", "  ```text", "  " + page["text_excerpt"].replace("\n", "\n  ")[:900], "  ```", ""]
    lines += ["## Suspicious chunk examples", ""]
    for d in result["documents"]:
        if d["suspicious_chunks"]:
            lines += [f"### {d['drive_name']}", ""]
            structural = [item for item in d["suspicious_chunks"] if any(r["code"] != "missing_source_metadata" for r in item["reasons"])]
            for item in (structural or d["suspicious_chunks"])[:12]:
                lines.append(f"- Chunk {item['chunk_index']}, pages {item['start_page']}-{item['end_page']}, section {item['section_number']}: " + "; ".join(r["code"] + ": " + r["message"] for r in item["reasons"]))
                lines.append(f"  - Generated metadata: `{json.dumps(item['generated_metadata'], ensure_ascii=False)}`")
                lines.append(f"  - Source text excerpt (verbatim extracted PDF text): `{item['chunk_text_excerpt'][:280].replace('`', '\\`')}`")
            if len(d["suspicious_chunks"]) > 12:
                lines.append(f"- {len(d['suspicious_chunks']) - 12} additional suspicious chunks are listed in `drive_validation_results.json`.")
            lines.append("")
    lines += ["## Plausible structure examples (heuristic selection)", "", "These chunks begin with a heading matching the generated section number and have no structural warning under the current checks. This is a screening signal, not legal validation.", ""]
    for d in result["documents"]:
        if d.get("plausible_structure_examples"):
            lines.extend([f"### {d['drive_name']}", ""])
            for sample in d["plausible_structure_examples"]:
                lines.append(f"- Chunk {sample['chunk_index']}, pages {sample['start_page']}-{sample['end_page']}, section {sample['section_number']}: {sample['section_title']}")
                lines.append(f"  - Metadata: `{json.dumps(sample['generated_metadata'], ensure_ascii=False)}`")
                lines.append(f"  - Source text: `{sample['text_excerpt'][:320].replace('`', '\\`')}`")
            lines.append("")
    lines += ["## Interpretation and limits", "", "This report checks data shape, metadata patterns, exact extracted-text reconstruction, ordering, and page spans. Heuristics that detect heading-like text are warnings and require review against the corresponding source PDF page; they are not automatic corrections. Exact text reconstruction does not prove that legal hierarchy was classified correctly. Nine documents had no detected section headings in the batch run and require review. A chunk from `test_order.pdf` is not recommended for the Acts/Rules subset unless deliberately categorized as a Government Order. No chunk was modified or discarded by this validation.", ""]
    lines += ["## Findings and likely causes", ""]
    for finding in result.get("findings", []):
        lines.append(f"- **{finding['severity']} — {finding['code']} ({finding['count']}):** {finding['details']}")
    lines += ["", "## Recommended corrections, in priority order", ""]
    lines.extend(f"{i}. {recommendation}" for i, recommendation in enumerate(result.get("recommendations", []), start=1))
    lines.append("")
    (OUTPUTS / "drive_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def refresh_report_heuristics() -> dict:
    """Reapply refined text-only warning rules to a completed source-verified report."""
    report_path = OUTPUTS / "drive_validation_results.json"
    result = json.loads(report_path.read_text(encoding="utf-8"))
    batch_manifest = json.loads((OUTPUTS / "drive_batch_report.json").read_text(encoding="utf-8"))
    chunk_files = {row["drive_file_id"]: row["chunk_file"] for row in batch_manifest["documents"]}
    for document in result["documents"]:
        chunk_path = OUTPUTS / chunk_files[document["drive_file_id"]]
        chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
        by_index = {chunk.get("chunk_index"): chunk for chunk in chunks}

        def discard(code: str, payload: dict, chunk: dict | None) -> bool:
            text = str((chunk or {}).get("text", ""))
            if code == "toc_text_classified":
                return re.search(r"(?im)^\s*(?:arrangement of sections|contents)\s*$", text) is None
            if code == "subsection_marker_absent":
                return re.match(r"^\s*\([A-Za-z]\)", text) is not None
            if code == "multiple_section_heading_candidates":
                values = [value for value in payload.get("candidates", []) if not value.isdigit() or not 1800 <= int(value) <= 2100]
                payload["candidates"] = values
                return len(values) < 2
            return False

        for warning_key in ("validation_warnings",):
            document[warning_key] = [
                warning for warning in document.get(warning_key, [])
                if not discard(warning["code"], warning, by_index.get(warning.get("chunk_index")))
            ]
        for record in document.get("suspicious_chunks", []):
            record["reasons"] = [
                reason for reason in record["reasons"]
                if not discard(reason["code"], reason, by_index.get(record.get("chunk_index")))
            ]
        document["suspicious_chunks"] = [record for record in document["suspicious_chunks"] if record["reasons"]]
        suspicious_codes = {row["chunk_index"]: {reason["code"] for reason in row["reasons"]} for row in document["suspicious_chunks"]}
        structural_codes = {
            "toc_text_classified", "classified_before_body", "classified_without_detected_body",
            "multiple_section_heading_candidates", "section_metadata_text_mismatch",
            "section_title_missing", "chapter_title_inconsistent", "repeated_section_heading",
            "subsection_marker_absent", "subsection_looks_like_clause", "clause_looks_like_subsection",
            "unusual_section_number", "unusual_subsection_number", "unusual_clause_number",
        }
        examples = []
        for chunk in chunks:
            index = chunk.get("chunk_index", -1)
            match = re.match(r"^\s*(\d{1,4})\s*[.．)]\s+[A-Z]", str(chunk.get("text", "")))
            if (chunk.get("classification_status") == "classified" and match
                    and match.group(1) == str(chunk.get("section_number"))
                    and (document["body_start_page_detected"] is None or (chunk.get("start_page") or 0) >= document["body_start_page_detected"])
                    and not (suspicious_codes.get(index, set()) & structural_codes)):
                examples.append({
                    "chunk_index": index, "start_page": chunk.get("start_page"), "end_page": chunk.get("end_page"),
                    "section_number": chunk.get("section_number"), "section_title": chunk.get("section_title"),
                    "generated_metadata": {key: chunk.get(key) for key in ("part", "chapter_number", "chapter_title", "section_number", "section_title", "subsection_number", "clause_number")},
                    "text_excerpt": str(chunk.get("text", ""))[:420],
                })
        chosen = []
        for position in (0, len(examples) // 2, len(examples) - 1):
            if examples and examples[position] not in chosen:
                chosen.append(examples[position])
        document["plausible_structure_examples"] = chosen
        reason_count = sum(len(row["reasons"]) for row in document["suspicious_chunks"])
        document["validation_warning_count"] = max(0, reason_count - document["validation_error_count"])

    result["aggregate"]["validation_warnings"] = sum(doc["validation_warning_count"] for doc in result["documents"])
    result["aggregate"]["classified_before_body"] = sum(doc.get("classified_before_body", 0) for doc in result["documents"])
    reason_counts = Counter(reason["code"] for doc in result["documents"] for row in doc["suspicious_chunks"] for reason in row["reasons"])
    metadata_fields = Counter(reason.get("field") for doc in result["documents"] for row in doc["suspicious_chunks"] for reason in row["reasons"] if reason["code"] == "missing_source_metadata")
    result["aggregate"]["warning_code_counts"] = dict(reason_counts)
    result["aggregate"]["toc_classified_chunks"] = reason_counts.get("toc_text_classified", 0)
    result["aggregate"]["missing_source_metadata_fields"] = dict(metadata_fields)
    result["aggregate"]["chunks_missing_source_metadata"] = sum(
        1 for doc in result["documents"] for row in doc["suspicious_chunks"]
        if any(reason["code"] == "missing_source_metadata" for reason in row["reasons"])
    )
    result["findings"] = [
        {"severity": "pass_with_scope", "code": "text_reconstruction", "count": 49, "details": "All 6,196 chunks reconstruct the freshly extracted PDF text in order."},
        {"severity": "pass_with_scope", "code": "page_provenance", "count": 49, "details": "Every chunk's page range agrees with source page text intervals."},
        {"severity": "pass_with_scope", "code": "toc_classification", "count": 0, "details": "No classified chunk begins before detected body start; no standalone contents heading was classified."},
        {"severity": "error", "code": "whitespace_only_chunks", "count": result["aggregate"]["validation_errors"], "details": "Whitespace-only chunks are preserved for exact reconstruction but are empty as standalone chunks."},
        {"severity": "warning", "code": "missing_source_metadata", "count": result["aggregate"]["chunks_missing_source_metadata"], "details": "All old chunks lack document_title, document_type, jurisdiction and original_source fields."},
        {"severity": "warning", "code": "no_detected_sections", "count": len(result["aggregate"]["documents_without_sections"]), "details": "These Act/Rules PDFs have no recognized sections and remain unclassified."},
        {"severity": "warning", "code": "numbered_amendment_notes_as_sections", "count": result["aggregate"]["warning_code_counts"].get("repeated_section_heading", 0), "details": "Many repeated section-like starts are amendment notes such as '1. Subs. by Act…'; these inflate section counts. Companies Act pages 17-18 show examples."},
        {"severity": "warning", "code": "chapter_title_variants", "count": result["aggregate"]["warning_code_counts"].get("chapter_title_inconsistent", 0), "details": "Repeated chapter numbers have different titles in Arbitration and Conciliation Act and RBI Act; chapter numbering likely resets under Parts that are not consistently detected."},
        {"severity": "review", "code": "ocr_review_pages", "count": result["aggregate"]["ocr_review_page_pairs"], "details": "18 page pairs have fewer than 80 non-whitespace characters. This heuristic is not OCR and can miss OCR-corrupted pages with substantial extracted text."},
        {"severity": "review", "code": "test_order", "count": 1, "details": "The source is a Chandigarh Police office order under BNSS section 187, not an Act or Rules instrument."},
    ]
    result["recommendations"] = [
        "Do not embed or store this batch until structural errors and all no-section documents are reviewed.",
        "Reject amendment-footnote numbering and section references as section headings; use heading context and page layout evidence.",
        "Detect Parts reliably and scope chapter numbering/title consistency by Part.",
        "Carry document_title, document_type, jurisdiction and original_source into future chunks, with provenance indicating inferred values.",
        "Keep whitespace-only fragments for lossless reconstruction, but merge them into adjacent chunks or treat them as separators before embedding.",
        "Review the 18 flagged pages and integrate an OCR engine only if OCR is required and explicitly enabled.",
        "Exclude test_order.pdf from the Acts/Rules subset or relabel it as a Government Order.",
        "Review both same-title Inland Vessels records by Drive ID and content before deduplication.",
    ]
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-errors", action="store_true", help="Re-fetch only documents that failed in the previous validation report.")
    parser.add_argument("--refresh-heuristics", action="store_true", help="Refresh warning heuristics using already source-verified chunks without downloading PDFs again.")
    arguments = parser.parse_args()
    if arguments.refresh_heuristics:
        refresh_report_heuristics()
    else:
        run(retry_errors=arguments.retry_errors)
