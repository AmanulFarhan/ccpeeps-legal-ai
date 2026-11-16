"""Chunk PDFs from a Google Drive folder one at a time, retaining outputs locally."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone

from services.rag.ingestion.legal_chunker import ChunkingConfig, chunk_document
from services.rag.ingestion.drive_client import DriveClient, load_env_file
from services.rag.ingestion.pdf_extractor import extract_pdf_text
from services.rag.ingestion.document_mapping import infer_jurisdiction_from_title

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(os.getenv("LEGAL_CHUNK_OUTPUT_DIR", ROOT / "services/rag/processed/legal_chunking"))
ENV_FILE = ROOT / ".env.local"
def _load_local_env() -> None:
    load_env_file(ENV_FILE)


def _output_stem(name: str, file_id: str) -> str:
    stem = Path(name).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "document"
    return f"{safe}--{file_id}"


def run() -> dict:
    _load_local_env()
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
    if not folder_id:
        raise RuntimeError(
            "Set GOOGLE_DRIVE_FOLDER_ID in .env.local to the folder containing the PDFs "
            "(a top-level folder is fine; subfolders are scanned recursively)."
        )

    # Run PoC regression tests before using it on the Drive collection.
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "pytest", "services/rag/test/test_legal_chunker.py"],
        cwd=ROOT,
        check=True,
    )
    drive = DriveClient.from_env(env_file=ENV_FILE)
    pdfs, discovery_errors = drive.discover_pdfs(folder_id)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    force_reprocess = os.getenv("GOOGLE_DRIVE_FORCE_REPROCESS", "").lower() in {"1", "true", "yes"}
    results: list[dict] = []
    checkpoint_path = OUTPUT_DIR / "drive_batch_checkpoint.json"
    prior_path = OUTPUT_DIR / "drive_batch_report.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    prior_rows = {row.get("drive_file_id"): row for row in [*prior.get("documents", []), *checkpoint.get("documents", [])]}
    for position, item in enumerate(pdfs, start=1):
        record = {
            "drive_file_id": item["id"],
            "drive_name": item["name"],
            "drive_web_view_link": item.get("webViewLink"),
            "drive_size_bytes": item.get("size"),
            "drive_md5_checksum": item.get("md5Checksum"),
            "drive_modified_time": item.get("modifiedTime"),
            "status": "error",
        }
        output_name = _output_stem(item["name"], item["id"]) + "_chunks.json"
        output_path = OUTPUT_DIR / output_name
        old = prior_rows.get(item["id"], {})
        same_version = (
            item.get("md5Checksum") is not None
            and old.get("drive_md5_checksum") == item.get("md5Checksum")
        ) or (
            item.get("md5Checksum") is None
            and item.get("modifiedTime") is not None
            and old.get("drive_modified_time") == item.get("modifiedTime")
            and str(old.get("drive_size_bytes")) == str(item.get("size"))
        )
        if output_path.is_file() and same_version and not force_reprocess:
            # Reuse only when the saved manifest proves the Drive version is unchanged.
            chunks = json.loads(output_path.read_text(encoding="utf-8"))
            record.update(
                {
                    **old,
                    "status": "processed",
                    "chunk_file": output_name,
                    "resumed_existing_output": True,
                }
            )
            results.append(record)
            print(f"[{position}/{len(pdfs)}] Reusing {item['name']}", flush=True)
            checkpoint_path.write_text(json.dumps({"documents": results}, ensure_ascii=False, indent=2), encoding="utf-8")
            continue
        try:
            print(f"[{position}/{len(pdfs)}] Downloading and processing {item['name']}", flush=True)
            with drive.temporary_pdf(item["id"]) as pdf_path:
                pages = extract_pdf_text(str(pdf_path))
                chunks, analysis = chunk_document(pages, item["id"], ChunkingConfig(max_words=1200))
                source_name = item["name"]
                source_lower = source_name.casefold()
                document_type = "rules" if "rule" in source_lower else "act" if "act" in source_lower else "government_order" if "order" in source_lower else "unknown"
                jurisdiction = infer_jurisdiction_from_title(source_name)
                source_metadata = {
                    "provider": "google_drive", "file_id": item["id"], "file_name": source_name,
                    "web_view_link": item.get("webViewLink"),
                    "modified_time": item.get("modifiedTime"), "md5_checksum": item.get("md5Checksum"),
                    "metadata_inference": "Jurisdiction inferred from title prefix: Kerala => state; otherwise central. Document type remains separately unverified.",
                }
                for chunk in chunks:
                    chunk.update({
                        "document_title": Path(source_name).stem,
                        "document_type": document_type,
                        "jurisdiction": jurisdiction,
                        "original_source": source_metadata,
                        "metadata_inference": source_metadata["metadata_inference"],
                    })
                original = "\n".join(page["text"] for page in pages)
                reconstructed = "".join(chunk["text"] for chunk in chunks)
                if reconstructed != original:
                    raise AssertionError(
                        f"Exact text reconstruction failed ({len(original)} source vs "
                        f"{len(reconstructed)} reconstructed characters)."
                    )
                output_path.write_text(
                    json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                classified = sum(c["classification_status"] == "classified" for c in chunks)
                word_counts = [c["word_count"] for c in chunks]
                record.update(
                    {
                        "status": "processed",
                        "chunk_file": output_name,
                        "pdf_pages": len(pages),
                        "body_start_page": analysis["body_start_page"],
                        "parts": analysis["parts"],
                        "chapters": analysis["chapters"],
                        "sections": analysis["sections"],
                        "subsections": analysis["subsections"],
                        "clauses": analysis["clauses"],
                        "chunk_count": len(chunks),
                        "classified_chunks": classified,
                        "unclassified_chunks": len(chunks) - classified,
                        "average_chunk_words": round(sum(word_counts) / len(word_counts), 1) if word_counts else 0,
                        "maximum_chunk_words": max(word_counts, default=0),
                        "exact_reconstruction": True,
                        "source_characters": len(original),
                        "ocr_review_pages": analysis["ocr_review_pages"],
                    }
                )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        results.append(record)
        checkpoint_path.write_text(json.dumps({"documents": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{position}/{len(pdfs)}] {record['status']}: {item['name']}", flush=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_folder_id": folder_id,
        "output_directory": str(OUTPUT_DIR.relative_to(ROOT)),
        "pdfs_discovered": len(pdfs),
        "processed": sum(row["status"] == "processed" for row in results),
        "failed": sum(row["status"] != "processed" for row in results),
        "total_pdf_pages": sum(row.get("pdf_pages", 0) for row in results),
        "total_chunks": sum(row.get("chunk_count", 0) for row in results),
        "classified_chunks": sum(row.get("classified_chunks", 0) for row in results),
        "unclassified_chunks": sum(row.get("unclassified_chunks", 0) for row in results),
        "exact_reconstruction_documents": sum(row.get("exact_reconstruction") is True for row in results),
        "documents_without_detected_sections": [row["drive_name"] for row in results if row.get("sections") == 0],
        "discovery_errors": discovery_errors,
        "documents": results,
        "ocr_note": "OCR was not performed. Reported pages are only low-extracted-text candidates.",
    }
    (OUTPUT_DIR / "drive_batch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Google Drive legal PDF chunking batch",
        "",
        f"- PDFs discovered: {report['pdfs_discovered']}",
        f"- Processed: {report['processed']}; failed: {report['failed']}",
        f"- PDF pages: {report['total_pdf_pages']}; chunks: {report['total_chunks']} "
        f"(classified: {report['classified_chunks']}; unclassified: {report['unclassified_chunks']})",
        f"- Exact reconstruction verified: {report['exact_reconstruction_documents']}/{report['processed']} documents",
        f"- Documents with no detected sections: {len(report['documents_without_detected_sections'])}",
        "- Files were downloaded one at a time to temporary storage and removed after processing.",
        "- Outputs are local; nothing was uploaded to Supabase.",
        "- OCR was not performed.",
        "",
        "| File | Status | Pages | Body start | Parts / Chapters / Sections | Chunks (classified / unclassified) | Avg / max words | Exact reconstruction | OCR review pages |",
        "|---|---|---:|---:|---|---|---:|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['drive_name'].replace('|', '\\|')} | {row['status']} | "
            f"{row.get('pdf_pages', '')} | {row.get('body_start_page', '')} | "
            f"{row.get('parts', '')} / {row.get('chapters', '')} / {row.get('sections', '')} | "
            f"{row.get('chunk_count', '')} ({row.get('classified_chunks', '')} / {row.get('unclassified_chunks', '')}) | "
            f"{row.get('average_chunk_words', '')} / {row.get('maximum_chunk_words', '')} | "
            f"{row.get('exact_reconstruction', False)} | {row.get('ocr_review_pages', row.get('error', ''))} |"
        )
    (OUTPUT_DIR / "drive_batch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
