
from pathlib import Path
from hashlib import sha256
from typing import Any

from services.rag.ingestion.pdf_extractor import extract_pdf_text
from services.rag.ingestion.text_cleaner import clean_extracted_text
from services.rag.ingestion.chunker import chunk_legal_document
from services.rag.ingestion.chunk_validator import validate_chunks
from services.rag.ingestion.document_mapping import infer_jurisdiction_from_title


def process_pdf(pdf_path: str) -> list[dict]:
    """
    Extract and clean a PDF page by page.

    Keeps original text, cleaned text, and page numbers.
    """
    extracted_pages = extract_pdf_text(pdf_path)
    processed_pages = []

    for page in extracted_pages:
        original_text = page["text"]
        cleaned_text = clean_extracted_text(original_text)

        processed_pages.append(
            {
                "page_number": page["page_number"],
                "original_text": original_text,
                "cleaned_text": cleaned_text,
            }
        )

    return processed_pages


def process_document(
    pdf_path: str | Path,
    *,
    document_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    chunk_config=None,
) -> dict[str, Any]:
    """Extract, clean, legally chunk, and validate without embedding or database writes."""
    path = Path(pdf_path)
    extracted = extract_pdf_text(str(path))
    pages = []
    for page in extracted:
        original = page["text"]
        pages.append({
            "page_number": page["page_number"],
            "text": original,
            "original_text": original,
            "cleaned_text": clean_extracted_text(original),
        })
    source_hash = sha256(path.read_bytes()).hexdigest()
    doc_id = document_id or source_hash
    chunks, analysis = chunk_legal_document(pages, doc_id, chunk_config)
    document_metadata = {"title": path.stem, **(metadata or {})}
    source_name = document_metadata.get("source_name", path.name)
    source_lower = str(source_name).casefold()
    document_type = document_metadata.get("document_type")
    if document_type is None:
        document_type = "rules" if "rule" in source_lower else "act" if "act" in source_lower else "government_order" if "order" in source_lower else "unknown"
    jurisdiction = infer_jurisdiction_from_title(document_metadata.get("title", source_name))
    original_source = document_metadata.get("original_source", {
        "provider": "local_file", "source_name": source_name, "source_sha256": source_hash,
    })
    for chunk in chunks:
        chunk.update({
            "document_title": document_metadata.get("title", path.stem),
            "document_type": document_type,
            "jurisdiction": jurisdiction,
            "original_source": original_source,
            "metadata_inference": document_metadata.get("metadata_inference", "filename_or_caller_metadata"),
        })
    validation = validate_chunks(chunks, pages=pages)
    return {
        "document": {"document_id": doc_id, "source_name": path.name, **document_metadata},
        "source_sha256": source_hash,
        "pages": pages,
        "chunks": chunks,
        "analysis": analysis,
        "validation": validation,
        "validated": validation["valid"] and validation["exact_reconstruction"],
        "dry_run": True,
    }


def process_drive_document(drive_client, drive_file: dict[str, Any], *, chunk_config=None) -> dict[str, Any]:
    """Download one Drive PDF temporarily, process it, then remove its local copy."""
    with drive_client.temporary_pdf(drive_file["id"]) as pdf_path:
        return process_document(
            pdf_path,
            document_id=drive_file["id"],
            metadata={
                "title": Path(drive_file.get("name", "document.pdf")).stem,
                "source_drive_id": drive_file["id"],
                "source_name": drive_file.get("name"),
                "source_modified_time": drive_file.get("modifiedTime"),
                "source_md5_checksum": drive_file.get("md5Checksum"),
                "original_source": {
                    "provider": "google_drive",
                    "file_id": drive_file["id"],
                    "file_name": drive_file.get("name"),
                    "modified_time": drive_file.get("modifiedTime"),
                    "md5_checksum": drive_file.get("md5Checksum"),
                },
                "metadata_inference": "Jurisdiction inferred from title prefix: Kerala => state; otherwise central. Document type remains separately unverified.",
            },
            chunk_config=chunk_config,
        )


def embed_document_result(result: dict[str, Any], embedder, *, cache_path: str | Path) -> list[dict[str, Any]]:
    """Explicit embedding stage; refuses invalid chunk payloads."""
    if not result.get("validated"):
        raise ValueError("Refusing to embed document chunks that failed validation.")
    return embedder.embed_chunks(result["chunks"], cache_path=cache_path)


def store_document_result(
    result: dict[str, Any], store, *, mappings, rows, user_approved: bool = False, dry_run: bool = True
) -> dict[str, int]:
    """Explicit guarded database stage; caller must provide inspected mappings and approval."""
    if not result.get("validated"):
        raise ValueError("Refusing to store document chunks that failed validation.")
    if dry_run or not user_approved:
        raise RuntimeError("Storage requires dry_run=False and explicit user_approved=True.")
    return store(result, mappings=mappings, rows=rows, validated=True, user_approved=True, dry_run=False)
