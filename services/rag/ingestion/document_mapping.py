"""Safe mapping from Google Drive source identity to legal document UUIDs."""

from __future__ import annotations

from copy import deepcopy
from urllib.parse import urlparse
from typing import Any
from uuid import UUID


class DocumentMappingError(ValueError):
    """Source identity or verified metadata is insufficient or inconsistent."""


def infer_jurisdiction_from_title(title: str) -> str:
    """Apply the corpus rule: titles beginning with Kerala are state-level."""
    return "state" if str(title or "").strip().casefold().startswith("kerala") else "central"


def build_legal_document_record(
    drive_file: dict[str, Any],
    *,
    verified_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a DB-ready document row without inventing legal facts or URLs.

    ``title`` falls back to the exact Drive filename as a source label. Legal
    Citation, authority, version, status, type and language are only populated
    when supplied by a trusted metadata source. Jurisdiction follows the
    configured title-prefix rule (Kerala => state; otherwise central).
    """
    source_id = drive_file.get("id") or drive_file.get("source_drive_file_id")
    source_name = drive_file.get("name") or drive_file.get("file_name")
    source_url = drive_file.get("webViewLink") or drive_file.get("web_view_link")
    if not isinstance(source_id, str) or not source_id.strip():
        raise DocumentMappingError("A Google Drive file ID is required as source identity.")
    if not isinstance(source_name, str) or not source_name.strip():
        raise DocumentMappingError("A verified Drive filename is required for the document title fallback.")
    if not isinstance(source_url, str) or not source_url.strip():
        raise DocumentMappingError("A verified Drive webViewLink is required; refusing to fabricate source_url.")
    parsed = urlparse(source_url)
    if parsed.scheme != "https" or not parsed.hostname or not (
        parsed.hostname == "google.com"
        or parsed.hostname.endswith(".google.com")
        or parsed.hostname.endswith(".googleusercontent.com")
    ):
        raise DocumentMappingError("source_url must be the HTTPS Google Drive webViewLink returned by Drive API.")

    metadata = verified_metadata or {}
    title = metadata.get("title")
    title_origin = "verified_metadata" if isinstance(title, str) and title.strip() else "drive_filename"
    title = title.strip() if title_origin == "verified_metadata" else source_name
    # These required database columns use an explicit unknown/unverified marker
    # rather than a guessed value or the schema's potentially misleading default.
    return {
        "title": title,
        "citation": metadata.get("citation"),
        "document_type": metadata.get("document_type") or "unknown",
        "jurisdiction": infer_jurisdiction_from_title(title),
        "language": metadata.get("language") or "unknown",
        "source_url": source_url,
        "source_authority": metadata.get("source_authority"),
        "version_label": metadata.get("version_label"),
        "legal_status": metadata.get("legal_status") or "unknown",
        "verification_status": metadata.get("verification_status") or "unverified",
        "source_drive_file_id": source_id,
        "ingestion_metadata": {
            "source_drive_file_id": source_id,
            "drive_file_name": source_name,
            "title_origin": title_origin,
            "jurisdiction_origin": "title_prefix_kerala_state_otherwise_central",
            "source_url_origin": "google_drive_api.webViewLink",
        },
    }


def map_chunks_to_document_uuid(
    chunks: list[dict[str, Any]],
    *,
    document_uuid: str | UUID,
    source_drive_file_id: str,
) -> list[dict[str, Any]]:
    """Replace the chunk's Drive-ID pseudo-document ID with its DB UUID.

    The Drive ID is retained in per-chunk ingestion metadata and remains the
    authoritative document lookup key in ``legal_documents``.
    """
    try:
        uuid_value = str(UUID(str(document_uuid)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise DocumentMappingError("The legal_documents primary key must be a valid UUID.") from exc
    if not isinstance(source_drive_file_id, str) or not source_drive_file_id.strip():
        raise DocumentMappingError("source_drive_file_id is required when mapping chunks.")
    mapped = []
    for chunk in chunks:
        original_source_id = chunk.get("document_id")
        if original_source_id not in (None, source_drive_file_id):
            raise DocumentMappingError(
                f"Chunk {chunk.get('chunk_index')} has document_id {original_source_id!r}, "
                f"which does not match source Drive ID {source_drive_file_id!r}."
            )
        row = deepcopy(chunk)
        row["document_id"] = uuid_value
        row["ingestion_metadata"] = {
            **(row.get("ingestion_metadata") if isinstance(row.get("ingestion_metadata"), dict) else {}),
            "source_drive_file_id": source_drive_file_id,
            "source_chunk_document_id": original_source_id,
        }
        mapped.append(row)
    return mapped


def map_chunk_to_database_row(chunk: dict[str, Any], *, embedding_model: str, embedding: list[float]) -> dict[str, Any]:
    """Map a UUID-linked chunk into the existing ``legal_chunks`` columns."""
    chapter_number = chunk.get("chapter_number")
    chapter_title = chunk.get("chapter_title")
    chapter = " ? ".join(str(v) for v in (chapter_number, chapter_title) if v not in (None, "")) or None
    return {
        "document_id": chunk["document_id"],
        "chunk_index": chunk["chunk_index"],
        "section_number": chunk.get("section_number"),
        "section_title": chunk.get("section_title"),
        "subsection_number": chunk.get("subsection_number"),
        "clause": chunk.get("clause_number"),
        "chapter": chapter,
        "page_number": chunk.get("start_page"),
        "end_page": chunk.get("end_page"),
        "content": chunk["text"],
        "embedding": embedding,
        "embedding_model": embedding_model,
    }


def legal_document_chunk_mapping():
    """Return explicit field maps for the inspected public legal tables."""
    from services.rag.ingestion.vector_store import DocumentChunkMapping, TableMapping

    documents = TableMapping(
        table="legal_documents",
        field_to_column={
            field: field for field in (
                "id", "title", "citation", "document_type", "jurisdiction", "language",
                "source_url", "source_authority", "version_label", "legal_status",
                "verification_status", "source_drive_file_id",
            )
        },
        conflict_columns=("id",),
    )
    chunks = TableMapping(
        table="legal_chunks",
        field_to_column={
            field: field for field in (
                "document_id", "chunk_index", "section_number", "section_title",
                "subsection_number", "clause", "chapter", "page_number", "end_page",
                "content", "embedding", "embedding_model",
            )
        },
        conflict_columns=("document_id", "chunk_index"),
        embedding_field="embedding",
    )
    return DocumentChunkMapping(documents=documents, chunks=chunks)
