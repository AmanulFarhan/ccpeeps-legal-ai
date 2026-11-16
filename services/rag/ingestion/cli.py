"""Command-line entry points for independent legal ingestion stages.

Embedding generation and database writes are explicit commands. Database writes
also require both --write and --approve-ingestion; dry-run is the default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from services.rag.ingestion.chunk_validator import validate_chunks
from services.rag.ingestion.drive_client import DriveClient, load_env_file
from services.rag.ingestion.document_processor import process_document
from services.rag.ingestion.embedder import BgeM3Embedder, GeminiEmbedder
from services.rag.ingestion.pdf_extractor import extract_pdf_text
from services.rag.ingestion.text_cleaner import clean_extracted_pages
from services.rag.ingestion.vector_store import (
    DatabaseSchema,
    DocumentChunkMapping,
    TableMapping,
    SupabaseVectorStore,
    inspect_schema,
)

ROOT = Path(__file__).resolve().parents[3]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _table_mapping(data: dict) -> TableMapping:
    return TableMapping(
        table=data["table"],
        field_to_column=data["field_to_column"],
        conflict_columns=tuple(data["conflict_columns"]),
        embedding_field=data.get("embedding_field"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("discover", "extract", "clean", "chunk", "validate", "embed", "inspect-schema", "store"))
    parser.add_argument("--pdf")
    parser.add_argument("--folder-id")
    parser.add_argument("--chunks")
    parser.add_argument("--pages")
    parser.add_argument("--document")
    parser.add_argument("--embeddings")
    parser.add_argument("--cache", default="services/rag/processed/legal_chunking/embedding_cache.json")
    parser.add_argument("--embedding-provider", choices=("gemini", "bge-m3"), default="gemini")
    parser.add_argument("--output")
    parser.add_argument("--mapping")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--approve-ingestion", action="store_true")
    args = parser.parse_args(argv)
    load_env_file(ROOT / ".env.local")

    if args.stage == "discover":
        folder_id = args.folder_id or os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        if not folder_id:
            parser.error("--folder-id or GOOGLE_DRIVE_FOLDER_ID is required")
        files, errors = DriveClient.from_env(env_file=ROOT / ".env.local").discover_pdfs(folder_id)
        output = {"pdf_count": len(files), "files": files, "errors": errors}
    elif args.stage == "extract":
        if not args.pdf or not args.output:
            parser.error("extract requires --pdf and --output")
        output = extract_pdf_text(args.pdf)
        _write(args.output, output)
    elif args.stage == "clean":
        if not args.pages or not args.output:
            parser.error("clean requires --pages and --output")
        output = clean_extracted_pages(_read(args.pages))
        _write(args.output, output)
    elif args.stage == "chunk":
        if not args.pdf or not args.output:
            parser.error("chunk requires --pdf and --output")
        result = process_document(args.pdf)
        output = {"document": result["document"], "source_sha256": result["source_sha256"],
                  "analysis": result["analysis"], "validation": result["validation"], "chunks": result["chunks"]}
        _write(args.output, output)
    elif args.stage == "validate":
        if not args.pdf or not args.chunks:
            parser.error("validate requires --pdf and --chunks")
        pages = extract_pdf_text(args.pdf)
        chunks_data = _read(args.chunks)
        chunks = chunks_data.get("chunks", chunks_data)
        output = validate_chunks(chunks, pages=pages)
        if args.output:
            _write(args.output, output)
    elif args.stage == "embed":
        if not args.chunks:
            parser.error("embed requires --chunks")
        chunk_data = _read(args.chunks)
        chunks = chunk_data.get("chunks", chunk_data)
        validation = chunk_data.get("validation")
        if validation is None or not validation.get("valid"):
            raise RuntimeError("Embedding requires a chunk file with a passing validation object; generate it with --stage chunk first.")
        embedder = BgeM3Embedder() if args.embedding_provider == "bge-m3" else GeminiEmbedder()
        vectors = embedder.embed_chunks(chunks, cache_path=args.cache)
        output = {"model": embedder.model_name if isinstance(embedder, BgeM3Embedder) else embedder.model,
                  "dimensions": embedder.dimensions, "embeddings": vectors, "cache_path": args.cache}
        if args.output:
            _write(args.output, output)
    elif args.stage == "inspect-schema":
        schema = inspect_schema()
        output = {
            "schema_name": schema.schema_name,
            "tables": {table: {name: {"type": col.data_type, "udt": col.udt_name, "nullable": col.nullable} for name, col in columns.items()} for table, columns in schema.tables.items()},
            "foreign_keys": schema.foreign_keys,
            "unique_constraints": schema.unique_constraints,
        }
        if args.output:
            _write(args.output, output)
    elif args.stage == "store":
        if not args.chunks or not args.embeddings or not args.document or not args.mapping:
            parser.error("store requires --chunks, --embeddings, --document, and --mapping")
        mapping_json = _read(args.mapping)
        connection = __import__("services.rag.ingestion.vector_store", fromlist=["connect_from_env"]).connect_from_env()
        try:
            schema: DatabaseSchema = inspect_schema(connection, mapping_json.get("schema_name", "public"))
            mapping = DocumentChunkMapping(_table_mapping(mapping_json["documents"]), _table_mapping(mapping_json["chunks"]))
            doc_record = _read(args.document)
            if "document_record" in doc_record:
                doc_record = doc_record["document_record"]
            chunk_data = _read(args.chunks)
            embed_data_file = _read(args.embeddings)
            embed_data = embed_data_file["embeddings"]
            dimensions = embed_data_file.get("dimensions")
            if dimensions is None and embed_data:
                dimensions = len(embed_data[0]["embedding"])
            if not dimensions or any(len(item["embedding"]) != dimensions for item in embed_data):
                raise RuntimeError("Embedding output has missing or inconsistent vector dimensions.")
            store = SupabaseVectorStore(connection, schema, mapping, vector_dimensions=dimensions)
            errors = store.validate_schema()
            if errors:
                raise RuntimeError("Schema mapping did not validate: " + "; ".join(errors))
            if not chunk_data.get("validation", {}).get("valid"):
                raise RuntimeError("Refusing to store chunks without passing validation metadata.")
            if not isinstance(chunk_data.get("chunks"), list):
                raise RuntimeError("Chunk artifact must contain a chunks list.")
            vectors = {item["chunk_index"]: item["embedding"] for item in embed_data}
            if len(vectors) != len(embed_data):
                raise RuntimeError("Embedding artifact contains duplicate chunk indices.")
            models = {item.get("model") for item in embed_data if item.get("model")}
            embedding_model = embed_data_file.get("model") or (next(iter(models)) if len(models) == 1 else None)
            if not embedding_model or (models and models != {embedding_model}):
                raise RuntimeError("Embedding artifact has missing or inconsistent model identifiers.")
            if not (args.write and args.approve_ingestion):
                output = {
                    "dry_run": True,
                    "source_drive_file_id": doc_record.get("source_drive_file_id"),
                    "document_rows": 1,
                    "chunk_rows": len(chunk_data["chunks"]),
                    "embedding_model": embedding_model,
                    "embedding_dimensions": dimensions,
                    "schema_valid": True,
                    "uuid_resolution": "deferred until an explicitly approved write",
                    "message": "No database writes performed. Pass both --write and --approve-ingestion to write.",
                }
            else:
                counts = store.upsert_source_document_chunks(
                    doc_record,
                    chunk_data["chunks"],
                    vectors,
                    embedding_model=embedding_model,
                    validated=True,
                    user_approved=True,
                    dry_run=False,
                )
                output = {"dry_run": False, **counts}
        finally:
            connection.close()
    else:  # pragma: no cover - argparse constrains stages
        parser.error("Unknown pipeline stage")

    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
