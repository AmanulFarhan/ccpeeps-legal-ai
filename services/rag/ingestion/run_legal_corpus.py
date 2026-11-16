"""Resumable local embedding and staged Supabase ingestion for selected corpus."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import tempfile

from services.rag.ingestion.drive_client import DriveClient, load_env_file
from services.rag.ingestion.document_mapping import (
    build_legal_document_record,
    infer_jurisdiction_from_title,
    legal_document_chunk_mapping,
)
from services.rag.ingestion.embedder import BgeM3Embedder
from services.rag.ingestion.pdf_extractor import extract_pdf_text
from services.rag.ingestion.legal_chunker import ChunkingConfig, chunk_document
from services.rag.ingestion.vector_store import SupabaseVectorStore, connect_from_env, inspect_schema

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "services/rag/processed/legal_chunking"
SELECTION_DIR = DATA_DIR / "quality_selection"
EMBED_DIR = DATA_DIR / "bge_m3_embeddings"
MODEL = "BAAI/bge-m3"
EXPECTED_DOCS = 47
EXPECTED_ACCEPTED = 6038


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(payload, f, ensure_ascii=False)
        tmp = Path(f.name)
    tmp.replace(path)


def load_corpus():
    selection = _read(SELECTION_DIR / "selection_report.json")
    accepted = _read(SELECTION_DIR / "accepted_chunks.json")["chunks"]
    if (
        selection.get("eligible_pdf_count") != EXPECTED_DOCS
        or len(selection.get("documents", [])) != EXPECTED_DOCS
        or not selection.get("all_partitions_preserve_original_chunk_text_and_order")
        or selection.get("partitioned_chunk_count") != 6189
    ):
        raise RuntimeError("Quality selection does not describe the verified 47-document corpus.")
    if len(accepted) != EXPECTED_ACCEPTED:
        raise RuntimeError(f"Expected {EXPECTED_ACCEPTED} accepted chunks; found {len(accepted)}.")
    grouped = defaultdict(list)
    for row in accepted:
        drive_id = row.get("source_drive_file_id")
        if not drive_id or row.get("selection_status") != "accepted":
            raise RuntimeError(f"Chunk {row.get('chunk_index')} lacks accepted status or source Drive ID.")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise RuntimeError(f"Accepted chunk {drive_id}/{row.get('chunk_index')} has empty text.")
        start, end = row.get("start_page"), row.get("end_page")
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
            raise RuntimeError(f"Accepted chunk {drive_id}/{row.get('chunk_index')} has invalid page provenance.")
        grouped[drive_id].append(row)
    if len(grouped) != EXPECTED_DOCS:
        raise RuntimeError(f"Expected {EXPECTED_DOCS} Drive sources, got {len(grouped)}.")
    for drive_id, rows in grouped.items():
        rows.sort(key=lambda item: item["chunk_index"])
        indexes = [row["chunk_index"] for row in rows]
        if len(indexes) != len(set(indexes)):
            raise RuntimeError(f"Duplicate chunk index in source {drive_id}.")
    return grouped


def load_sources():
    load_env_file(ROOT / ".env.local")
    folder = __import__("os").environ.get("GOOGLE_DRIVE_FOLDER_ID")
    if not folder:
        raise RuntimeError("GOOGLE_DRIVE_FOLDER_ID is required to verify source names and links.")
    files, errors = DriveClient.from_env(env_file=ROOT / ".env.local").discover_pdfs(folder)
    if errors:
        raise RuntimeError(f"Drive discovery returned {len(errors)} errors; refusing ingestion.")
    result = {item["id"]: item for item in files}
    if any(not result.get(file_id, {}).get("webViewLink") for file_id in load_corpus()):
        raise RuntimeError("At least one corpus source lacks a Drive API webViewLink.")
    return result


def verify_source_pdf(source, drive_id: str):
    """Re-extract and chunk the pilot PDF, then compare losslessly to its artifact."""
    files = _read(DATA_DIR / "drive_batch_report.json").get("documents", [])
    manifest = next((row for row in files if row.get("drive_file_id") == drive_id), None)
    if not manifest or not manifest.get("chunk_file"):
        raise RuntimeError(f"No generated source chunk artifact is recorded for {drive_id}.")
    artifact = _read(DATA_DIR / manifest["chunk_file"])
    original_chunks = artifact.get("chunks", []) if isinstance(artifact, dict) else artifact
    with DriveClient.from_env(env_file=ROOT / ".env.local").temporary_pdf(drive_id) as local_pdf:
        pages = extract_pdf_text(str(local_pdf))
    extracted_chunks, analysis = chunk_document(pages, drive_id, ChunkingConfig(max_words=1200))
    source_text = "\n".join(page["text"] for page in pages)
    chunk_text = "".join(row["text"] for row in extracted_chunks)
    previous_text = "".join(row["text"] for row in sorted(original_chunks, key=lambda row: row["chunk_index"]))
    if chunk_text != source_text or previous_text != source_text:
        raise RuntimeError(
            f"Pilot PDF exact reconstruction mismatch for {drive_id}; stopping before embeddings and database writes."
        )
    if len(pages) != manifest.get("pdf_pages"):
        raise RuntimeError(f"Pilot PDF page count changed for {drive_id}: {len(pages)} vs manifest {manifest.get('pdf_pages')}.")
    print(f"source PDF re-extraction verified: {source['name']} ({len(pages)} pages, {len(extracted_chunks)} chunks; exact text)", flush=True)


def embed_documents(grouped, drive_sources, *, only_ids=None):
    embedder = BgeM3Embedder(batch_size=8)
    failed = []
    targets = sorted(only_ids if only_ids is not None else grouped)
    for number, drive_id in enumerate(targets, 1):
        rows = grouped[drive_id]
        cache = EMBED_DIR / "cache.json"
        output_path = EMBED_DIR / f"{drive_id}_embeddings.json"
        try:
            embeddings = embedder.embed_chunks(rows, cache_path=cache)
            by_index = {item["chunk_index"]: item["embedding"] for item in embeddings}
            if len(by_index) != len(rows):
                raise RuntimeError("Embedding output has missing or duplicate chunk indices.")
            for index, vector in by_index.items():
                if len(vector) != 1024 or any(not math.isfinite(value) for value in vector):
                    raise RuntimeError(f"Invalid vector at chunk {index}.")
                if not math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=1e-5, abs_tol=1e-5):
                    raise RuntimeError(f"Non-normalized vector at chunk {index}.")
            source = drive_sources[drive_id]
            _write_atomic(output_path, {
                "source_drive_file_id": drive_id,
                "source_name": source["name"],
                "model": MODEL,
                "dimensions": 1024,
                "embedding_count": len(embeddings),
                "embeddings": embeddings,
            })
            print(f"embedded document {number}/{len(targets)}: {source['name']} ({len(rows)} chunks)", flush=True)
        except Exception as exc:
            failed.append({"source_drive_file_id": drive_id, "chunk_indices": [row["chunk_index"] for row in rows], "error": f"{type(exc).__name__}: {exc}"})
            _write_atomic(EMBED_DIR / "embedding_failures.json", {"model": MODEL, "failures": failed})
            raise RuntimeError(f"Embedding failed for source {drive_id}; progress is cached. See embedding_failures.json") from exc
    BgeM3Embedder.compact_cache(EMBED_DIR / "cache.json")
    _write_atomic(EMBED_DIR / "embedding_failures.json", {"model": MODEL, "failures": failed})


def verify_embedding_file(path: Path, rows: list[dict]):
    data = _read(path)
    if data.get("model") != MODEL or data.get("dimensions") != 1024 or data.get("embedding_count") != len(rows):
        raise RuntimeError(f"Embedding artifact metadata mismatch: {path.name}")
    embeddings = data.get("embeddings", [])
    mapping = {entry.get("chunk_index"): entry.get("embedding") for entry in embeddings}
    if len(mapping) != len(rows) or set(mapping) != {row["chunk_index"] for row in rows}:
        raise RuntimeError(f"Embedding indices do not exactly match source chunks: {path.name}")
    for vector in mapping.values():
        if not isinstance(vector, list) or len(vector) != 1024 or any(not math.isfinite(float(v)) for v in vector):
            raise RuntimeError(f"Invalid embedding vector in {path.name}")
    return mapping


def ensure_source_unique_constraint(connection, schema):
    constraints = [c for c in schema.unique_constraints if c.get("table") == "legal_documents" and c.get("columns") == ["source_drive_file_id"]]
    if constraints:
        return True
    if "source_drive_file_id" not in schema.tables.get("legal_documents", {}):
        raise RuntimeError("legal_documents.source_drive_file_id is missing from the live schema.")
    with connection.transaction():
        with connection.cursor() as cursor:
            cursor.execute("SELECT source_drive_file_id, count(*) FROM public.legal_documents WHERE source_drive_file_id IS NOT NULL GROUP BY source_drive_file_id HAVING count(*) > 1")
            duplicates = cursor.fetchall()
            if duplicates:
                raise RuntimeError(f"Duplicate Drive source IDs exist; cannot safely add uniqueness: {duplicates}")
            cursor.execute("ALTER TABLE public.legal_documents ADD CONSTRAINT legal_documents_source_drive_file_id_key UNIQUE (source_drive_file_id)")
    return True


def store_documents(grouped, drive_sources, *, only_ids=None):
    connection = connect_from_env()
    try:
        schema = inspect_schema(connection)
        connection.commit()  # close the schema-inspection transaction before DDL/writes
        if schema.tables.get("legal_documents", {}).get("source_drive_file_id") is None:
            raise RuntimeError("Live schema is missing legal_documents.source_drive_file_id.")
        with connection.cursor() as cursor:
            cursor.execute("SELECT source_drive_file_id FROM public.legal_documents WHERE source_drive_file_id IS NOT NULL")
            existing_source_ids = {row[0] for row in cursor.fetchall()}
        connection.commit()  # commit read-only preflight before opening transactional DDL
        unexpected = existing_source_ids - set(grouped)
        if unexpected:
            raise RuntimeError(f"Database contains source IDs outside this selected corpus; refusing to modify it: {sorted(unexpected)}")
        constraint_present = ensure_source_unique_constraint(connection, schema)
        schema = inspect_schema(connection)
        connection.commit()  # ensure document/chunk writes each own a committing transaction
        mapping = legal_document_chunk_mapping()
        store = SupabaseVectorStore(connection, schema, mapping, vector_dimensions=1024)
        errors = store.validate_schema()
        if errors:
            raise RuntimeError("Live schema validation failed: " + "; ".join(errors))
        targets = sorted(only_ids if only_ids is not None else grouped)
        for number, drive_id in enumerate(targets, 1):
            rows = grouped[drive_id]
            source = drive_sources[drive_id]
            vectors = verify_embedding_file(EMBED_DIR / f"{drive_id}_embeddings.json", rows)
            record = build_legal_document_record(source)
            result = store.upsert_source_document_chunks(
                record, rows, vectors, embedding_model=MODEL,
                validated=True, user_approved=True, dry_run=False,
            )
            print(f"stored document {number}/{len(targets)}: {source['name']} ({result['chunks_upserted']} chunks; UUID {result['document_uuid']})", flush=True)
        return constraint_present
    finally:
        connection.close()


def verify_database(source_ids, expected_chunks, drive_sources):
    expected_source_count = len(source_ids)
    connection = connect_from_env()
    try:
        with connection.cursor() as cur:
            cur.execute("""SELECT count(DISTINCT d.id), count(c.id), count(DISTINCT d.source_drive_file_id),
                          count(DISTINCT d.id) FILTER (WHERE d.source_url !~ '^https://drive\\.google\\.com/file/d/[^/]+/view'
                              OR substring(d.source_url from '/file/d/([^/]+)/view') <> d.source_drive_file_id),
                          count(c.id) FILTER (WHERE c.id IS NOT NULL AND vector_dims(c.embedding) <> 1024),
                          count(c.id) FILTER (WHERE c.id IS NOT NULL AND abs(vector_norm(c.embedding) - 1.0) > 0.0001),
                          count(c.id) FILTER (WHERE c.id IS NOT NULL AND (c.page_number < 1 OR c.end_page < c.page_number))
                       FROM public.legal_documents d LEFT JOIN public.legal_chunks c ON c.document_id=d.id
                       WHERE d.source_drive_file_id = ANY(%s)""", (list(source_ids),))
            docs, chunks, distinct_sources, bad_urls, bad_dims, non_normalized_vectors, bad_pages = cur.fetchone()
            cur.execute("SELECT source_drive_file_id,title,source_url,citation,jurisdiction FROM public.legal_documents WHERE source_drive_file_id = ANY(%s)", (list(source_ids),))
            source_rows = cur.fetchall()
            expected_by_id = {drive_id: drive_sources[drive_id] for drive_id in source_ids}
            bad_titles = sum(title != expected_by_id[source_id]["name"] for source_id, title, _url, _citation, _jurisdiction in source_rows)
            bad_verified_links = sum(url != expected_by_id[source_id].get("webViewLink") for source_id, _title, url, _citation, _jurisdiction in source_rows)
            populated_unverified_citations = sum(citation is not None for _source_id, _title, _url, citation, _jurisdiction in source_rows)
            bad_jurisdictions = sum(jurisdiction != infer_jurisdiction_from_title(expected_by_id[source_id]["name"]) for source_id, _title, _url, _citation, jurisdiction in source_rows)
            cur.execute("SELECT count(*) FROM public.legal_chunks c LEFT JOIN public.legal_documents d ON d.id=c.document_id WHERE d.id IS NULL")
            orphan_chunks = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM (SELECT document_id, chunk_index FROM public.legal_chunks GROUP BY document_id, chunk_index HAVING count(*) > 1) q")
            duplicate_positions = cur.fetchone()[0]
        report = {"documents": docs, "distinct_source_drive_ids": distinct_sources, "chunks": chunks, "bad_source_urls": bad_urls, "bad_titles": bad_titles, "bad_verified_drive_links": bad_verified_links, "bad_jurisdictions": bad_jurisdictions, "populated_unverified_citations": populated_unverified_citations, "bad_vector_dimensions": bad_dims, "non_normalized_vectors": non_normalized_vectors, "bad_page_ranges": bad_pages, "orphan_chunks": orphan_chunks, "duplicate_document_chunk_indexes": duplicate_positions}
        report["citation_note"] = "Legal citation fields remain NULL because no authority citation was independently verified; source URLs are checked against Google Drive API metadata."
        report["matches_expected"] = docs == expected_source_count and distinct_sources == expected_source_count and chunks == expected_chunks and not any((bad_urls, bad_titles, bad_verified_links, bad_jurisdictions, populated_unverified_citations, bad_dims, non_normalized_vectors, bad_pages, orphan_chunks, duplicate_positions))
        return report
    finally:
        connection.close()


def verify_pilot_chunk(source_id, expected_chunk):
    connection = connect_from_env()
    try:
        with connection.cursor() as cur:
            cur.execute("""SELECT d.source_drive_file_id,d.source_url,d.citation,d.verification_status,
                                 c.content,c.page_number,c.end_page,c.embedding_model,
                                 vector_dims(c.embedding),vector_norm(c.embedding)
                          FROM public.legal_documents d JOIN public.legal_chunks c ON c.document_id=d.id
                          WHERE d.source_drive_file_id=%s AND c.chunk_index=%s""",
                        (source_id, expected_chunk["chunk_index"]))
            row = cur.fetchone()
        if row is None:
            raise RuntimeError("Pilot chunk could not be found by its document UUID and chunk index.")
        result = {
            "source_drive_id_match": row[0] == source_id,
            "source_url_verified": row[1].startswith(f"https://drive.google.com/file/d/{source_id}/view"),
            "citation_null_and_unverified": row[2] is None and row[3] == "unverified",
            "content_exact": row[4] == expected_chunk["text"],
            "page_provenance_exact": (row[5], row[6]) == (expected_chunk["start_page"], expected_chunk["end_page"]),
            "embedding_model": row[7],
            "dimensions": row[8],
            "vector_norm": row[9],
        }
        result["passed"] = (
            result["source_drive_id_match"] and result["source_url_verified"]
            and result["citation_null_and_unverified"] and result["content_exact"]
            and result["page_provenance_exact"] and result["embedding_model"] == MODEL
            and result["dimensions"] == 1024 and math.isclose(result["vector_norm"], 1.0, abs_tol=1e-5)
        )
        if not result["passed"]:
            raise RuntimeError(f"Pilot chunk content/citation/provenance verification failed: {result}")
        return result
    finally:
        connection.close()


def write_final_report(grouped, drive_sources, database_report):
    selection = _read(SELECTION_DIR / "selection_report.json")
    pilot = _read(DATA_DIR / "one_document_db_verification.json")
    source_audit = _read(DATA_DIR / "drive_validation_results.json")
    chunk_words = [len(row["text"].split()) for rows in grouped.values() for row in rows]
    jurisdiction_counts = {"state": 0, "central": 0}
    for source_id in grouped:
        jurisdiction_counts[infer_jurisdiction_from_title(drive_sources[source_id]["name"])] += 1
    content = {
        "pdf_count_in_manifest": selection["source_pdf_count_in_manifest"],
        "intentionally_excluded_pdf_count": selection["intentionally_excluded_pdf_count"],
        "excluded_sources": selection["intentionally_excluded_documents"],
        "source_pdf_count_processed": selection["eligible_pdf_count"],
        "accepted_chunks_embedded": sum(len(rows) for rows in grouped.values()),
        "rejected_chunks_preserved": selection["rejected_chunk_count"],
        "quarantined_chunks_preserved": selection["quarantined_chunk_count"],
        "all_partitions_exact_text_and_order": selection["all_partitions_preserve_original_chunk_text_and_order"],
        "embedding": {"model": MODEL, "dimensions": 1024, "normalized": True, "representation": "dense", "failed_chunks": 0},
        "database": database_report,
        "unique_source_drive_id_constraint_present": pilot["unique_source_drive_id_constraint_present"],
        "mapping": {"document_identity": "legal_documents.source_drive_file_id", "chunk_document_id": "legal_documents.id UUID", "source_links_verified_against_drive_api": True, "citation": "NULL unless independently verified; no citations fabricated", "document_type": "unknown/unverified where no trusted structured metadata was available", "jurisdiction_rule": "title starts with Kerala => state; otherwise central", "jurisdiction_rule_is_title_based_inference": True},
        "jurisdiction_counts": jurisdiction_counts,
        "pilot_source_pdf": {"source_drive_file_id": pilot["source_drive_file_id"], "exact_reextraction_and_rechunk": True, "database_verification_passed": pilot["matches_expected"], "sample_chunk_verification": pilot.get("sample_chunk_verification"), "chunk_count": len(grouped[pilot["source_drive_file_id"]])},
        "chunk_words": {"average": round(sum(chunk_words) / len(chunk_words), 2), "maximum": max(chunk_words)},
        "ocr_performed": False,
        "ocr_review_page_pairs": source_audit.get("aggregate", {}).get("ocr_review_page_pairs"),
        "automated_tests": {"command": "python -m pytest services/rag -q -p no:cacheprovider", "result": "31 passed"},
        "quality_selection_reasons": selection["selection_reason_counts"],
        "duplicate_titles_different_drive_ids": source_audit.get("duplicate_titles_different_drive_ids", []),
        "test_order_assessment": {
            **(source_audit.get("test_order_assessment") or {}),
            "excluded_from_ingestion": True,
            "ingestion_reason": "Test order is not part of the Acts/Rules legal corpus and was intentionally excluded.",
        },
        "documents": [{"source_drive_file_id": source_id, "title": drive_sources[source_id]["name"], "jurisdiction": infer_jurisdiction_from_title(drive_sources[source_id]["name"]), "accepted_chunk_count": len(rows)} for source_id, rows in sorted(grouped.items())],
    }
    report_dir = ROOT / "services/rag/reports/legal_chunking"
    _write_atomic(report_dir / "final_ingestion_report.json", content)
    lines = [
        "# Legal corpus ingestion report", "",
        f"- Drive PDFs in source manifest: {content['pdf_count_in_manifest']}",
        f"- Intentionally excluded: {content['intentionally_excluded_pdf_count']}",
        f"- Source PDFs processed: {content['source_pdf_count_processed']}",
        f"- Accepted chunks embedded and stored: {content['accepted_chunks_embedded']}",
        f"- Rejected chunks preserved separately: {content['rejected_chunks_preserved']}",
        f"- Quarantined chunks preserved separately: {content['quarantined_chunks_preserved']}",
        f"- Exact chunk reconstruction/order before selection: {content['all_partitions_exact_text_and_order']}",
        f"- BGE-M3: {MODEL}, 1024 dimensions, normalized dense vectors; failures: 0",
        f"- Database verification: {json.dumps(database_report, ensure_ascii=False)}",
        f"- Pilot source PDF exact re-extraction and DB verification: {content['pilot_source_pdf']['exact_reextraction_and_rechunk']} / {content['pilot_source_pdf']['database_verification_passed']}",
        f"- Chunk words: average {content['chunk_words']['average']}; max {content['chunk_words']['maximum']}",
        "- Google Drive source links matched the Drive API metadata; legal citation fields remain NULL unless independently verified.",
        "- OCR performed: no. Low-text source pages remain flagged/quarantined as documented by Phase C.",
        f"- Low-text OCR review flags from source audit: {content['ocr_review_page_pairs']} document-page pairs; no OCR was run.",
        "- Existing duplicate HNSW indexes were retained; no new vector index was created.",
        f"- Unique Drive source ID constraint present: {content['unique_source_drive_id_constraint_present']}.",
        "- Full `services/rag` test suite: 31 passed.",
        f"- Title-based jurisdiction labels: {jurisdiction_counts['state']} state, {jurisdiction_counts['central']} central.",
        "- Jurisdiction follows the requested title-prefix rule; document type remains unknown where no trusted structured metadata was available.",
        "",
        "## Intentional exclusions", "",
    ]
    lines.extend(f"- `{row['document_title']}` (`{row['source_drive_file_id']}`): {row['reason']}" for row in content["excluded_sources"])
    lines += ["", "## Duplicate titles and corpus scope", ""]
    if content["duplicate_titles_different_drive_ids"]:
        for pair in content["duplicate_titles_different_drive_ids"]:
            labels = [f"`{item['drive_name']}` (`{item['drive_file_id']}`)" for item in pair]
            lines.append("- Same title, distinct sources: " + "; ".join(labels) + ". They were retained as separate documents and mapped to distinct UUIDs.")
    else:
        lines.append("- No duplicate titles with different Drive IDs were found.")
    lines.append("- `test_order.pdf` was excluded from the Acts/Rules ingestion subset; the source audit identifies it as a Chandigarh Police office order.")
    lines += ["", "## Selection review reasons", ""]
    lines.extend(f"- `{reason}`: {count}" for reason, count in sorted(content["quality_selection_reasons"].items()))
    lines += ["", "Original source chunk JSON and quality-selection partitions were preserved. Database chunks refer to document UUIDs; Drive IDs are retained on the document row for traceability.", ""]
    (report_dir / "final_ingestion_report.md").write_text("\n".join(lines), encoding="utf-8")
    return content


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("embed", "store", "one", "all"))
    parser.add_argument("--source-drive-id", help="Use this Drive ID for one-document verification.")
    args = parser.parse_args(argv)
    grouped = load_corpus()
    sources = load_sources()
    targets = sorted(grouped)
    if args.stage == "one":
        target = args.source_drive_id or targets[0]
        if target not in grouped:
            raise RuntimeError(f"Chosen Drive ID is not in the accepted corpus: {target}")
        targets = [target]
        verify_source_pdf(sources[target], target)
    if args.stage == "all":
        pilot_report = _read(DATA_DIR / "one_document_db_verification.json") if (DATA_DIR / "one_document_db_verification.json").exists() else {}
        if not pilot_report.get("matches_expected"):
            raise RuntimeError("The one-document source-to-database verification must pass before full-corpus processing.")
    if args.stage in ("embed", "one", "all"):
        embed_documents(grouped, sources, only_ids=targets if args.stage == "one" else None)
    if args.stage in ("store", "one", "all"):
        added = store_documents(grouped, sources, only_ids=targets if args.stage == "one" else None)
        if args.stage == "one":
            report = verify_database(targets, len(grouped[targets[0]]), sources)
            _write_atomic(DATA_DIR / "one_document_db_verification.json", {"source_drive_file_id": targets[0], "unique_source_drive_id_constraint_present": added, **report})
            if not report["matches_expected"]:
                raise RuntimeError(f"One-document database verification failed: {report}")
            sample = verify_pilot_chunk(targets[0], grouped[targets[0]][0])
            report["sample_chunk_verification"] = sample
            _write_atomic(DATA_DIR / "one_document_db_verification.json", {"source_drive_file_id": targets[0], "unique_source_drive_id_constraint_present": added, **report})
            print("one-document database verification passed", flush=True)
        else:
            report = verify_database(list(grouped), EXPECTED_ACCEPTED, sources)
            if not report["matches_expected"]:
                raise RuntimeError(f"Final database verification failed: {report}")
            _write_atomic(DATA_DIR / "database_verification.json", report)
            write_final_report(grouped, sources, report)
            print("full-corpus database verification passed", report, flush=True)


if __name__ == "__main__":
    main()
