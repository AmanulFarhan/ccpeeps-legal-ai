import json
from types import SimpleNamespace
from uuid import UUID

import pytest

from services.rag.ingestion.chunk_validator import validate_chunks
from services.rag.ingestion.drive_client import DriveClient
from services.rag.ingestion import document_processor
from services.rag.ingestion.document_mapping import (
    DocumentMappingError,
    build_legal_document_record,
    infer_jurisdiction_from_title,
    legal_document_chunk_mapping,
    map_chunk_to_database_row,
    map_chunks_to_document_uuid,
)
from services.rag.ingestion.embedder import BgeM3Embedder, GeminiEmbedder
from services.rag.ingestion.vector_store import (
    ColumnInfo,
    DatabaseSchema,
    DocumentChunkMapping,
    TableMapping,
    SupabaseVectorStore,
    validate_mapping,
    upsert_rows,
)


def _chunk(text="1. Short title\nBody", **changes):
    row = {
        "document_id": "doc", "part": None, "chapter_number": None,
        "chapter_title": None, "section_number": "1", "section_title": "Short title",
        "subsection_number": None, "clause_number": None, "text": text,
        "start_page": 1, "end_page": 1, "chunk_index": 0,
        "chunking_method": "section", "word_count": len(text.split()),
        "classification_status": "classified",
    }
    row.update(changes)
    return row


def test_chunk_validator_checks_exact_text_and_page_mapping():
    pages = [{"page_number": 1, "text": "1. Short title\nBody"}]
    result = validate_chunks([_chunk()], pages=pages)
    assert result["valid"]
    assert result["exact_reconstruction"]
    assert result["page_provenance_checked"]


def test_chunk_validator_reports_missing_metadata_without_mutating():
    row = _chunk()
    row.pop("chapter_title")
    result = validate_chunks([row])
    assert not result["valid"]
    assert "chapter_title" in result["errors"][0]["fields"]
    assert "chapter_title" not in row


class _Request:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class _DriveFiles:
    def list(self, **kwargs):
        self.last_kwargs = kwargs
        parent = kwargs["q"].split("'")[1]
        rows = {
            "root": [{"id": "sub", "name": "sub", "mimeType": "application/vnd.google-apps.folder"}],
            "sub": [{"id": "pdf", "name": "act.pdf", "mimeType": "application/pdf"}],
        }
        return _Request({"files": rows.get(parent, [])})


class _DriveService:
    def __init__(self):
        self._files = _DriveFiles()

    def files(self):
        return self._files


def test_drive_client_recursively_discovers_pdfs():
    files, errors = DriveClient(_DriveService()).discover_pdfs("root")
    assert [f["id"] for f in files] == ["pdf"]
    assert errors == []


def test_drive_discovery_requests_verified_web_view_link():
    service = _DriveService()
    DriveClient(service).list_children("root")
    # The Drive API fields parameter includes the actual link; no URL is synthesized.
    assert "webViewLink" in service.files().last_kwargs["fields"]


def test_document_record_uses_drive_identity_and_does_not_invent_legal_facts():
    drive_file = {
        "id": "drive-123", "name": "Central Act.pdf",
        "webViewLink": "https://drive.google.com/file/d/drive-123/view",
    }
    record = build_legal_document_record(drive_file)
    assert record["title"] == "Central Act.pdf"
    assert record["source_url"] == drive_file["webViewLink"]
    assert record["source_drive_file_id"] == "drive-123"
    assert record["citation"] is None
    assert record["source_authority"] is None
    assert record["document_type"] == "unknown"
    assert record["jurisdiction"] == "central"
    assert record["verification_status"] == "unverified"
    assert record["ingestion_metadata"]["title_origin"] == "drive_filename"
    assert record["ingestion_metadata"]["jurisdiction_origin"] == "title_prefix_kerala_state_otherwise_central"


def test_jurisdiction_uses_only_case_insensitive_kerala_title_prefix():
    assert infer_jurisdiction_from_title("Kerala_Act_Education.pdf") == "state"
    assert infer_jurisdiction_from_title("  KERALA Rule.pdf") == "state"
    assert infer_jurisdiction_from_title("The Kerala Education Act.pdf") == "central"
    assert infer_jurisdiction_from_title("Companies Act.pdf") == "central"


def test_document_record_refuses_unverified_source_url():
    with pytest.raises(DocumentMappingError, match="refusing to fabricate"):
        build_legal_document_record({"id": "drive-123", "name": "Act.pdf"})


def test_chunk_mapping_uses_document_uuid_and_keeps_drive_id_trace():
    drive_id = "drive-123"
    doc_uuid = "3a89951f-71da-43f4-9f89-7d9bf3f7e6e2"
    chunks = [_chunk(document_id=drive_id, chunk_index=0), _chunk(document_id=drive_id, chunk_index=1)]
    mapped = map_chunks_to_document_uuid(chunks, document_uuid=doc_uuid, source_drive_file_id=drive_id)
    assert {row["document_id"] for row in mapped} == {doc_uuid}
    assert all(row["ingestion_metadata"]["source_drive_file_id"] == drive_id for row in mapped)
    assert chunks[0]["document_id"] == drive_id  # mapping does not mutate source JSON
    with pytest.raises(DocumentMappingError, match="does not match"):
        map_chunks_to_document_uuid(chunks, document_uuid=doc_uuid, source_drive_file_id="other-drive")


def test_chunk_database_mapping_preserves_citation_provenance_fields():
    chunk = map_chunks_to_document_uuid(
        [_chunk(document_id="drive-1")],
        document_uuid="3a89951f-71da-43f4-9f89-7d9bf3f7e6e2",
        source_drive_file_id="drive-1",
    )[0]
    row = map_chunk_to_database_row(chunk, embedding_model="BAAI/bge-m3", embedding=[0.1] * 1024)
    assert row["document_id"] == chunk["document_id"]
    assert row["content"] == chunk["text"]
    assert row["page_number"] == chunk["start_page"]
    assert row["end_page"] == chunk["end_page"]
    assert row["embedding_model"] == "BAAI/bge-m3"
    assert len(row["embedding"]) == 1024
    assert "source_drive_file_id" not in row  # joined from legal_documents, not duplicated in DB


class _FakeDocumentCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []
        self.lastrow = None

    def __enter__(self): return self
    def __exit__(self, *_): return False

    def execute(self, sql, params=()):
        if "pg_advisory_xact_lock" in sql:
            return
        if sql.startswith("SELECT"):
            drive_id = params[0]
            self.rows = [(row["id"],) for row in self.connection.documents if row["source_drive_file_id"] == drive_id]
            return
        if sql.startswith("INSERT INTO"):
            cols = [part.strip().strip('"') for part in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
            row = dict(zip(cols, params))
            if '"legal_chunks"' in sql:
                previous = next((item for item in self.connection.chunks
                                 if item["document_id"] == row["document_id"]
                                 and item["chunk_index"] == row["chunk_index"]), None)
                if previous is None:
                    self.connection.chunks.append(row)
                else:
                    previous.update(row)
                return
            row["id"] = str(UUID(int=len(self.connection.documents) + 1))
            self.connection.documents.append(row)
            self.lastrow = (row["id"],)
            return
        raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchall(self): return self.rows
    def fetchone(self): return self.lastrow


class _FakeDocumentConnection:
    def __init__(self):
        self.documents = []
        self.chunks = []
    def transaction(self):
        from contextlib import nullcontext
        return nullcontext()
    def cursor(self): return _FakeDocumentCursor(self)


def _document_store(connection):
    mapping = legal_document_chunk_mapping()
    tables = {
        "legal_documents": {
            name: ColumnInfo(name, "text", "text", True)
            for name in mapping.documents.field_to_column.values()
        },
        "legal_chunks": {
            name: ColumnInfo(name, "vector(1024)" if name == "embedding" else "text", "vector" if name == "embedding" else "text", True)
            for name in mapping.chunks.field_to_column.values()
        },
    }
    tables["legal_documents"]["id"] = ColumnInfo("id", "uuid", "uuid", False)
    tables["legal_documents"]["source_drive_file_id"] = ColumnInfo("source_drive_file_id", "text", "text", True)
    return SupabaseVectorStore(
        connection,
        DatabaseSchema("public", tables, [{
            "table": "legal_chunks", "column": "document_id",
            "references_table": "legal_documents", "references_column": "id",
        }], [
            {"table": "legal_documents", "columns": ["id"]},
            {"table": "legal_chunks", "columns": ["document_id", "chunk_index"]},
        ]),
        mapping,
        vector_dimensions=1024,
    )


def test_document_uuid_resolution_is_idempotent_by_drive_id_not_title():
    connection = _FakeDocumentConnection()
    store = _document_store(connection)
    first = build_legal_document_record({
        "id": "drive-a", "name": "Same title.pdf", "webViewLink": "https://drive.google.com/file/d/drive-a/view"
    })
    second = build_legal_document_record({
        "id": "drive-b", "name": "Same title.pdf", "webViewLink": "https://drive.google.com/file/d/drive-b/view"
    })
    authorized = {"validated": True, "user_approved": True, "dry_run": False}
    uuid_a, created_a = store.resolve_document_uuid(first, **authorized)
    repeated_uuid_a, created_again = store.resolve_document_uuid(first, **authorized)
    uuid_b, created_b = store.resolve_document_uuid(second, **authorized)
    assert (uuid_a, created_a) == (str(UUID(int=1)), True)
    assert (repeated_uuid_a, created_again) == (str(UUID(int=1)), False)
    assert (uuid_b, created_b) == (str(UUID(int=2)), True)
    assert len(connection.documents) == 2
    assert [r["source_drive_file_id"] for r in connection.documents] == ["drive-a", "drive-b"]


def test_document_chunk_mapping_validates_against_uuid_fk_and_1024_vector():
    store = _document_store(_FakeDocumentConnection())
    assert store.validate_schema() == []


def test_source_aware_storage_path_never_uses_drive_id_as_chunk_fk():
    connection = _FakeDocumentConnection()
    store = _document_store(connection)
    source_id = "drive-one"
    source = build_legal_document_record({
        "id": source_id,
        "name": "Verified source title.pdf",
        "webViewLink": "https://drive.google.com/file/d/drive-one/view",
    })
    chunks = [_chunk("provision text", document_id=source_id)]
    vectors = {0: [0.25] * 1024}
    result = store.upsert_source_document_chunks(
        source, chunks, vectors, embedding_model="BAAI/bge-m3",
        validated=True, user_approved=True, dry_run=False,
    )
    assert result["document_uuid"] == str(UUID(int=1))
    assert connection.documents[0]["source_drive_file_id"] == source_id
    assert connection.chunks[0]["document_id"] == result["document_uuid"]
    assert connection.chunks[0]["document_id"] != source_id
    assert connection.chunks[0]["content"] == "provision text"
    again = store.upsert_source_document_chunks(
        source, chunks, vectors, embedding_model="BAAI/bge-m3",
        validated=True, user_approved=True, dry_run=False,
    )
    assert again["document_uuid"] == result["document_uuid"]
    assert again["document_created"] is False
    assert len(connection.documents) == 1
    assert len(connection.chunks) == 1


class _FakeEmbedClient:
    def __init__(self):
        self.calls = []
        self.models = self

    def embed_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.25] * 768) for _ in contents])


def test_embedder_batches_caches_and_reuses_unchanged_text(tmp_path):
    client = _FakeEmbedClient()
    embedder = GeminiEmbedder(client=client, min_interval_seconds=0, batch_size=2)
    chunks = [_chunk("same text", chunk_index=i) for i in range(3)]
    cache = tmp_path / "cache.json"
    first = embedder.embed_chunks(chunks, cache_path=cache)
    assert len(first) == 3
    assert len(first[0]["embedding"]) == 768
    assert first[0]["task_type"] == "RETRIEVAL_DOCUMENT"
    assert len(client.calls) == 1  # repeated text is represented by one content input
    second = embedder.embed_chunks(chunks, cache_path=cache)
    assert len(second) == 3
    assert len(client.calls) == 1
    assert json.loads(cache.read_text())["embeddings"]


def test_embedder_rejects_non_768_dimensions():
    with pytest.raises(ValueError, match="768"):
        GeminiEmbedder(client=_FakeEmbedClient(), dimensions=512)


class _FakeBgeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {"dense_vecs": [[0.5] * 1024 for _ in texts]}


def test_local_bge_m3_embedder_is_resumable_and_validates_dimensions(tmp_path):
    model = _FakeBgeModel()
    embedder = BgeM3Embedder(model=model, batch_size=2)
    cache = tmp_path / "bge-cache.json"
    chunks = [_chunk("one", chunk_index=1), _chunk("two", chunk_index=2)]
    result = embedder.embed_chunks(chunks, cache_path=cache)
    assert len(result[0]["embedding"]) == 1024
    assert sum(value * value for value in result[0]["embedding"]) == pytest.approx(1.0)
    assert result[0]["model"] == "BAAI/bge-m3"
    assert model.calls[0][1]["max_length"] == 8192
    assert len(embedder.embed_query("legal query")) == 1024
    call_count = len(model.calls)
    embedder.embed_chunks(chunks, cache_path=cache)
    assert len(model.calls) == call_count  # unchanged document text reuses its cache
    restarted = BgeM3Embedder(model=model, batch_size=2)
    restarted.embed_chunks(chunks, cache_path=cache)
    assert len(model.calls) == call_count  # the append-only checkpoint survives a new process/embedder
    journal = BgeM3Embedder._journal_path(cache)
    assert journal.exists()
    BgeM3Embedder.compact_cache(cache)
    assert not journal.exists()
    assert json.loads(cache.read_text())["embeddings"]


def test_local_bge_m3_splits_oom_batches_and_normalizes():
    class OomOnPair(_FakeBgeModel):
        def encode(self, texts, **kwargs):
            if len(texts) > 1:
                raise RuntimeError("CUDA out of memory")
            return {"dense_vecs": [[0.5] * 1024 for _ in texts]}

    embedder = BgeM3Embedder(model=OomOnPair(), batch_size=4)
    vectors = embedder._embed_batch(["a", "b", "c", "d"])
    assert len(vectors) == 4
    assert all(sum(value * value for value in vector) == pytest.approx(1.0) for vector in vectors)


def test_local_bge_m3_rejects_nonfinite_embedding_values():
    class NonFinite(_FakeBgeModel):
        def encode(self, texts, **kwargs):
            vector = [0.5] * 1024
            vector[0] = float("nan")
            return {"dense_vecs": [vector for _ in texts]}

    embedder = BgeM3Embedder(model=NonFinite())
    with pytest.raises(ValueError, match="non-finite"):
        embedder._embed_batch(["bad"])


def test_vector_mapping_checks_existing_columns_vector_width_and_unique_key():
    schema = DatabaseSchema(
        "public",
        {"chunks": {
            "id": ColumnInfo("id", "uuid", "uuid", False),
            "embedding": ColumnInfo("embedding", "vector(768)", "vector", True),
        }},
        [],
        [{"table": "chunks", "columns": ["id"]}],
    )
    mapping = TableMapping("chunks", {"chunk_id": "id", "embedding": "embedding"}, ("id",), "embedding")
    assert validate_mapping(schema, mapping) == []
    wrong = TableMapping("chunks", {"embedding": "embedding"}, ("embedding",), "embedding")
    assert any("unique/primary-key" in issue for issue in validate_mapping(schema, wrong))


def test_vector_store_requires_validation_and_approval():
    schema = DatabaseSchema("public", {"chunks": {"id": ColumnInfo("id", "uuid", "uuid", False)}}, [], [{"table": "chunks", "columns": ["id"]}])
    mapping = TableMapping("chunks", {"chunk_id": "id"}, ("id",))
    with pytest.raises(RuntimeError, match="validation"):
        upsert_rows(object(), schema, mapping, [{"chunk_id": "x"}])
    with pytest.raises(RuntimeError, match="explicit user_approved"):
        upsert_rows(object(), schema, mapping, [{"chunk_id": "x"}], validated=True)


def test_document_processor_orchestrates_without_embedding_or_database_write(tmp_path, monkeypatch):
    source = tmp_path / "sample-act.pdf"
    source.write_bytes(b"test source bytes")
    monkeypatch.setattr(document_processor, "extract_pdf_text", lambda _: [
        {"page_number": 1, "text": "PART I\nPRELIMINARY\n1. Short title\n(1) This Act may be called Sample Act."}
    ])
    result = document_processor.process_document(
        source,
        document_id="drive-id",
        metadata={"jurisdiction": "Kerala", "document_type": "Act", "source_drive_id": "drive-id"},
    )
    assert result["validated"]
    assert result["dry_run"]
    assert result["document"]["document_id"] == "drive-id"
    assert result["chunks"][0]["document_title"] == "sample-act"
    assert result["chunks"][0]["jurisdiction"] == "central"
    assert result["chunks"][0]["original_source"]["source_sha256"] == result["source_sha256"]
