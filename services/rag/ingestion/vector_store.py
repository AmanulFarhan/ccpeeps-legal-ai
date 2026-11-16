"""Schema-checked PostgreSQL/pgvector writer with explicit write approval gates.

No table or column names are assumed. Inspect the live database, define a mapping
against its existing schema, validate the mapping, and only then enable writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Any


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    data_type: str
    udt_name: str
    nullable: bool
    character_maximum_length: int | None = None


@dataclass
class DatabaseSchema:
    schema_name: str
    tables: dict[str, dict[str, ColumnInfo]]
    foreign_keys: list[dict[str, str]]
    unique_constraints: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TableMapping:
    table: str
    field_to_column: dict[str, str]
    conflict_columns: tuple[str, ...]
    embedding_field: str | None = None


@dataclass(frozen=True)
class DocumentChunkMapping:
    documents: TableMapping
    chunks: TableMapping
    document_id_field: str = "document_id"
    document_primary_key_field: str = "id"


def connect_from_env():
    url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("Set SUPABASE_DB_URL or DATABASE_URL to inspect/use the existing PostgreSQL schema.")
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install psycopg[binary] to connect to Supabase PostgreSQL.") from exc
    return psycopg.connect(url)


def inspect_schema(connection=None, schema_name: str = "public") -> DatabaseSchema:
    """Read actual column and FK metadata; caller owns/closes injected connections."""
    owns_connection = connection is None
    connection = connection or connect_from_env()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT table_name, column_name, data_type, udt_name, is_nullable,
                          character_maximum_length
                   FROM information_schema.columns
                   WHERE table_schema = %s ORDER BY table_name, ordinal_position""",
                (schema_name,),
            )
            tables: dict[str, dict[str, ColumnInfo]] = {}
            for table, name, data_type, udt_name, nullable, length in cursor.fetchall():
                tables.setdefault(table, {})[name] = ColumnInfo(
                    name, data_type, udt_name, nullable == "YES", length
                )
            cursor.execute(
                """SELECT tc.table_name, kcu.column_name, ccu.table_name,
                          ccu.column_name
                   FROM information_schema.table_constraints tc
                   JOIN information_schema.key_column_usage kcu
                     ON tc.constraint_name = kcu.constraint_name
                    AND tc.constraint_schema = kcu.constraint_schema
                   JOIN information_schema.constraint_column_usage ccu
                     ON ccu.constraint_name = tc.constraint_name
                    AND ccu.constraint_schema = tc.constraint_schema
                   WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.constraint_schema = %s""",
                (schema_name,),
            )
            fks = [dict(zip(("table", "column", "references_table", "references_column"), row)) for row in cursor.fetchall()]
            cursor.execute(
                """SELECT tc.table_name, tc.constraint_name, tc.constraint_type,
                          array_agg(kcu.column_name::text ORDER BY kcu.ordinal_position)
                   FROM information_schema.table_constraints tc
                   JOIN information_schema.key_column_usage kcu
                     ON tc.constraint_name = kcu.constraint_name
                    AND tc.constraint_schema = kcu.constraint_schema
                   WHERE tc.constraint_schema = %s
                     AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
                   GROUP BY tc.table_name, tc.constraint_name, tc.constraint_type""",
                (schema_name,),
            )
            unique_constraints = [dict(zip(("table", "name", "type", "columns"), row)) for row in cursor.fetchall()]
            cursor.execute(
                """SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod)
                   FROM pg_attribute a
                   JOIN pg_class c ON c.oid = a.attrelid
                   JOIN pg_namespace n ON n.oid = c.relnamespace
                   WHERE n.nspname = %s AND a.attnum > 0 AND NOT a.attisdropped
                     AND a.atttypid = 'vector'::regtype""",
                (schema_name,),
            )
            for table, column, formatted_type in cursor.fetchall():
                if table in tables and column in tables[table]:
                    info = tables[table][column]
                    tables[table][column] = ColumnInfo(info.name, formatted_type, info.udt_name, info.nullable, info.character_maximum_length)
        return DatabaseSchema(schema_name, tables, fks, unique_constraints)
    finally:
        if owns_connection:
            connection.close()


def validate_mapping(schema: DatabaseSchema, mapping: TableMapping, *, vector_dimensions: int = 768) -> list[str]:
    errors = []
    columns = schema.tables.get(mapping.table)
    if columns is None:
        return [f"Table {schema.schema_name}.{mapping.table} does not exist in the inspected schema."]
    selected = set(mapping.field_to_column.values())
    for col in selected - columns.keys():
        errors.append(f"Mapped column {mapping.table}.{col} is absent from the inspected schema.")
    for col in mapping.conflict_columns:
        if col not in columns:
            errors.append(f"Conflict column {mapping.table}.{col} is absent from the inspected schema.")
    if not mapping.conflict_columns:
        errors.append("At least one existing unique/primary-key column must be configured for upsert.")
    elif not any(
        constraint["table"] == mapping.table
        and set(constraint["columns"]) == set(mapping.conflict_columns)
        for constraint in schema.unique_constraints
    ):
        errors.append("Configured conflict columns do not match an inspected unique/primary-key constraint.")
    if mapping.embedding_field:
        vector_col = mapping.field_to_column.get(mapping.embedding_field)
        info = columns.get(vector_col) if vector_col else None
        if info is None:
            errors.append("Embedding field must map to an existing vector column.")
        elif info.udt_name != "vector":
            errors.append(f"{mapping.table}.{vector_col} has type {info.data_type}/{info.udt_name}, not pgvector.")
        else:
            match = re.search(r"vector\((\d+)\)", info.data_type, re.I)
            if not match:
                errors.append(f"Could not verify the dimension of {mapping.table}.{vector_col} from PostgreSQL metadata.")
            elif int(match.group(1)) != vector_dimensions:
                errors.append(f"{mapping.table}.{vector_col} dimension is {match.group(1)}, expected {vector_dimensions}.")
    return errors


def _quote_identifier(identifier: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return '"' + identifier + '"'


def upsert_rows(
    connection,
    schema: DatabaseSchema,
    mapping: TableMapping,
    rows: list[dict[str, Any]],
    *,
    validated: bool = False,
    user_approved: bool = False,
    dry_run: bool = True,
    vector_dimensions: int = 768,
) -> int:
    """Write mapped records only after schema validation and explicit authorization."""
    errors = validate_mapping(schema, mapping, vector_dimensions=vector_dimensions)
    if errors:
        raise ValueError("Schema mapping is not valid: " + "; ".join(errors))
    if not validated:
        raise RuntimeError("Chunk validation must pass before vector-store writes.")
    if dry_run or not user_approved:
        raise RuntimeError("Writes require dry_run=False and explicit user_approved=True.")
    if not rows:
        return 0

    db_columns = list(dict.fromkeys(mapping.field_to_column.values()))
    column_to_field = {column: field for field, column in mapping.field_to_column.items()}
    for row in rows:
        absent = set(mapping.field_to_column) - row.keys()
        if absent:
            raise ValueError(f"Input row lacks mapped fields: {sorted(absent)}")
    columns_sql = ", ".join(_quote_identifier(name) for name in db_columns)
    placeholders = []
    for name in db_columns:
        if mapping.embedding_field and mapping.field_to_column[mapping.embedding_field] == name:
            placeholders.append("%s::vector")
        else:
            placeholders.append("%s")
    conflict = ", ".join(_quote_identifier(name) for name in mapping.conflict_columns)
    updates = [name for name in db_columns if name not in mapping.conflict_columns]
    action = ("DO UPDATE SET " + ", ".join(f"{_quote_identifier(name)} = EXCLUDED.{_quote_identifier(name)}" for name in updates)) if updates else "DO NOTHING"
    sql = f"INSERT INTO {_quote_identifier(schema.schema_name)}.{_quote_identifier(mapping.table)} ({columns_sql}) VALUES ({', '.join(placeholders)}) ON CONFLICT ({conflict}) {action}"
    with connection.cursor() as cursor:
        for row in rows:
            params = []
            for column in db_columns:
                value = row[column_to_field[column]]
                if mapping.embedding_field and column == mapping.field_to_column[mapping.embedding_field]:
                    if not isinstance(value, (list, tuple)) or len(value) != vector_dimensions:
                        raise ValueError(f"Embedding must contain exactly {vector_dimensions} values.")
                    value = "[" + ",".join(str(float(v)) for v in value) + "]"
                params.append(value)
            cursor.execute(sql, params)
    return len(rows)


class SupabaseVectorStore:
    """Explicitly mapped, live-schema-checked writer for an existing database."""

    def __init__(self, connection, schema: DatabaseSchema, mapping: DocumentChunkMapping, *, vector_dimensions: int = 768):
        self.connection = connection
        self.schema = schema
        self.mapping = mapping
        self.vector_dimensions = vector_dimensions

    def validate_schema(self) -> list[str]:
        errors = validate_mapping(self.schema, self.mapping.documents)
        errors.extend(validate_mapping(self.schema, self.mapping.chunks, vector_dimensions=self.vector_dimensions))
        document_column = self.mapping.documents.field_to_column.get(self.mapping.document_primary_key_field)
        chunk_column = self.mapping.chunks.field_to_column.get(self.mapping.document_id_field)
        if not document_column or not chunk_column:
            errors.append(
                f"Document mapping must include {self.mapping.document_primary_key_field!r} and "
                f"chunk mapping must include {self.mapping.document_id_field!r}."
            )
        elif not any(
            fk["table"] == self.mapping.chunks.table
            and fk["column"] == chunk_column
            and fk["references_table"] == self.mapping.documents.table
            and fk["references_column"] == document_column
            for fk in self.schema.foreign_keys
        ):
            errors.append("No inspected foreign key connects the configured chunk document_id to the document table.")
        return errors

    def _resolve_document_uuid_in_transaction(self, document_record: dict[str, Any]) -> tuple[str, bool]:
        """Lookup or create by Drive ID while serializing concurrent pipeline runs.

        Supabase currently has no UNIQUE constraint on source_drive_file_id.
        A transaction-scoped advisory lock makes this lookup/insert idempotent
        among writers that use this method, without changing the schema.
        """
        document_map = self.mapping.documents
        drive_field = "source_drive_file_id"
        drive_id = document_record.get(drive_field)
        if not isinstance(drive_id, str) or not drive_id.strip():
            raise ValueError("A non-empty source_drive_file_id is required to resolve a document UUID.")
        drive_column = document_map.field_to_column.get(drive_field)
        pk_column = document_map.field_to_column.get(self.mapping.document_primary_key_field)
        columns = self.schema.tables.get(document_map.table, {})
        if not drive_column or drive_column not in columns:
            raise ValueError("Document table mapping must include the inspected source_drive_file_id column.")
        if not pk_column or pk_column not in columns:
            raise ValueError("Document table mapping must include its inspected UUID primary key column.")
        table_sql = f"{_quote_identifier(self.schema.schema_name)}.{_quote_identifier(document_map.table)}"
        drive_sql = _quote_identifier(drive_column)
        pk_sql = _quote_identifier(pk_column)
        with self.connection.cursor() as cursor:
            # This lock is transaction-scoped. It serializes retries for this ID
            # even though the inspected schema has no unique source-ID key.
            cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (drive_id,))
            cursor.execute(f"SELECT {pk_sql} FROM {table_sql} WHERE {drive_sql} = %s FOR UPDATE", (drive_id,))
            found = cursor.fetchall()
            if len(found) > 1:
                raise ValueError(f"Multiple legal_documents rows already use source_drive_file_id {drive_id!r}.")
            if found:
                row = found[0]
                document_uuid = row[0] if not isinstance(row, dict) else row[pk_column]
                return str(document_uuid), False

            insert_fields = [
                field for field in document_map.field_to_column
                if field != self.mapping.document_primary_key_field
                and field in document_record
                and document_map.field_to_column[field] in columns
            ]
            if drive_field not in insert_fields:
                raise ValueError("source_drive_file_id must be included in the mapped insert fields.")
            db_columns = [document_map.field_to_column[field] for field in insert_fields]
            columns_sql = ", ".join(_quote_identifier(column) for column in db_columns)
            placeholders = ", ".join(["%s"] * len(insert_fields))
            values = [document_record[field] for field in insert_fields]
            cursor.execute(
                f"INSERT INTO {table_sql} ({columns_sql}) VALUES ({placeholders}) RETURNING {pk_sql}",
                values,
            )
            inserted = cursor.fetchone()
            if inserted is None:
                raise RuntimeError("Document insert returned no UUID primary key.")
            document_uuid = inserted[0] if not isinstance(inserted, dict) else inserted[pk_column]
            return str(document_uuid), True

    def resolve_document_uuid(
        self,
        document_record: dict[str, Any],
        *,
        validated: bool = False,
        user_approved: bool = False,
        dry_run: bool = True,
    ) -> tuple[str, bool]:
        """Resolve by Drive ID under the same explicit write gates as ingestion."""
        if not validated:
            raise RuntimeError("Document creation requires validated ingestion metadata.")
        if dry_run or not user_approved:
            raise RuntimeError("Document creation requires dry_run=False and explicit user_approved=True.")
        if self.connection is None:
            raise RuntimeError("A PostgreSQL connection is required to resolve a document UUID.")
        with self.connection.transaction():
            return self._resolve_document_uuid_in_transaction(document_record)

    def upsert_source_document_chunks(
        self,
        document_record: dict[str, Any],
        source_chunks: list[dict[str, Any]],
        embeddings_by_chunk_index: dict[int, list[float]],
        *,
        embedding_model: str,
        validated: bool,
        user_approved: bool = False,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Atomically resolve a document UUID and upsert its UUID-linked chunks.

        This is the source-aware write path. The Drive file ID is used only to
        find/create ``legal_documents``; ``legal_chunks.document_id`` always
        receives the resulting UUID. Callers must pass already validated
        chunks and explicitly approve writes.
        """
        errors = self.validate_schema()
        if errors:
            raise ValueError("Live schema mapping failed: " + "; ".join(errors))
        if not validated:
            raise RuntimeError("Refusing to write records before all chunks pass validation.")
        if dry_run or not user_approved:
            raise RuntimeError("Writes require dry_run=False and explicit user_approved=True.")
        if not source_chunks:
            raise ValueError("At least one accepted source chunk is required.")
        if not embedding_model:
            raise ValueError("Embedding model identifier is required.")

        indexes = [chunk.get("chunk_index") for chunk in source_chunks]
        if any(not isinstance(index, int) for index in indexes) or len(indexes) != len(set(indexes)):
            raise ValueError("Source chunks must have unique integer chunk_index values.")
        missing_embeddings = set(indexes) - embeddings_by_chunk_index.keys()
        if missing_embeddings:
            raise ValueError(f"Missing embeddings for chunk indices: {sorted(missing_embeddings)}")
        extra_embeddings = embeddings_by_chunk_index.keys() - set(indexes)
        if extra_embeddings:
            raise ValueError(f"Embeddings reference absent chunk indices: {sorted(extra_embeddings)}")

        from services.rag.ingestion.document_mapping import (
            map_chunk_to_database_row,
            map_chunks_to_document_uuid,
        )

        with self.connection.transaction():
            document_uuid, created = self._resolve_document_uuid_in_transaction(document_record)
            drive_id = document_record["source_drive_file_id"]
            linked_chunks = map_chunks_to_document_uuid(
                source_chunks,
                document_uuid=document_uuid,
                source_drive_file_id=drive_id,
            )
            chunk_rows = [
                map_chunk_to_database_row(
                    chunk,
                    embedding_model=embedding_model,
                    embedding=embeddings_by_chunk_index[chunk["chunk_index"]],
                )
                for chunk in linked_chunks
            ]
            written = upsert_rows(
                self.connection,
                self.schema,
                self.mapping.chunks,
                chunk_rows,
                validated=True,
                user_approved=True,
                dry_run=False,
                vector_dimensions=self.vector_dimensions,
            )
        return {"document_uuid": document_uuid, "document_created": created, "chunks_upserted": written}

    def upsert_document_chunks(
        self,
        document_rows: list[dict[str, Any]],
        chunk_rows: list[dict[str, Any]],
        *,
        validated: bool,
        user_approved: bool = False,
        dry_run: bool = True,
    ) -> dict[str, int]:
        errors = self.validate_schema()
        if errors:
            raise ValueError("Live schema mapping failed: " + "; ".join(errors))
        if not validated:
            raise RuntimeError("Refusing to write records before all chunks pass validation.")
        if dry_run or not user_approved:
            raise RuntimeError("Writes require dry_run=False and explicit user_approved=True.")
        # psycopg transaction contexts make document and chunk upserts atomic.
        with self.connection.transaction():
            documents_written = upsert_rows(
                self.connection, self.schema, self.mapping.documents, document_rows,
                validated=True, user_approved=True, dry_run=False,
            )
            chunks_written = upsert_rows(
                self.connection, self.schema, self.mapping.chunks, chunk_rows,
                validated=True, user_approved=True, dry_run=False,
                vector_dimensions=self.vector_dimensions,
            )
        return {"documents": documents_written, "chunks": chunks_written}
