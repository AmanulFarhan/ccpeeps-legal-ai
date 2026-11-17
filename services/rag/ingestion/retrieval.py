"""Database-side hybrid retrieval for the legal_chunks pgvector table.

Only the top semantic and keyword candidates are transferred to Python. Vector
nearest-neighbour search is delegated to PostgreSQL so an existing HNSW index
can be used; ranked lists are fused with reciprocal rank fusion (RRF).
"""

from __future__ import annotations

import math
from typing import Any


_RESULT_COLUMNS = (
    "document_id", "chunk_index", "section_number", "section_title",
    "subsection_number", "clause", "chapter", "page_number", "end_page",
    "content", "title", "citation", "document_type", "jurisdiction",
    "source_url",
)


class LegalRetriever:
    """Hybrid retriever that queries pgvector and PostgreSQL full-text search.

    Args:
        connection: Open psycopg connection to the database holding
            ``public.legal_chunks`` and ``public.legal_documents``.
        embedder: Object exposing ``embed_query(text)``. Use the same BGE-M3
            model used to create the stored 1024-dimensional chunk vectors.

    The connection is borrowed, not closed by this class. The keyword query
    uses PostgreSQL's ``simple`` text configuration to retain legal terms and
    identifiers such as section numbers.
    """

    def __init__(self, connection, embedder):
        if connection is None:
            raise ValueError("A PostgreSQL connection is required.")
        if not callable(getattr(embedder, "embed_query", None)):
            raise ValueError("embedder must provide embed_query(text).")
        self.connection = connection
        self.embedder = embedder

    @staticmethod
    def _as_dicts(cursor, rows) -> list[dict[str, Any]]:
        if not rows:
            return []
        if isinstance(rows[0], dict):
            return [dict(row) for row in rows]
        names = [column.name if hasattr(column, "name") else column[0]
                 for column in cursor.description]
        return [dict(zip(names, row)) for row in rows]

    @staticmethod
    def _vector_literal(vector) -> str:
        values = [float(value) for value in vector]
        if len(values) != 1024:
            raise ValueError(f"Expected a 1024-dimensional BGE-M3 query vector, got {len(values)}.")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Query embedding contains a non-finite value.")
        if not any(values):
            raise ValueError("Query embedding is a zero vector.")
        return "[" + ",".join(str(value) for value in values) + "]"

    def _semantic_candidates(self, vector: str, limit: int) -> list[dict[str, Any]]:
        sql = """
            SELECT c.document_id, c.chunk_index, c.section_number,
                   c.section_title, c.subsection_number, c.clause, c.chapter,
                   c.page_number, c.end_page, c.content,
                   d.title, d.citation, d.document_type, d.jurisdiction,
                   d.source_url, (c.embedding <=> %s::vector) AS vector_distance
            FROM public.legal_chunks AS c
            LEFT JOIN public.legal_documents AS d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        with self.connection.cursor() as cursor:
            cursor.execute(sql, (vector, vector, limit))
            return self._as_dicts(cursor, cursor.fetchall())

    def _keyword_candidates(self, query: str, limit: int) -> list[dict[str, Any]]:
        sql = """
            WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS query)
            SELECT c.document_id, c.chunk_index, c.section_number,
                   c.section_title, c.subsection_number, c.clause, c.chapter,
                   c.page_number, c.end_page, c.content,
                   d.title, d.citation, d.document_type, d.jurisdiction,
                   d.source_url,
                   ts_rank_cd(to_tsvector('simple', c.content), q.query) AS text_score
            FROM public.legal_chunks AS c
            LEFT JOIN public.legal_documents AS d ON d.id = c.document_id
            CROSS JOIN q
            WHERE to_tsvector('simple', c.content) @@ q.query
            ORDER BY text_score DESC, c.document_id, c.chunk_index
            LIMIT %s
        """
        with self.connection.cursor() as cursor:
            cursor.execute(sql, (query, limit))
            return self._as_dicts(cursor, cursor.fetchall())

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        candidate_k: int = 40,
        rrf_k: int = 60,
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Return fused semantic and keyword results without loading the corpus."""
        if not isinstance(query, str) or not query.strip():
            return []
        if top_k <= 0 or candidate_k <= 0 or rrf_k <= 0:
            return []
        if semantic_weight < 0 or keyword_weight < 0:
            raise ValueError("Search weights must be non-negative.")

        vector = self._vector_literal(self.embedder.embed_query(query.strip()))
        semantic = self._semantic_candidates(vector, candidate_k)
        keyword = self._keyword_candidates(query.strip(), candidate_k)

        fused: dict[tuple[str, int], dict[str, Any]] = {}
        for source, weight, score_key in (
            (semantic, semantic_weight, "semantic_score"),
            (keyword, keyword_weight, "keyword_score"),
        ):
            for rank, chunk in enumerate(source, start=1):
                identity = (str(chunk["document_id"]), int(chunk["chunk_index"]))
                item = fused.setdefault(identity, {**chunk, "rrf_score": 0.0})
                item["rrf_score"] += weight / (rrf_k + rank)
                item[score_key] = chunk.get("vector_distance" if score_key == "semantic_score" else "text_score")
                item["semantic_rank" if score_key == "semantic_score" else "keyword_rank"] = rank

        return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)[:top_k]
