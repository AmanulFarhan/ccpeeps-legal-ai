"""Resumable local BGE-M3 and Gemini embedding services.

Embedding generation is deliberately separate from persistence. BGE-M3 emits
1024-dimensional dense vectors; never truncate them to fit an incompatible DB.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

from google import genai
from google.genai import types


class GeminiEmbedder:
    def __init__(
        self,
        *,
        client=None,
        model: str | None = None,
        dimensions: int = 768,
        batch_size: int = 64,
        min_interval_seconds: float = 0.5,
        max_retries: int = 5,
        api_key: str | None = None,
    ):
        self.model = model or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.dimensions = dimensions
        self.batch_size = max(1, batch_size)
        self.min_interval_seconds = max(0, min_interval_seconds)
        self.max_retries = max(0, max_retries)
        if dimensions != 768:
            raise ValueError("This project requires 768-dimensional embeddings (Supabase VECTOR(768)).")
        if client is None:
            key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before generating embeddings.")
            client = genai.Client(api_key=key)
        self.client = client
        self._last_request = 0.0

    def _fingerprint(self, text: str) -> str:
        payload = f"{self.model}\0RETRIEVAL_DOCUMENT\0{self.dimensions}\0{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _read_cache(path: Path) -> dict[str, list[float]]:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {key: value for key, value in raw.get("embeddings", {}).items() if isinstance(value, list)}

    @staticmethod
    def _write_cache(path: Path, cache: dict[str, list[float]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            json.dump({"embeddings": cache}, handle, ensure_ascii=False)
            temporary = Path(handle.name)
        temporary.replace(path)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code is not None:
            try:
                return int(code) in {408, 429} or int(code) >= 500
            except (TypeError, ValueError):
                pass
        return any(term in type(exc).__name__.lower() for term in ("timeout", "connection", "temporar"))

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last_error = None
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_seconds - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=self.dimensions,
                    ),
                )
                vectors = [list(item.values) for item in response.embeddings]
                if len(vectors) != len(texts):
                    raise ValueError(f"Gemini returned {len(vectors)} vectors for {len(texts)} inputs.")
                if any(len(vector) != self.dimensions for vector in vectors):
                    actual = sorted({len(vector) for vector in vectors})
                    raise ValueError(f"Expected {self.dimensions}-dimensional vectors, received {actual}.")
                return vectors
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries or not self._retryable(exc):
                    raise
                time.sleep(min(60.0, (2**attempt) + random.random()))
        raise RuntimeError("Embedding request exhausted retries.") from last_error

    def embed_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        cache_path: str | Path,
        on_progress=None,
    ) -> list[dict[str, Any]]:
        """Embed changed text only. Cache is saved after every successful batch."""
        path = Path(cache_path)
        cache = self._read_cache(path)
        fingerprints = [self._fingerprint(str(chunk.get("text", ""))) for chunk in chunks]
        missing = list(dict.fromkeys(fp for fp in fingerprints if fp not in cache))
        for offset in range(0, len(missing), self.batch_size):
            batch_keys = missing[offset : offset + self.batch_size]
            text_by_key = {}
            for fingerprint, chunk in zip(fingerprints, chunks):
                if fingerprint in batch_keys and fingerprint not in text_by_key:
                    text_by_key[fingerprint] = str(chunk.get("text", ""))
            vectors = self._embed_batch([text_by_key[key] for key in batch_keys])
            cache.update(dict(zip(batch_keys, vectors)))
            self._write_cache(path, cache)
            if on_progress:
                on_progress(min(offset + len(batch_keys), len(missing)), len(missing))
        return [
            {
                "chunk_index": chunk.get("chunk_index"),
                "document_id": chunk.get("document_id"),
                "fingerprint": fingerprint,
                "model": self.model,
                "dimensions": self.dimensions,
                "task_type": "RETRIEVAL_DOCUMENT",
                "embedding": cache[fingerprint],
            }
            for chunk, fingerprint in zip(chunks, fingerprints)
        ]


class BgeM3Embedder:
    """Local dense embeddings using BAAI/bge-m3 through FlagEmbedding."""

    model_name = "BAAI/bge-m3"
    dimensions = 1024
    max_length = 8192

    def __init__(self, *, model=None, batch_size: int = 8, use_fp16: bool | None = None):
        self.batch_size = max(1, batch_size)
        self.use_fp16 = use_fp16
        self._injected_model = model is not None
        self._cpu_fallback_model = None
        self._cache: dict[str, list[float]] | None = None
        local_default = Path.home() / ".cache" / "bge-m3-local"
        self.model_source = os.getenv(
            "BGE_M3_MODEL_PATH",
            str(local_default) if (local_default / "pytorch_model.bin").is_file() else self.model_name,
        )
        if model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise RuntimeError(
                    "Install FlagEmbedding and its PyTorch dependencies to use local BGE-M3."
                ) from exc
            if use_fp16 is None:
                try:
                    import torch
                    use_fp16 = bool(torch.cuda.is_available())
                except ImportError:
                    use_fp16 = False
            self._model_class = BGEM3FlagModel
            model = BGEM3FlagModel(
                self.model_source,
                use_fp16=use_fp16,
                normalize_embeddings=False,
            )
        self.model = model

    def _fingerprint(self, text: str) -> str:
        payload = f"{self.model_name}\0dense\0{self.dimensions}\0{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _journal_path(path: Path) -> Path:
        return path.with_name(path.name + ".journal")

    @classmethod
    def _read_resumable_cache(cls, path: Path) -> dict[str, list[float]]:
        cache = GeminiEmbedder._read_cache(path)
        journal = cls._journal_path(path)
        if not journal.exists():
            return cache
        lines = journal.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            try:
                delta = json.loads(line).get("embeddings", {})
            except (json.JSONDecodeError, AttributeError):
                if index == len(lines) - 1:
                    # An interrupted final append is safe to recompute.
                    continue
                raise ValueError(f"Corrupt BGE-M3 checkpoint journal at line {index + 1}.")
            cache.update({key: value for key, value in delta.items() if isinstance(value, list)})
        return cache

    @classmethod
    def _append_cache_delta(cls, path: Path, delta: dict[str, list[float]]) -> None:
        journal = cls._journal_path(path)
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"embeddings": delta}, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @classmethod
    def compact_cache(cls, cache_path: str | Path) -> None:
        path = Path(cache_path)
        cache = cls._read_resumable_cache(path)
        GeminiEmbedder._write_cache(path, cache)
        cls._journal_path(path).unlink(missing_ok=True)

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        return "out of memory" in str(exc).lower() or "cuda oom" in type(exc).__name__.lower()

    def _encode(self, model, texts: list[str], batch_size: int) -> list[list[float]]:
        result = model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vectors = result["dense_vecs"]
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        vectors = [[float(value) for value in vector] for vector in vectors]
        if len(vectors) != len(texts) or any(len(vector) != self.dimensions for vector in vectors):
            got = sorted({len(vector) for vector in vectors})
            raise ValueError(f"BGE-M3 must return one 1024-dimensional vector per text; got {got}.")
        normalized = []
        for vector in vectors:
            if any(not math.isfinite(value) for value in vector):
                raise ValueError("BGE-M3 returned a non-finite value.")
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0:
                raise ValueError("BGE-M3 returned a zero vector; cannot normalize it.")
            normalized.append([value / norm for value in vector])
        vectors = normalized
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._encode(self.model, texts, min(self.batch_size, len(texts)))
        except Exception as exc:
            if not self._is_oom(exc):
                raise
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            if len(texts) > 1:
                midpoint = len(texts) // 2
                return self._embed_batch(texts[:midpoint]) + self._embed_batch(texts[midpoint:])
            if self._injected_model:
                raise RuntimeError("BGE-M3 ran out of memory on one text; CPU fallback unavailable for injected model.") from exc
            if self._cpu_fallback_model is None:
                self._cpu_fallback_model = self._model_class(
                    self.model_source, use_fp16=False, devices="cpu", normalize_embeddings=False
                )
            return self._encode(self._cpu_fallback_model, texts, 1)

    def embed_query(self, text: str) -> list[float]:
        """Encode a search query in the same dense vector space as documents."""
        return self._embed_batch([text])[0]

    def embed_chunks(self, chunks: list[dict[str, Any]], *, cache_path: str | Path, on_progress=None):
        """Embed locally with a persistent content-addressed cache for resuming."""
        path = Path(cache_path)
        if self._cache is None:
            self._cache = self._read_resumable_cache(path)
        cache = self._cache
        fingerprints = [self._fingerprint(str(chunk.get("text", ""))) for chunk in chunks]
        missing = list(dict.fromkeys(fp for fp in fingerprints if fp not in cache))
        for offset in range(0, len(missing), self.batch_size):
            keys = missing[offset : offset + self.batch_size]
            texts_by_key = {}
            for fingerprint, chunk in zip(fingerprints, chunks):
                if fingerprint in keys and fingerprint not in texts_by_key:
                    texts_by_key[fingerprint] = str(chunk.get("text", ""))
            vectors = self._embed_batch([texts_by_key[key] for key in keys])
            delta = dict(zip(keys, vectors))
            cache.update(delta)
            self._append_cache_delta(path, delta)
            if on_progress:
                on_progress(min(offset + len(keys), len(missing)), len(missing))
        return [
            {
                "chunk_index": chunk.get("chunk_index"),
                "document_id": chunk.get("document_id"),
                "fingerprint": fingerprint,
                "model": self.model_name,
                "dimensions": self.dimensions,
                "task_type": "dense",
                "embedding": cache[fingerprint],
            }
            for chunk, fingerprint in zip(chunks, fingerprints)
        ]
