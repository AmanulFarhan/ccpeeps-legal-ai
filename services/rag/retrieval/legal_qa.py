"""Application service for hybrid retrieval and Gemini legal Q&A."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from services.rag.ingestion.embedder import BgeM3Embedder
from services.rag.retrieval.retrieval import LegalRetriever
from services.rag.ingestion.vector_store import connect_from_env


_LEGAL_SYSTEM_PROMPT = """You are a careful Indian legal information assistant teaching a reader who has no previous legal knowledge.

Answer using only the retrieved legal passages supplied in the user's message. Your goal is to help the reader understand the law, not merely list provisions.

ACCURACY AND SCOPE RULES:
1. Do not treat related legal terms as identical. For example, "human rights" is broader than "Fundamental Rights". Explain the distinction when it matters and state which part the retrieved documents support.
2. Never imply that the retrieved documents are the whole of Indian law. For a broad question, clearly state the scope of the available evidence.
3. Use only the supplied passages for legal claims. Do not invent or add laws, sections, cases, penalties, procedures, remedies, or facts.
4. Preserve every important condition, exception, qualification, and limitation found in the passages.
5. Cite each substantive legal claim immediately with one or more supplied labels, such as [S1] or [S1, S3]. Do not cite a passage that does not support the claim.
6. These labels identify retrieved source passages; they are not legal section or article numbers.
7. If evidence is missing or only partially answers the question, explain exactly what can be answered and what cannot be confirmed from the retrieved documents.

WRITING RULES FOR A BEGINNER:
1. Start with a short, direct answer of two or three sentences.
2. Add a section called "First, understand the basic idea" and explain the central legal term in everyday language.
3. Explain important points under clear numbered headings. For each point, include when supported: what it means in simple language; a short everyday hypothetical marked "Example"; and any legal limitation or exception in the passages.
4. Define unavoidable legal terms immediately in plain language. Prefer short sentences and common words. Do not assume the reader knows legal vocabulary.
5. For broad questions, finish with "What this means in practice" and give a concise practical understanding based only on the supplied evidence.
6. Use enough detail to teach the topic properly, but do not repeat information or add filler. Keep narrow questions focused; broad questions may need a longer structured answer.
7. End with one brief sentence stating that the response provides general legal information and is not a substitute for advice from a qualified legal professional.

FORMAT:
- Use clean Markdown with ## for major sections and ### only when a subsection is genuinely needed.
- Use **bold** for important legal terms and conclusions.
- Use short paragraphs and properly indented bullets or numbered lists.
- Do not begin with a heading that says "Legal assistant".
- Do not create tables unless a comparison is genuinely clearer.

If there are no retrieved passages, say that the available evidence does not support an answer and do not make legal claims."""

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class LegalQAPipeline:
    """Retrieve the top legal chunks and use them as Gemini answer context.

    Keep one pipeline instance for the lifetime of an app process. It reuses
    the database connection, local embedding model, and Gemini chat model.
    """

    def __init__(
        self,
        retriever: LegalRetriever,
        llm: Any,
        *,
        top_k: int = 5,
        progress_callback: Callable[[str], None] | None = None,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.progress_callback = progress_callback

    @classmethod
    def from_env(
        cls,
        *,
        top_k: int = 5,
        candidate_k: int = 40,
        progress_callback: Callable[[str], None] | None = None,
    ) -> "LegalQAPipeline":
        """Build the pipeline from project .env.local/.env and process env vars.

        Process environment variables take precedence. When both dotenv files
        define a key, .env.local takes precedence over .env.
        """
        # Explicit paths make startup independent of the shell's current
        # directory; load .env.local first so it wins over the shared .env.
        load_dotenv(dotenv_path=_PROJECT_ROOT / ".env.local")
        load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")
        connection = connect_from_env()
        try:
            with connection.cursor() as cursor:
                # Keep a slow or unindexed retrieval query from appearing hung.
                cursor.execute("SET statement_timeout = '60s'")
            embedder = BgeM3Embedder()
            retriever = LegalRetriever(
                connection, embedder, progress_callback=progress_callback
            )
            llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
                timeout=float(os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "120")),
                max_retries=1,
            )
            pipeline = cls(
                retriever,
                llm,
                top_k=top_k,
                progress_callback=progress_callback,
            )
            pipeline._connection = connection
            pipeline._candidate_k = candidate_k
            return pipeline
        except Exception:
            connection.close()
            raise

    @staticmethod
    def _format_context(chunks: list[dict[str, Any]]) -> str:
        source_blocks = []
        for number, chunk in enumerate(chunks, start=1):
            title = chunk.get("title") or str(chunk.get("document_id", "Unknown document"))
            location = (
                f"section {chunk.get('section_number') or 'unknown'}"
                f" {chunk.get('section_title') or ''}; "
                f"pages {chunk.get('page_number') or '?'}-{chunk.get('end_page') or '?'}"
            )
            citation = f"; citation {chunk['citation']}" if chunk.get("citation") else ""
            source_blocks.append(
                f"[S{number}: {title}, {location}{citation}]\n"
                f"{chunk.get('content', '')}"
            )
        return "\n\n---\n\n".join(source_blocks)

    @staticmethod
    def _response_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                item["text"] for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
        return str(content)

    def answer(self, query: str, *, response_language: str = "English") -> dict[str, Any]:
        """Answer one user query and return the answer plus its source chunks."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if response_language not in {"English", "Malayalam"}:
            raise ValueError("response_language must be English or Malayalam.")

        chunks = self.retriever.hybrid_search(
            query.strip(),
            top_k=self.top_k,
            candidate_k=getattr(self, "_candidate_k", 40),
        )
        if not chunks:
            self._report_progress("No matching chunks found; skipping Gemini generation.")
            return {"answer": "I could not find relevant chunks in the legal database.", "sources": []}

        messages = [
            ("system", _LEGAL_SYSTEM_PROMPT),
            (
                "human",
                f"## RESPONSE LANGUAGE\n\nRespond in {response_language}.\n\n"
                f"## USER QUESTION\n\n{query.strip()}\n\n"
                f"## RETRIEVED LEGAL PASSAGES\n\n{self._format_context(chunks)}",
            ),
        ]
        self._report_progress(
            f"Retrieved {len(chunks)} chunks. Waiting for Gemini to generate the answer..."
        )
        response = self.llm.invoke(messages)
        self._report_progress("Gemini response received.")
        return {
            "answer": self._response_text(response.content),
            "sources": chunks,
        }

    def _report_progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    def close(self) -> None:
        """Close the owned database connection, if built with from_env()."""
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None

    def __enter__(self) -> "LegalQAPipeline":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
