from contextlib import asynccontextmanager
import logging
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
_pipeline = None
_pipeline_init_lock = Lock()
_answer_lock = Lock()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    response_language: Literal["English", "Malayalam"] = "English"


def _get_pipeline():
    """Load the embedding model and Q&A service only when the first question arrives."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_init_lock:
            if _pipeline is None:
                from services.rag.retrieval.legal_qa import LegalQAPipeline

                _pipeline = LegalQAPipeline.from_env(top_k=5)
    return _pipeline


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    if _pipeline is not None:
        _pipeline.close()

app = FastAPI(
    title="CCpeeps Legal AI - RAG Service",
    description="Backend service for legal document ingestion and retrieval.",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/")
def root():
    return "Hello Amanul"

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "backend is running"
    }


@app.get("/api/health")
def rag_health_check():
    return {"status": "ok", "service": "legal-rag"}


@app.post("/api/ask")
def ask_legal_question(payload: AskRequest):
    """Retrieve legal passages and generate a grounded answer."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank.")

    try:
        pipeline = _get_pipeline()
        # The pipeline holds one model and one PostgreSQL connection. Serialize
        # access so concurrent requests cannot use that connection at once.
        with _answer_lock:
            result = pipeline.answer(
                question,
                response_language=payload.response_language,
            )
        return {
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except Exception as exc:
        logger.exception("Legal Q&A request failed")
        raise HTTPException(
            status_code=503,
            detail="The legal assistant could not process this question. Check that the RAG backend, database, and model credentials are available.",
        ) from exc
