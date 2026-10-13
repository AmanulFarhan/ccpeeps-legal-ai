from fastapi import FastAPI

app = FastAPI(
    title="CCpeeps Legal AI - RAG Service",
    description="Backend service for legal document ingestion and retrieval.",
    version="0.1.0",
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