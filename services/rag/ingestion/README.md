# RAG ingestion services

The reusable PDF extraction, cleaning, legal chunking, validation, embedding, and vector-store services are in this package. Run commands from the repository root.

Install pipeline dependencies with `python -m pip install -r services/rag/requirements-ingestion.txt`. For local BGE-M3 embeddings, also install `services/rag/requirements-bge-m3.txt`.

## Common commands

```powershell
# Process one local PDF and validate its chunks
python -m services.rag.ingestion.cli chunk --pdf path\to\act.pdf --output out\act_chunks.json

# Discover Drive PDFs, then chunk them resumably to ignored local output storage
python -m services.rag.ingestion.batch_runner

# Re-fetch PDF sources and audit the generated batch
python -m services.rag.ingestion.batch_validator

# Generate local BGE-M3 embeddings from a chunk artifact that passed validation
python -m services.rag.ingestion.cli embed --embedding-provider bge-m3 --chunks out\act_chunks.json --cache out\bge_cache.json --output out\act_embeddings.json
```

The batch output directory defaults to `services/rag/processed/legal_chunking/` and is Git-ignored. Set `LEGAL_CHUNK_OUTPUT_DIR` to choose another local directory. Drive credentials use `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_DRIVE_FOLDER_ID`. BGE-M3 downloads model weights on first use and emits 1024-dimensional vectors. Ensure the database vector column and query embedding model/dimensions match before storage or retrieval. No OCR is performed by these services.

## Selected corpus embedding and staged storage

`run_legal_corpus` uses the Phase C `quality_selection` outputs. It checks that accepted text has source IDs and page provenance, uses the local `BAAI/bge-m3` dense vectors (1024 dimensions, normalized), and resumes from a content-addressed cache. It stores per-source embedding artifacts and failure records under the ignored processed directory.

BGE-M3 writes each successful batch to an append-only cache journal and compacts that journal into the JSON cache after all documents finish embedding. If a run stops early, rerun the same stage; cached vectors are reused.

```powershell
# Re-extract and verify one source PDF, embed its accepted chunks, then insert and verify that source
python -m services.rag.ingestion.run_legal_corpus one

# After one_document_db_verification.json reports success, embed remaining chunks and ingest all sources
python -m services.rag.ingestion.run_legal_corpus all
```

The staged runner obtains document names and `source_url` values from Google Drive API metadata. It preserves Drive IDs in `legal_documents.source_drive_file_id`; chunks reference the database document UUID. It adds a unique constraint on `source_drive_file_id` only when absent and only after checking for duplicate IDs. The runner stops on embedding or integrity errors. No OCR is performed; low-text pages remain subject to the existing quality-selection quarantine.

Jurisdiction is assigned from the document title prefix: titles beginning with `Kerala` (case-insensitive) are `state`; all others are `central`. This is a title-based classification heuristic, not independent legal-authority verification. Document type remains separate.

Set `BGE_M3_MODEL_PATH` only if the local model is stored outside the default Hugging Face cache. The default local weight directory used by the staged runner is `%USERPROFILE%\.cache\bge-m3-local` when it contains `pytorch_model.bin`.

The completed 49-PDF audit reports are retained in `services/rag/reports/legal_chunking/`. New batch reports, chunk JSON, checkpoints, and detailed machine output are written to the ignored local output directory and should not be committed.
