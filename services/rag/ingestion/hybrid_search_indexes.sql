-- Run once against Supabase/Postgres. CREATE INDEX CONCURRENTLY must run
-- outside a transaction. Both indexes match the operators/expressions used by
-- retrieval.py. If an index with the same name already exists, inspect its
-- definition first; IF NOT EXISTS does not replace an existing index.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_legal_chunks_embedding
ON public.legal_chunks
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS legal_chunks_content_fts_simple_idx
ON public.legal_chunks
USING GIN (to_tsvector('simple'::regconfig, content));
