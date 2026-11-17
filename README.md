# CCpeeps Legal AI Assistant

This repository contains the Next.js frontend and the Python legal RAG service.

## Run locally

Run both services from the repository root, in separate terminals.

Install the Python dependencies into the active virtual environment:

```powershell
python -m pip install -r services/rag/requirements-rag.txt
```

Start the Python API:

```powershell
python -m uvicorn services.main:app --reload --port 8000
```

Start the Next.js frontend:

```powershell
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The chat sends questions to the Next.js `/api/ask` route, which forwards them to the Python RAG API at `http://127.0.0.1:8000`. Set `RAG_API_URL` in the frontend environment when the Python API uses a different URL.

The RAG service needs `SUPABASE_DB_URL` (or `DATABASE_URL`), a Gemini API key, and the BGE-M3 model/cache configured as described in [the ingestion guide](services/rag/ingestion/README.md). The Python service exposes `GET /health` and `POST /api/ask`. The ask endpoint accepts a question and response language (`English` or `Malayalam`), and returns the generated answer with retrieved source metadata.

The main frontend page is [src/app/page.tsx](src/app/page.tsx). The API forwarding route is [src/app/api/ask/route.ts](src/app/api/ask/route.ts). The Python endpoint is [services/main.py](services/main.py).

## Learn Next.js

- [Next.js Documentation](https://nextjs.org/docs)
- [Learn Next.js](https://nextjs.org/learn)
- [Next.js on GitHub](https://github.com/vercel/next.js)
