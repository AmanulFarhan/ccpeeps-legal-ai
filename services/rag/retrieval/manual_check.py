"""Interactive terminal smoke check for the legal Q&A pipeline.

Run from the repository root:
    python -m services.rag.retrieval.manual_check
"""

from services.rag.retrieval.legal_qa import LegalQAPipeline


def main() -> None:
    print("Legal Q&A manual check. Enter a question, or type 'exit' to quit.\n")
    with LegalQAPipeline.from_env(
        top_k=10,
        progress_callback=lambda message: print(f"[RAG] {message}", flush=True),
    ) as pipeline:
        while True:
            query = input("Question: ").strip()
            if query.casefold() in {"exit", "quit"}:
                break
            if not query:
                continue

            result = pipeline.answer(query)
            print("\nAnswer:\n")
            print(result["answer"])
            print("\nRetrieved source chunks:")
            for index, source in enumerate(result["sources"], start=1):
                title = source.get("title") or source.get("document_id", "Unknown document")
                print(
                    f"\n[S{index}] {title} | "
                    f"section {source.get('section_number') or '?'} | "
                    f"pages {source.get('page_number') or '?'}-"
                    f"{source.get('end_page') or '?'}"
                )
                if source.get("citation"):
                    print(f"Citation metadata: {source['citation']}")
                print(source.get("content", ""))
            print("\nCheck each answer claim against these passages and their [S#] labels.\n")


if __name__ == "__main__":
    main()
