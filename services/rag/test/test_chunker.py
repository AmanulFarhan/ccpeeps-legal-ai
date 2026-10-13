
from pathlib import Path

from services.rag.ingestion.document_processor import process_pdf
from services.rag.ingestion.chunker import chunk_document


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = PROJECT_ROOT / "data" / "source_documents" / "bns_2023.pdf"


def main():
    # Extract and clean the PDF.
    pages = process_pdf(str(PDF_PATH))

    # Divide the document into candidate section chunks.
    chunks = chunk_document(pages)

    print(f"Total PDF pages: {len(pages)}")
    print(f"Total candidate chunks: {len(chunks)}")

    print("\nFirst 15 detected chunks:")

    for chunk in chunks[:15]:
        print(
            f"\nSection {chunk['section_number']}: "
            f"{chunk['heading']}"
        )
        print(
            f"Pages: {chunk['start_page']}-"
            f"{chunk['end_page']}"
        )
        print(f"Characters: {len(chunk['text'])}")
        print(f"Preview: {chunk['text'][:180]!r}")

    # Basic checks.
    assert len(pages) == 112, "Unexpected PDF page count"
    assert chunks, "No chunks were detected"

    print("\nBasic checks passed.")


if __name__ == "__main__":
    main()
