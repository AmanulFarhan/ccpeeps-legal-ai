
from pathlib import Path

from services.rag.ingestion.document_processor import process_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = PROJECT_ROOT / "data" / "source_documents" / "bns_2023.pdf"


def main():
    pages = process_pdf(str(PDF_PATH))

    print(f"Pages processed: {len(pages)}")

    first_page = pages[0]

    print("\nPage number:", first_page["page_number"])
    print("\nOriginal text:")
    print(first_page["original_text"][:300])

    print("\nCleaned text:")
    print(first_page["cleaned_text"][:300])

    assert len(pages) == 112, "Unexpected page count"
    assert first_page["page_number"] == 1
    assert first_page["original_text"]
    assert first_page["cleaned_text"]

    print("\nAll basic checks passed!")

    for page in pages[20:22]:
        print(f"\n--- Page {page['page_number']} ---")
        print(page["cleaned_text"])


if __name__ == "__main__":
    main()
