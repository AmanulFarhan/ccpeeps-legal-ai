
from pathlib import Path

from ..ingestion.pdf_extractor import extract_pdf_text


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = PROJECT_ROOT / "data" / "source_documents" / "bns_2023.pdf"


def main():
    pages = extract_pdf_text(str(PDF_PATH))

    print(f"Total pages extracted: {len(pages)}")

    for page in pages[:3]:
        print(f"\n--- Page {page['page_number']} ---")
        print(page["text"][:500] or "[No text extracted]")


if __name__ == "__main__":
    main()
