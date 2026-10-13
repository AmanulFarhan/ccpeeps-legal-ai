from services.rag.ingestion.pdf_extractor import extract_pdf_text
from services.rag.ingestion.text_cleaner import clean_extracted_text

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = PROJECT_ROOT / "data" / "source_documents" / "bns_2023.pdf"


def main():
    pages = extract_pdf_text(str(PDF_PATH))

    # Keep the original text unchanged.
    original_text = pages[0]["text"]

    # Create a separate cleaned version.
    cleaned_text = clean_extracted_text(original_text)

    print("ORIGINAL TEXT")
    print(repr(original_text[:500]))

    print("\nCLEANED TEXT")
    print(repr(cleaned_text[:500]))

    print("\nOriginal length:", len(original_text))
    print("Cleaned length:", len(cleaned_text))


if __name__ == "__main__":
    main()
