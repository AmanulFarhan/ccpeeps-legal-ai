
from pathlib import Path

from services.rag.ingestion.pdf_extractor import extract_pdf_text
from services.rag.ingestion.text_cleaner import clean_extracted_text


def process_pdf(pdf_path: str) -> list[dict]:
    """
    Extract and clean a PDF page by page.

    Keeps original text, cleaned text, and page numbers.
    """
    extracted_pages = extract_pdf_text(pdf_path)
    processed_pages = []

    for page in extracted_pages:
        original_text = page["text"]
        cleaned_text = clean_extracted_text(original_text)

        processed_pages.append(
            {
                "page_number": page["page_number"],
                "original_text": original_text,
                "cleaned_text": cleaned_text,
            }
        )

    return processed_pages
