from pathlib import Path
import pymupdf

def extract_pdf_text(pdf_path: str) -> list[dict]:

    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            pages.append({
                "page_number": page_number,
                "text": text
            })

    return pages