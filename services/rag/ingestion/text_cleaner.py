
import re


def clean_extracted_text(text: str) -> str:
    """
    Normalize whitespace and remove known IndiaCode PDF artifacts.

    Preserve legal wording, punctuation, section numbers,
    subsections, clauses, and illustrations.
    """

    # 1. Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Remove the known publisher footer.
    # Match only a standalone line containing "IndiaCode".
    text = re.sub(
        r"(?im)^[ \t]*IndiaCode[ \t]*$",
        "",
        text,
    )

    # 3. Remove standalone printed page numbers.
    # This is appropriate for the current IndiaCode PDF format.
    # Revisit this rule for documents where standalone numbers
    # may carry legal meaning.
    text = re.sub(
        r"(?m)^[ \t]*\d{1,3}[ \t]*$",
        "",
        text,
    )

    # 4. Remove trailing whitespace from individual lines.
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 5. Normalize repeated spaces and excessive blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
