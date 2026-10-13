import re


SECTION_PATTERN = re.compile(
    r"(?m)^[ \t]*(\d{1,3})\.\s+([^\n]+)"
)

FIRST_SECTION_PATTERN = re.compile(
    r"(?m)^[ \t]*1\.\s+"
    r"Short title,\s*commencement and application\."
    r"\s*(?:—|–|––|-\s*)"
    r"\s*\(1\)\s*This Act may be called"
)


def chunk_document(pages: list[dict]) -> list[dict]:
    """Group the main body of the BNS PDF into numbered sections."""
    full_text = ""
    page_ranges = []

    for page in pages:
        start = len(full_text)
        full_text += page["cleaned_text"] + "\n"
        end = len(full_text)
        page_ranges.append({
            "page_number": page["page_number"],
            "start": start,
            "end": end,
        })

    first_section = FIRST_SECTION_PATTERN.search(full_text)
    if first_section is None:
        raise ValueError(
            "Could not find the beginning of Section 1. "
            "Check the extracted PDF text."
        )

    body_start = first_section.start()
    candidates = [
        match
        for match in SECTION_PATTERN.finditer(full_text, body_start)
    ]

    valid_sections = []
    expected_number = 1

    for match in candidates:
        number = int(match.group(1))
        if number == expected_number:
            valid_sections.append(match)
            expected_number += 1

    chunks = []

    for index, match in enumerate(valid_sections):
        section_start = match.start()
        if index + 1 < len(valid_sections):
            section_end = valid_sections[index + 1].start()
        else:
            section_end = len(full_text)

        section_text = full_text[section_start:section_end].strip()

        source_pages = [
            page["page_number"]
            for page in page_ranges
            if page["start"] < section_end and page["end"] > section_start
        ]

        raw_heading = match.group(2).strip()
        heading = re.split(r"—|–", raw_heading, maxsplit=1)[0].strip()

        chunks.append({
            "section_number": match.group(1),
            "heading": heading,
            "text": section_text,
            "start_page": min(source_pages) if source_pages else None,
            "end_page": max(source_pages) if source_pages else None,
        })

    return chunks
