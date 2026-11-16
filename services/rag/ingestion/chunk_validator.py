"""Validate legal chunk structure and provenance without changing any source text."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any


REQUIRED_FIELDS = {
    "document_id", "part", "chapter_number", "chapter_title", "section_number",
    "section_title", "subsection_number", "clause_number", "text", "start_page",
    "end_page", "chunk_index", "chunking_method", "word_count", "classification_status",
}
SECTION_LINE = re.compile(r"(?m)^\s*(\d{1,4})\s*[.．)]\s+([A-Z][^\n]{2,140})")


def _page_intervals(pages: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    intervals, offset = [], 0
    for i, page in enumerate(pages):
        if i:
            offset += 1  # matches the page-stream joining newline in legal_chunker
        text = str(page.get("original_text", page.get("text", page.get("cleaned_text", ""))) or "")
        start, end = offset, offset + len(text)
        intervals.append((int(page.get("page_number", page.get("page", i + 1))), start, end))
        offset = end
    return intervals


def validate_chunks(
    chunks: list[dict[str, Any]],
    *,
    pages: list[dict[str, Any]] | None = None,
    expected_text: str | None = None,
    short_chunk_words: int = 12,
) -> dict[str, Any]:
    """Return machine-readable errors and warnings; never mutate or discard chunks."""
    errors: list[dict] = []
    warnings: list[dict] = []

    def add(target: list[dict], index, code: str, message: str, **extra):
        target.append({"chunk_index": index, "code": code, "message": message, **extra})

    indices = [c.get("chunk_index") for c in chunks]
    duplicates = [i for i, n in Counter(indices).items() if n is not None and n > 1]
    for duplicate in duplicates:
        add(errors, duplicate, "duplicate_chunk_index", "Chunk index occurs more than once.")
    if indices != list(range(len(chunks))):
        add(errors, None, "chunk_index_order", "Chunk indices must be unique and ordered from zero.", actual_indices=indices[:30])

    available_pages = {p for p, _, _ in _page_intervals(pages)} if pages is not None else None
    for position, chunk in enumerate(chunks):
        index = chunk.get("chunk_index", position)
        missing = sorted(REQUIRED_FIELDS - chunk.keys())
        if missing:
            add(errors, index, "missing_fields", "Required chunk fields are missing.", fields=missing)
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            add(errors, index, "empty_text", "Chunk text must be a non-empty string.")
            text = text if isinstance(text, str) else ""
        if isinstance(text, str) and chunk.get("word_count") != len(text.split()):
            add(errors, index, "word_count_mismatch", "word_count does not match chunk text.")
        if chunk.get("classification_status") not in {"classified", "unclassified"}:
            add(errors, index, "invalid_classification", "Unknown classification_status.")

        start, end = chunk.get("start_page"), chunk.get("end_page")
        if start is None or end is None or not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
            add(errors, index, "invalid_page_range", "Page range is missing or invalid.", start_page=start, end_page=end)
        elif available_pages is not None and (start not in available_pages or end not in available_pages):
            add(errors, index, "page_out_of_range", "Page range is not present in supplied source pages.", start_page=start, end_page=end)

        if chunk.get("section_number") is not None and not re.fullmatch(r"\d{1,4}", str(chunk["section_number"])):
            add(warnings, index, "unusual_section_number", "Section number is not a plain numeric marker.")
        if chunk.get("subsection_number") is not None and not re.fullmatch(r"\d{1,4}", str(chunk["subsection_number"])):
            add(warnings, index, "unusual_subsection_number", "Subsection marker is not numeric; inspect source/OCR before interpreting it.")
        if chunk.get("clause_number") is not None and not re.fullmatch(r"[A-Za-z0-9ivxIVX]{1,8}", str(chunk["clause_number"])):
            add(warnings, index, "unusual_clause_number", "Clause marker has an unusual format.")
        if chunk.get("clause_number") is not None and str(chunk["clause_number"]).isdigit():
            add(warnings, index, "clause_looks_like_subsection", "Numeric clause metadata may be a subsection marker.")
        if chunk.get("subsection_number") is not None and re.fullmatch(r"[A-Za-z]", str(chunk["subsection_number"])):
            add(warnings, index, "subsection_looks_like_clause", "Alphabetic subsection metadata may be a clause marker.")
        if chunk.get("subsection_number") is not None:
            marker = re.sub(r"\s+", "", str(text))
            expected_marker = f"({chunk['subsection_number']})"
            if expected_marker not in marker and not re.match(r"^\([A-Za-z]\)", marker):
                add(warnings, index, "subsection_marker_absent", "Subsection metadata marker is absent from this chunk text.", expected_marker=expected_marker)
        if chunk.get("clause_number") is not None:
            marker = re.sub(r"\s+", "", str(text)).casefold()
            expected_marker = f"({chunk['clause_number']})".casefold()
            if expected_marker not in marker:
                add(warnings, index, "clause_marker_absent", "Clause metadata marker is absent from this chunk text.", expected_marker=expected_marker)
        if chunk.get("chapter_number") is not None and not re.fullmatch(r"(?:[IVXLCDM]+|\d+)", str(chunk["chapter_number"]), re.I):
            add(warnings, index, "unusual_chapter_number", "Chapter number has an unusual format.")
        if chunk.get("classification_status") == "classified" and chunk.get("section_number") is None:
            add(warnings, index, "classified_without_section", "Classified chunk has no section/rule number.")
        if chunk.get("classification_status") == "classified" and re.search(r"(?im)^\s*(?:arrangement of sections|contents)\s*$", text):
            add(warnings, index, "toc_text_classified", "Classified chunk contains a TOC/front-matter marker.")
        if chunk.get("classification_status") == "classified" and len(text.split()) < short_chunk_words:
            add(warnings, index, "short_classified_chunk", "Classified chunk is unusually short.", words=len(text.split()))

        section_lines = [m for m in SECTION_LINE.finditer(text) if not 1800 <= int(m.group(1)) <= 2100]
        if chunk.get("classification_status") == "classified" and len(section_lines) > 1:
            add(warnings, index, "multiple_section_heading_candidates", "Chunk contains multiple heading-like numbered lines.", candidates=[m.group(1) for m in section_lines])
        if chunk.get("classification_status") == "classified" and section_lines and chunk.get("section_number"):
            first = section_lines[0].group(1)
            if first != str(chunk["section_number"]):
                add(warnings, index, "section_metadata_text_mismatch", "First heading-like line disagrees with section metadata.", text_marker=first)

    duplicate_text = [t for t, n in Counter(c.get("text") for c in chunks if isinstance(c.get("text"), str)).items() if n > 1]
    for text in duplicate_text:
        add(warnings, None, "duplicate_text", "Identical chunk text occurs more than once.", text_preview=text[:160])

    concatenated = "".join(c.get("text", "") if isinstance(c.get("text"), str) else "" for c in chunks)
    if expected_text is not None:
        if concatenated != expected_text:
            add(errors, None, "text_reconstruction_mismatch", "Concatenated chunks differ from expected source text.", source_chars=len(expected_text), chunk_chars=len(concatenated))
    elif pages is not None:
        source = "\n".join(str(p.get("original_text", p.get("text", p.get("cleaned_text", ""))) or "") for p in pages)
        if concatenated != source:
            add(errors, None, "text_reconstruction_mismatch", "Concatenated chunks differ from supplied source pages.", source_chars=len(source), chunk_chars=len(concatenated))

    provenance_checked = False
    if pages is not None and concatenated == "".join(c.get("text", "") for c in chunks):
        intervals = _page_intervals(pages)
        offset = 0
        for chunk in chunks:
            text = chunk.get("text", "")
            end_offset = offset + len(text)
            touched = [page for page, left, right in intervals if left < end_offset and right > offset]
            if touched and (chunk.get("start_page"), chunk.get("end_page")) != (min(touched), max(touched)):
                add(errors, chunk.get("chunk_index"), "page_provenance_mismatch", "Page metadata does not match the characters in this chunk.", expected_start=min(touched), expected_end=max(touched), actual_start=chunk.get("start_page"), actual_end=chunk.get("end_page"))
            offset = end_offset
        provenance_checked = True

    return {
        "valid": not errors,
        "chunk_count": len(chunks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "exact_reconstruction": (concatenated == expected_text) if expected_text is not None else (
            concatenated == "\n".join(str(p.get("original_text", p.get("text", p.get("cleaned_text", ""))) or "") for p in pages)
            if pages is not None else None
        ),
        "page_provenance_checked": provenance_checked,
        "errors": errors,
        "warnings": warnings,
    }
