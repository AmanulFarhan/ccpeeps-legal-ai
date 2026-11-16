"""Deterministic, lossless structure-aware chunking for extracted legal PDF text."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class ChunkingConfig:
    max_words: int = 1200
    min_body_signals: int = 2  # retained for caller compatibility


@dataclass(frozen=True)
class _Page:
    number: int
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _Line:
    text: str
    start: int
    end: int
    content_end: int
    page: int


_SECTION = re.compile(r"^\s*(\d{1,3}|[IlOo])\s*[.\uFF0E)]\s*(.*?)\s*$")
_PART = re.compile(r"^\s*PART\s+([IVXLCDM]+|\d+)\s*[.:\u2014\u2013-]?\s*(.*?)\s*$", re.I)
_CHAPTER = re.compile(r"^\s*CHAPTER\s+([IVXLCDM]+|\d+)\s*[.:\u2014\u2013-]?\s*(.*?)\s*$", re.I)
_SUBSECTION = re.compile(r"^\s*\(\s*(\d+)\s*\)")
_CLAUSE = re.compile(r"^\s*\(\s*([a-z])\s*\)", re.I)
_CONTENTS_MARKER = re.compile(r"^\s*(?:CONTENTS|ARRANGEMENT\s+OF\s+SECTIONS)\s*$", re.I)
_ACT_PREAMBLE = re.compile(r"\b(?:AN\s+ACT\s+TO|WHEREAS\b|PREAMBLE\b)", re.I)
_SCHEDULE = re.compile(r"^\s*(?:THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|\d+(?:ST|ND|RD|TH))\s+SCHEDULE\s*$", re.I)


def _heading_like_title(title: str) -> bool:
    """Reject prose references that happen to start with PART/CHAPTER tokens."""
    title = title.strip()
    if not title:
        return True  # title may be on the following line
    words = re.findall(r"[A-Za-z]+", title)
    if not words or len(title) > 180:
        return False
    upper = sum(word.isupper() for word in words)
    return upper / len(words) >= 0.75


def _source_pages(pages: list[dict[str, Any]]) -> tuple[str, list[_Page]]:
    """Build the exact extracted-text stream and remember its page intervals."""
    pieces: list[str] = []
    mapped: list[_Page] = []
    offset = 0
    for index, item in enumerate(pages):
        if index:
            pieces.append("\n")
            offset += 1
        text = str(item.get("original_text", item.get("text", item.get("cleaned_text", ""))) or "")
        start = offset
        pieces.append(text)
        offset += len(text)
        mapped.append(_Page(int(item.get("page_number", item.get("page", index + 1))), text, start, offset))
    return "".join(pieces), mapped


def _lines(source: str, pages: list[_Page]) -> list[_Line]:
    result: list[_Line] = []
    page_index = 0
    for match in re.finditer(r"[^\n]*(?:\n|$)", source):
        raw = match.group(0)
        if not raw:
            continue
        content = raw[:-1] if raw.endswith("\n") else raw
        while page_index + 1 < len(pages) and match.start() >= pages[page_index].end + 1:
            page_index += 1
        page = pages[page_index].number if pages else 1
        result.append(_Line(content, match.start(), match.end(), match.start() + len(content), page))
    return result


def _is_toc_entry(text: str) -> bool:
    return bool(re.search(r"(?:\.{2,}|_{2,})", text) or re.search(r"\s{2,}\d{1,4}\s*$", text))


def _next_nonempty(lines: list[_Line], index: int) -> int | None:
    candidate = index + 1
    while candidate < len(lines) and not lines[candidate].text.strip():
        candidate += 1
    return candidate if candidate < len(lines) else None


def _section_heading(lines: list[_Line], index: int) -> tuple[str, str, int] | None:
    """Return section number/title and number of consumed lines, tolerating split/OCR headings."""
    match = _SECTION.match(lines[index].text)
    if not match or _is_toc_entry(lines[index].text):
        return None
    number, title = match.group(1), match.group(2).strip()
    consumed = 1
    next_line = _next_nonempty(lines, index)
    if not title and next_line is not None:
        title = lines[next_line].text.strip()
        consumed = next_line - index + 1
    elif len(title) < 4 and next_line is not None and not _SECTION.match(lines[next_line].text):
        title = (title + " " + lines[next_line].text.strip()).strip()
        consumed = next_line - index + 1
    if not re.search(r"[A-Za-z]{2,}", title):
        return None
    # Ambiguous OCR glyphs are retained as text and not silently converted to digits.
    if not number.isdigit():
        return None
    return number, title, consumed


def _body_start(lines: list[_Line]) -> int:
    """Find statutory text using prose and provision evidence after title/contents signals."""
    if not lines:
        return 0
    toc_markers = [i for i, line in enumerate(lines) if _CONTENTS_MARKER.search(line.text)]
    has_preamble = any(_ACT_PREAMBLE.search(line.text) for line in lines)
    for i, line in enumerate(lines):
        if toc_markers and i <= toc_markers[-1]:
            continue
        heading = _section_heading(lines, i)
        if not heading or heading[0] != "1":
            continue
        if _is_toc_entry(line.text):
            continue
        following = " ".join(x.text for x in lines[i + heading[2]:i + heading[2] + 8])
        legal_signal = bool(re.search(r"\(\s*(?:1|[Il])\s*\)|\b(?:shall|means|may be called|is hereby|hereinafter)\b", following, re.I))
        substantial = len(following.split()) >= 6
        chapter_or_part_context = any(
            _PART.match(x.text) or _CHAPTER.match(x.text)
            for x in lines[max(0, i - 8):i]
        )
        if legal_signal and substantial and (has_preamble or chapter_or_part_context or not toc_markers):
            prior_structures = [j for j in range(max(toc_markers[-1] + 1 if toc_markers else 0, i - 24), i)
                                if _PART.match(lines[j].text) or _CHAPTER.match(lines[j].text)]
            return prior_structures[-1] if prior_structures else i
    # Kerala-style Acts may have OCR-corrupted first numbered sections; a statutory Part
    # after the Act preamble is a safer body boundary than an OCR guess at section "1".
    for i, line in enumerate(lines):
        if not _PART.match(line.text):
            continue
        prior = " ".join(x.text for x in lines[max(0, i - 30):i])
        if _ACT_PREAMBLE.search(prior) and (not toc_markers or i > toc_markers[-1]):
            return i
    # If no statutory-body evidence is reliable, preserve everything as fallback.
    return len(lines)


def _word_count(text: str) -> int:
    return len(text.split())


def _greedy_ranges(text: str, start: int, end: int, boundaries: list[int], max_words: int) -> list[tuple[int, int]]:
    cuts = [start, *sorted({x for x in boundaries if start < x < end}), end]
    groups: list[tuple[int, int]] = []
    group_start = start
    for i, boundary in enumerate(cuts[1:-1], start=1):
        next_end = cuts[i + 1]
        if _word_count(text[group_start:next_end]) > max_words and boundary > group_start:
            groups.append((group_start, boundary))
            group_start = boundary
    groups.append((group_start, end))
    return groups


def _page_span(start: int, end: int, pages: list[_Page]) -> tuple[int | None, int | None]:
    touched = [p.number for p in pages if p.start < end and p.end > start]
    if not touched:
        nearest = next((p.number for p in pages if p.start <= start <= p.end), None)
        return nearest, nearest
    return min(touched), max(touched)


def _legal_events(lines: list[_Line], body_line: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current_part = current_chapter = None
    part_count = chapter_count = section_count = subsection_count = clause_count = 0
    i = body_line
    while i < len(lines):
        line = lines[i]
        if _SCHEDULE.match(line.text):
            events.append({"kind": "schedule", "start": line.start, "end": line.end,
                           "part": current_part, "chapter_number": None, "chapter_title": None})
            break
        part = _PART.match(line.text)
        if part and not _heading_like_title(part.group(2)):
            part = None
        chapter = _CHAPTER.match(line.text)
        if chapter and not _heading_like_title(chapter.group(2)):
            chapter = None
        section = _section_heading(lines, i)
        if part:
            current_part = f"{part.group(1)} {part.group(2)}".strip()
            current_chapter = None
            part_count += 1
            title = part.group(2).strip()
            consumed = 1
            next_line = _next_nonempty(lines, i)
            if not title and next_line is not None:
                title = lines[next_line].text.strip()
                consumed = next_line - i + 1
                current_part = f"{part.group(1)} {title}".strip()
            events.append({"kind": "part", "start": line.start, "end": lines[min(i + consumed - 1, len(lines) - 1)].end,
                           "part": current_part, "chapter_number": None, "chapter_title": None})
            i += consumed
            continue
        if chapter:
            title = chapter.group(2).strip()
            consumed = 1
            next_line = _next_nonempty(lines, i)
            if not title and next_line is not None:
                title = lines[next_line].text.strip()
                consumed = next_line - i + 1
            current_chapter = (chapter.group(1), title or None)
            chapter_count += 1
            events.append({"kind": "chapter", "start": line.start, "end": lines[min(i + consumed - 1, len(lines) - 1)].end,
                           "part": current_part, "chapter_number": current_chapter[0], "chapter_title": current_chapter[1]})
            i += consumed
            continue
        if section:
            number, title, consumed = section
            section_count += 1
            events.append({"kind": "section", "start": line.start, "end": lines[min(i + consumed - 1, len(lines) - 1)].end,
                           "part": current_part, "chapter_number": current_chapter[0] if current_chapter else None,
                           "chapter_title": current_chapter[1] if current_chapter else None,
                           "section_number": number, "section_title": title})
        sub = _SUBSECTION.match(line.text)
        clause = _CLAUSE.match(line.text)
        if sub:
            subsection_count += 1
            events.append({"kind": "subsection", "start": line.start, "end": line.end, "subsection_number": sub.group(1)})
            for match in re.finditer(r"\(\s*([a-z])\s*\)", line.text[sub.end():], re.I):
                clause_count += 1
                position = line.start + sub.end() + match.start()
                events.append({"kind": "clause", "start": position, "end": position + len(match.group(0)), "clause_number": match.group(1)})
        elif clause:
            clause_count += 1
            events.append({"kind": "clause", "start": line.start, "end": line.end, "clause_number": clause.group(1)})
        elif _SECTION.match(line.text):
            # PDF extraction often places the first subsection on the section-title line.
            for match in re.finditer(r"(?:[\u2014\u2013-])\s*\(\s*(\d+)\s*\)", line.text):
                subsection_count += 1
                events.append({"kind": "subsection", "start": line.start + match.start(), "end": line.start + match.end(), "subsection_number": match.group(1)})
            for match in re.finditer(r"\(\s*([a-z])\s*\)", line.text, re.I):
                clause_count += 1
                events.append({"kind": "clause", "start": line.start + match.start(), "end": line.start + match.end(), "clause_number": match.group(1)})
        i += 1
    return events, {"parts": part_count, "chapters": chapter_count, "sections": section_count,
                    "subsections": subsection_count, "clauses": clause_count}


def _split_boundaries(text: str, start: int, end: int, events: list[dict[str, Any]], max_words: int) -> list[tuple[int, int, str]]:
    if _word_count(text[start:end]) <= max_words:
        return [(start, end, "section")]
    marks = sorted({e["start"] for e in events if e["kind"] in {"subsection", "clause"} and start < e["start"] < end})
    candidates = [start, *marks, end]
    groups: list[tuple[int, int]] = []
    active_start = candidates[0]
    for index, boundary in enumerate(candidates[1:-1], start=1):
        next_boundary = candidates[index + 1]
        if _word_count(text[active_start:next_boundary]) > max_words and boundary > active_start:
            groups.append((active_start, boundary))
            active_start = boundary
    groups.append((active_start, end))
    output: list[tuple[int, int, str]] = []
    for a, b in groups:
        if _word_count(text[a:b]) <= max_words:
            output.append((a, b, "subsection_clause_boundary"))
            continue
        # Fall back to paragraph then sentence boundaries, maintaining exact offsets.
        sentence_bounds = [a + m.end() for m in re.finditer(r"(?:\n\s*\n|(?<=[.!?])\s+)", text[a:b])]
        sentence_groups = _greedy_ranges(text, a, b, sentence_bounds, max_words)
        for sentence_start, sentence_end in sentence_groups:
            if _word_count(text[sentence_start:sentence_end]) <= max_words:
                output.append((sentence_start, sentence_end, "paragraph_sentence_boundary"))
                continue
            whitespace_bounds = [sentence_start + m.end() for m in re.finditer(r"\s+", text[sentence_start:sentence_end])]
            for word_start, word_end in _greedy_ranges(text, sentence_start, sentence_end, whitespace_bounds, max_words):
                output.append((word_start, word_end, "paragraph_sentence_boundary"))
    return output


def analyze_document(pages: list[dict[str, Any]], config: ChunkingConfig | None = None) -> dict[str, Any]:
    config = config or ChunkingConfig()
    source, mapped_pages = _source_pages(pages)
    lines = _lines(source, mapped_pages)
    body_line = _body_start(lines)
    events, counts = _legal_events(lines, body_line)
    body_page = lines[body_line].page if body_line < len(lines) else None
    ocr_pages = [p.number for p in mapped_pages if len(re.sub(r"\s+", "", p.text)) < 80]
    return {"body_start_page": body_page, "body_start_line": body_line, **counts,
            "line_count": len(lines), "source_characters": len(source),
            "ocr_review_pages": ocr_pages, "max_words": config.max_words}


def chunk_document(pages: list[dict[str, Any]], document_id: str, config: ChunkingConfig | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Partition the exact extracted text into structured chunks plus lossless fallback chunks."""
    config = config or ChunkingConfig()
    source, mapped_pages = _source_pages(pages)
    lines = _lines(source, mapped_pages)
    body_line = _body_start(lines)
    body_start = lines[body_line].start if body_line < len(lines) else len(source)
    events, counts = _legal_events(lines, body_line)
    ranges: list[dict[str, Any]] = []
    def append_range(a: int, b: int, method: str, classified: bool, metadata: dict[str, Any] | None = None) -> None:
        if b <= a:
            return
        ranges.append({"start": a, "end": b, "chunking_method": method,
                       "classification_status": "classified" if classified else "unclassified",
                       **(metadata or {})})
    def emit_interval(a: int, b: int, section: dict[str, Any] | None) -> None:
        if b <= a:
            return
        if section is None:
            prefix = "front_matter" if a < body_start else "fallback"
            for left, right, method in _split_boundaries(source, a, b, [], config.max_words):
                append_range(left, right, prefix if method == "section" else f"{prefix}_{method}", False)
            return
        section_events = [e for e in events if e["kind"] in {"subsection", "clause"} and a <= e["start"] < b]
        meta = {k: section.get(k) for k in ("part", "chapter_number", "chapter_title", "section_number", "section_title")}
        pieces = _split_boundaries(source, a, b, events, config.max_words)
        for left, right, method in pieces:
            markers = [e for e in section_events if left <= e["start"] < right]
            prior_sub = prior_clause = None
            for event in section_events:
                if event["start"] >= left:
                    break
                if event["kind"] == "subsection":
                    prior_sub, prior_clause = event["subsection_number"], None
                elif event["kind"] == "clause":
                    prior_clause = event["clause_number"]
            current_subs = {e["subsection_number"] for e in markers if e["kind"] == "subsection"}
            current_clauses = {e["clause_number"] for e in markers if e["kind"] == "clause"}
            child_sub = next(iter(current_subs)) if len(current_subs) == 1 else (None if current_subs else prior_sub)
            child_clause = next(iter(current_clauses)) if len(current_clauses) == 1 else (None if current_clauses or current_subs else prior_clause)
            append_range(left, right, method, True, {**meta, "subsection_number": child_sub, "clause_number": child_clause})

    # TOC/front matter is unclassified. Structural boundaries thereafter are processed
    # in source order so Part/Chapter headings close prior sections and reset hierarchy.
    if body_start:
        emit_interval(0, body_start, None)
    active_section: dict[str, Any] | None = None
    cursor = body_start
    boundaries = [e for e in events if e["kind"] in {"part", "chapter", "section", "schedule"} and e["start"] >= body_start]
    for event in boundaries:
        if event["start"] < cursor:
            continue
        emit_interval(cursor, event["start"], active_section)
        if event["kind"] == "section":
            active_section = event
            cursor = event["start"]
        else:
            active_section = None
            emit_interval(event["start"], event["end"], None)
            cursor = event["end"]
    emit_interval(cursor, len(source), active_section)

    # Sort and merge only adjacent fallback intervals; no generated character is rewritten.
    ranges.sort(key=lambda r: (r["start"], r["end"]))
    chunks: list[dict[str, Any]] = []
    for record in ranges:
        text = source[record["start"]:record["end"]]
        start_page, end_page = _page_span(record["start"], record["end"], mapped_pages)
        chunk = {"document_id": document_id, "part": record.get("part"),
                 "chapter_number": record.get("chapter_number"), "chapter_title": record.get("chapter_title"),
                 "section_number": record.get("section_number"), "section_title": record.get("section_title"),
                 "subsection_number": record.get("subsection_number"), "clause_number": record.get("clause_number"),
                 "text": text, "start_page": start_page, "end_page": end_page,
                 "chunk_index": len(chunks), "chunking_method": record["chunking_method"],
                 "word_count": _word_count(text), "classification_status": record["classification_status"]}
        chunks.append(chunk)

    reconstruction = "".join(c["text"] for c in chunks)
    analysis = analyze_document(pages, config)
    analysis.update({**counts, "exact_reconstruction": reconstruction == source,
                     "source_characters": len(source), "reconstructed_characters": len(reconstruction),
                     "chunk_count": len(chunks)})
    return chunks, analysis
