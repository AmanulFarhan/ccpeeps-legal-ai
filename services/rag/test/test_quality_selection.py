from services.rag.ingestion.quality_selection import decide_chunk


def test_quality_selection_rejects_only_whitespace_and_keeps_source_text_elsewhere():
    result = decide_chunk({"text": " \n\t"}, body_start_page=None)
    assert result.status == "rejected"
    assert "empty_or_whitespace_chunk" in result.reasons


def test_missing_hierarchy_and_repeated_section_heading_do_not_exclude_content():
    chunk = {"text": "Short amendment note remains relevant.", "classification_status": "unclassified"}
    result = decide_chunk(
        chunk,
        body_start_page=None,
        warning_codes={"repeated_section_heading", "missing_source_metadata"},
    )
    assert result.status == "accepted"
    assert "repeated_heading_warning_preserved" in result.review_flags
    assert "unclassified_structure_preserved" in result.review_flags


def test_front_matter_and_toc_are_quarantined_not_discarded():
    chunk = {
        "text": "CONTENTS\n1. Preliminary ........ 1\n2. Definitions ........ 2\n3. Powers ........ 3",
        "start_page": 1,
        "end_page": 1,
        "classification_status": "unclassified",
    }
    result = decide_chunk(chunk, body_start_page=4)
    assert result.status == "quarantined"
    assert "front_matter_or_toc_before_detected_body" in result.reasons
    assert "toc_only_chunk" in result.reasons


def test_valid_provision_is_not_quarantined_by_short_chunk_warning():
    chunk = {"text": "145. Witnesses may be cross-examined.", "classification_status": "classified"}
    result = decide_chunk(chunk, body_start_page=None, warning_codes={"short_classified_chunk"})
    assert result.status == "accepted"
    assert "short_chunk_preserved_for_legal_review" in result.review_flags
