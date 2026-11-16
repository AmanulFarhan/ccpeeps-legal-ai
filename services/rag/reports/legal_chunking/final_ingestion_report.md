# Legal corpus ingestion report

- Drive PDFs in source manifest: 49
- Intentionally excluded: 2
- Source PDFs processed: 47
- Accepted chunks embedded and stored: 6038
- Rejected chunks preserved separately: 12
- Quarantined chunks preserved separately: 139
- Exact chunk reconstruction/order before selection: True
- BGE-M3: BAAI/bge-m3, 1024 dimensions, normalized dense vectors; failures: 0
- Database verification: {"documents": 47, "distinct_source_drive_ids": 47, "chunks": 6038, "bad_source_urls": 0, "bad_titles": 0, "bad_verified_drive_links": 0, "bad_jurisdictions": 0, "populated_unverified_citations": 0, "bad_vector_dimensions": 0, "non_normalized_vectors": 0, "bad_page_ranges": 0, "orphan_chunks": 0, "duplicate_document_chunk_indexes": 0, "citation_note": "Legal citation fields remain NULL because no authority citation was independently verified; source URLs are checked against Google Drive API metadata.", "matches_expected": true}
- Pilot source PDF exact re-extraction and DB verification: True / True
- Chunk words: average 155.99; max 1200
- Google Drive source links matched the Drive API metadata; legal citation fields remain NULL unless independently verified.
- OCR performed: no. Low-text source pages remain flagged/quarantined as documented by Phase C.
- Low-text OCR review flags from source audit: 18 document-page pairs; no OCR was run.
- Existing duplicate HNSW indexes were retained; no new vector index was created.
- Unique Drive source ID constraint present: True.
- Full `services/rag` test suite: 31 passed.
- Title-based jurisdiction labels: 19 state, 28 central.
- Jurisdiction follows the requested title-prefix rule; document type remains unknown where no trusted structured metadata was available.

## Intentional exclusions

- `Electronic Signature or Electronic Authentication Technique and Procedure (Amendment) Rules, 2015_88e925cd.pdf` (`1WoUidKMTBMNMvfakro3tSSYkZHLcWGfn`): Intentionally excluded: extracted content was reported as gibberish and unsuitable for the legal corpus.
- `test_order.pdf` (`1tyy1nrrsIR-ue8lLIddinJF1zPPxKKVU`): Intentionally excluded: extracted content was reported as gibberish and unsuitable for the legal corpus; this is an irrelevant test document.

## Duplicate titles and corpus scope

- Same title, distinct sources: `Kerala_Rule_Inland_Vessels_Act_1917_(Central_Act_1_of_1917)_b74f2a4d.pdf` (`1V9OALQMmSVqhXt8a4DDr7D9Il-hOAy3y`); `Kerala_Rule_Inland_Vessels_Act_1917_(Central_Act_1_of_1917)_bfe9a7cd.pdf` (`10r0H0KnYmtMiRJh03OdyVihUUQHKh6sZ`). They were retained as separate documents and mapped to distinct UUIDs.
- `test_order.pdf` was excluded from the Acts/Rules ingestion subset; the source audit identifies it as a Chandigarh Police office order.

## Selection review reasons

- `chapter_title_inconsistent`: 77
- `empty_or_whitespace_chunk`: 12
- `front_matter_or_toc_before_detected_body`: 49
- `multiple_section_heading_candidates`: 4
- `overlaps_low_text_ocr_review_page`: 10
- `section_metadata_text_mismatch`: 3
- `subsection_marker_absent`: 1

Original source chunk JSON and quality-selection partitions were preserved. Database chunks refer to document UUIDs; Drive IDs are retained on the document row for traceability.
