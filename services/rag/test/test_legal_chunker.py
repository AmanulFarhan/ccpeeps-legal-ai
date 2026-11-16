"""Focused regression tests for the isolated structure-aware chunker."""

import unittest

from services.rag.ingestion.legal_chunker import (
    ChunkingConfig,
    analyze_document,
    chunk_document,
)


def classified(chunks):
    return [chunk for chunk in chunks if chunk["classification_status"] == "classified"]


class LegalChunkingTests(unittest.TestCase):
    def test_bnss_toc_is_fallback_and_actual_body_is_detected(self):
        pages = [
            {"page_number": 1, "text": "TITLE\nARRANGEMENT OF SECTIONS\n1. Short title ........ 18\n2. Definitions ........ 18"},
            {"page_number": 2, "text": "CHAPTER I\nPRELIMINARY\n1. Short title, extent and commencement.\n(1) This Act may be called the Example Act."},
        ]
        chunks, analysis = chunk_document(pages, "bnss")
        self.assertEqual(analysis["body_start_page"], 2)
        self.assertTrue(any("ARRANGEMENT OF SECTIONS" in c["text"] and c["classification_status"] == "unclassified" for c in chunks))
        self.assertTrue(all(c["start_page"] >= 2 for c in classified(chunks)))
        self.assertEqual(classified(chunks)[0]["section_number"], "1")

    def test_kerala_ocr_spacing_preamble_and_part_are_detected(self):
        pages = [{"page_number": 1, "text": "CONTENTS\n1. Short title ...\nACT 6 OF 1959\nAn Act to provide education.\nWHEREAS the legislature considers it necessary.\nPART I\nPRELIMINARY\n1 . Short title\n(1) This Act may be called the Kerala Education Act."}]
        chunks, analysis = chunk_document(pages, "kerala")
        section = classified(chunks)[0]
        self.assertEqual(analysis["parts"], 1)
        self.assertEqual(section["section_number"], "1")
        self.assertEqual(section["part"], "I PRELIMINARY")

    def test_chapter_title_on_following_page_and_section_continuation(self):
        pages = [
            {"page_number": 4, "text": "CHAPTER II\n"},
            {"page_number": 5, "text": "CONSTITUTION OF COURTS\n1. Courts of the State\n(1) Text continues"},
            {"page_number": 6, "text": "on this page.\n2. Territorial divisions\n(1) Divisions are prescribed."},
        ]
        chunks, analysis = chunk_document(pages, "chapter")
        section = classified(chunks)[0]
        self.assertEqual(analysis["chapters"], 1)
        self.assertEqual(section["chapter_number"], "II")
        self.assertEqual(section["chapter_title"], "CONSTITUTION OF COURTS")
        self.assertEqual(section["start_page"], 5)
        self.assertEqual(section["end_page"], 6)
        self.assertIn("on this page.", section["text"])

    def test_subsection_clause_metadata_is_local_to_split_children(self):
        pages = [{"page_number": 7, "text": "1. Application\n(1) First provision.\n(a) Alpha.\n(b) Beta.\n(2) " + "word " * 20 + "\n(a) " + "alpha " * 20 + "\n(b) " + "beta " * 20 + "\n(3) Last."}]
        chunks, analysis = chunk_document(pages, "split", ChunkingConfig(max_words=15))
        children = classified(chunks)
        self.assertGreater(len(children), 1)
        self.assertGreaterEqual(analysis["subsections"], 3)
        self.assertGreaterEqual(analysis["clauses"], 2)
        self.assertTrue(all(c["section_number"] == "1" for c in children))
        self.assertIn("a", {c["clause_number"] for c in children})
        self.assertIn("b", {c["clause_number"] for c in children})
        self.assertTrue(all(c["subsection_number"] in {"1", "2", "3"} for c in children))

    def test_page_provenance_is_child_specific(self):
        pages = [
            {"page_number": 10, "text": "1. Long section\n(1) First part."},
            {"page_number": 11, "text": "(2) " + "word " * 30},
            {"page_number": 12, "text": "(3) Last part."},
        ]
        chunks, _ = chunk_document(pages, "pages", ChunkingConfig(max_words=10))
        children = classified(chunks)
        self.assertGreater(len(children), 1)
        self.assertEqual(children[0]["start_page"], 10)
        self.assertTrue(any(c["start_page"] == 11 for c in children))
        self.assertTrue(any(c["start_page"] == 12 for c in children))

    def test_exact_text_reconstruction_order_and_unclassified_preservation(self):
        pages = [
            {"page_number": 2, "text": "Preamble not recognized as a section.\nCONTENTS\n1. Summary ..."},
            {"page_number": 3, "text": "PART I\nPRELIMINARY\n1. Short title\n(1) Text."},
            {"page_number": 4, "text": "(2) Continued text."},
        ]
        chunks, analysis = chunk_document(pages, "exact")
        original = "\n".join(page["text"] for page in pages)
        self.assertTrue(analysis["exact_reconstruction"])
        self.assertEqual("".join(chunk["text"] for chunk in chunks), original)
        self.assertTrue(any(c["classification_status"] == "unclassified" and "CONTENTS" in c["text"] for c in chunks))
        self.assertLess(chunks[0]["start_page"], classified(chunks)[0]["start_page"])

    def test_empty_and_unrecognized_input_are_lossless(self):
        self.assertEqual(chunk_document([], "empty")[0], [])
        pages = [{"page_number": 1, "text": "Unknown structure\nKeep this legal text."}]
        chunks, analysis = chunk_document(pages, "unknown")
        self.assertEqual("".join(c["text"] for c in chunks), pages[0]["text"])
        self.assertTrue(all(c["classification_status"] == "unclassified" for c in chunks))
        self.assertTrue(analysis["exact_reconstruction"])
        self.assertIsNone(analyze_document([])["body_start_page"])


if __name__ == "__main__":
    unittest.main()
