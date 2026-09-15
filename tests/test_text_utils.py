from app.utils.text import extract_citation_ids, parse_list, split_text, stable_id


def test_stable_id_is_deterministic():
    assert stable_id("same") == stable_id("same")
    assert stable_id("same") != stable_id("different")


def test_parse_list_handles_numbered_items():
    parsed = parse_list("1. First\n2) Second\n- Third", ["fallback"], 5)
    assert parsed == ["First", "Second", "Third"]


def test_extract_citations_normalizes_labels():
    assert extract_citation_ids("Fact [source 1] and [Source 2].") == [
        "Source 1",
        "Source 2",
    ]


def test_split_text_creates_multiple_chunks():
    chunks = split_text("word " * 500, chunk_size=200, overlap=40)
    assert len(chunks) > 2
    assert all(chunk.strip() for chunk in chunks)
