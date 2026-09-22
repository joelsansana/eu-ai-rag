from safety_rag.ingestion.chunker import chunk_records


def _base_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "regulation": "ai_act",
        "part": "article",
        "article_num": 26,
        "recital_num": None,
        "annex_id": None,
        "chapter": None,
        "celex": "32024R1689",
        "effective_date": "2024-08-01",
        "lang": "EN",
        "title": "Obligations of deployers of high-risk AI systems",
        "text": "Deployers of high-risk AI systems shall take appropriate \
        measures.",
        "html_path": "tests/fixtures/32024R1689.html",
    }
    record.update(overrides)
    return record


def test_annex_iii_is_split_without_dropping_rows() -> None:
    rows = [
        "Biometric identification and categorisation of natural persons.",
        "Management and operation of critical infrastructure.",
        "Education and vocational training.",
        "Employment, workers management and access to self-employment.",
    ]

    record = _base_record(
        part="annex",
        article_num=None,
        annex_id="III",
        title="High-risk AI systems referred to in Article 6(2)",
        text="\n\n".join(row * 30 for row in rows),
    )

    chunks = chunk_records([record])

    assert len(chunks) > 1
    assert all(chunk["part"] == "annex" for chunk in chunks)
    assert all(chunk["annex_id"] == "III" for chunk in chunks)

    combined_text = "\n\n".join(chunk["text"] for chunk in chunks)

    for row in rows:
        expanded_row = row * 30
        assert expanded_row in combined_text


def test_recital_gets_contextual_header() -> None:
    record = _base_record(
        part="recital",
        article_num=None,
        recital_num=42,
        title="",
        text=(
            "This Regulation should be applied in accordance with "
            "the fundamental rights recognised by the Charter."
        ),
    )

    chunks = chunk_records([record])

    assert len(chunks) == 1
    chunk = chunks[0]

    assert chunk["part"] == "recital"
    assert chunk["recital_num"] == 42
    assert chunk["header"].startswith(
        "Recital 42, Regulation (EU) 2024/1689 (AI Act), "
        "effective 1 August 2024"
    )
    assert chunk["text"].startswith(chunk["header"])
    assert record["text"] in chunk["text"]


def test_cross_reference_is_preserved_verbatim() -> None:
    cross_reference = (
        "As referred to in Article 14, deployers shall ensure compliance."
    )

    record = _base_record(text=cross_reference)

    chunks = chunk_records([record])

    assert len(chunks) == 1
    assert cross_reference in chunks[0]["text"]


def test_empty_subdivision_is_skipped_without_crashing() -> None:
    empty_record = _base_record(
        text="",
        title="",
    )
    valid_record = _base_record(
        text="This is valid article content.",
    )

    chunks = chunk_records([empty_record, valid_record])

    assert len(chunks) == 1
    assert chunks[0]["text"].endswith("This is valid article content.")
