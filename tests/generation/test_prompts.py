from types import SimpleNamespace

from safety_rag.generation.prompts import (
    MAX_CHUNK_CHARS,
    RAG_PROMPT,
    _format_source_header,
    _truncate,
    build_prompt,
    format_context,
)


def make_result(payload: dict | None, score: float) -> SimpleNamespace:
    # Stand-in for a qdrant_client ScoredPoint: just needs .score and .payload.
    return SimpleNamespace(score=score, payload=payload)


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("short text") == "short text"

    def test_text_exactly_at_limit_unchanged(self):
        text = "x" * MAX_CHUNK_CHARS
        assert _truncate(text) == text

    def test_text_over_limit_is_truncated_with_ellipsis(self):
        text = "x" * (MAX_CHUNK_CHARS + 100)
        result = _truncate(text)
        assert len(result) == MAX_CHUNK_CHARS
        assert result.endswith("…")

    def test_truncation_strips_trailing_whitespace_before_ellipsis(self):
        text = "a" * (MAX_CHUNK_CHARS - 1) + "   more text that gets cut"
        result = _truncate(text)
        assert not result[:-1].endswith(" ")
        assert result.endswith("…")

    def test_custom_max_chars(self):
        result = _truncate("hello world", max_chars=5)
        assert len(result) == 5
        assert result == "hell…"

    def test_empty_string(self):
        assert _truncate("") == ""


# ---------------------------------------------------------------------------
# _format_source_header
# ---------------------------------------------------------------------------


class TestFormatSourceHeader:
    def test_article_chunk(self):
        payload = {
            "regulation": "ai_act",
            "article_num": "26",
            "chapter": "4 — Deployer obligations",
        }
        header = _format_source_header(payload)
        assert (
            header == "Regulation: ai_act | Article: 26 | Chapter: 4 — "
            "Deployer obligations"
        )

    def test_recital_only_chunk(self):
        payload = {"regulation": "ai_act", "recital_num": "27"}
        header = _format_source_header(payload)
        assert header == "Regulation: ai_act | Recital: 27"
        assert "Article" not in header

    def test_annex_only_chunk(self):
        payload = {"regulation": "ai_act", "annex_id": "III"}
        header = _format_source_header(payload)
        assert header == "Regulation: ai_act | Annex: III"

    def test_regulation_only(self):
        payload = {"regulation": "nis2"}
        assert _format_source_header(payload) == "Regulation: nis2"

    def test_field_order_is_stable(self):
        # article, chapter, annex, recital -- in that order when all present
        payload = {
            "regulation": "ai_act",
            "article_num": "9",
            "chapter": "2",
            "annex_id": "I",
            "recital_num": "5",
        }
        header = _format_source_header(payload)
        assert header == (
            "Regulation: ai_act | Article: 9 | Chapter: 2 | Annex: I | "
            "Recital: 5"
        )

    def test_missing_regulation_renders_none(self):
        # defensive: shouldn't crash even with a malformed payload
        header = _format_source_header({})
        assert header == "Regulation: None"


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_empty_results_returns_empty_string(self):
        assert format_context([]) == ""

    def test_single_result_format(self):
        result = make_result(
            {
                "regulation": "ai_act",
                "article_num": "26",
                "chapter": "4 — Deployer obligations",
                "text": "Deployers of high-risk AI systems shall...",
            },
            score=0.91,
        )
        context = format_context([result])
        expected = (
            "[Source 1 | Regulation: ai_act | Article: 26 | "
            "Chapter: 4 — Deployer obligations]\n"
            "Deployers of high-risk AI systems shall..."
        )
        assert context == expected

    def test_sorts_by_score_descending(self):
        low = make_result(
            {"regulation": "nis2", "text": "low score chunk"}, score=0.3
        )
        high = make_result(
            {"regulation": "ai_act", "text": "high score chunk"}, score=0.9
        )
        mid = make_result(
            {"regulation": "nis2", "text": "mid score chunk"}, score=0.6
        )

        context = format_context([low, high, mid])

        # "Source 1" should be the highest-scoring chunk
        assert context.index("[Source 1") < context.index("high score chunk")
        assert context.index("high score chunk") < context.index(
            "mid score chunk"
        )
        assert context.index("mid score chunk") < context.index(
            "low score chunk"
        )

    def test_source_numbering_reflects_sorted_order_not_input_order(self):
        low = make_result({"regulation": "a", "text": "chunk-low"}, score=0.1)
        high = make_result(
            {"regulation": "b", "text": "chunk-high"}, score=0.99
        )

        context = format_context([low, high])

        assert "[Source 1 | Regulation: b]\nchunk-high" in context
        assert "[Source 2 | Regulation: a]\nchunk-low" in context

    def test_blocks_separated_by_blank_line(self):
        r1 = make_result({"regulation": "a", "text": "first"}, score=0.9)
        r2 = make_result({"regulation": "b", "text": "second"}, score=0.5)
        context = format_context([r1, r2])
        assert "\n\n" in context
        blocks = context.split("\n\n")
        assert len(blocks) == 2

    def test_missing_text_field_defaults_to_empty(self):
        result = make_result({"regulation": "ai_act"}, score=0.5)
        context = format_context([result])
        assert context == "[Source 1 | Regulation: ai_act]\n"

    def test_none_payload_does_not_crash(self):
        result = make_result(None, score=0.5)
        context = format_context([result])
        assert context == "[Source 1 | Regulation: None]\n"

    def test_respects_custom_max_chunk_chars(self):
        result = make_result(
            {"regulation": "ai_act", "text": "x" * 50}, score=0.9
        )
        context = format_context([result], max_chunk_chars=10)
        # header line + truncated 10-char text
        text_line = context.split("\n", 1)[1]
        assert len(text_line) == 10
        assert text_line.endswith("…")

    def test_each_chunk_truncated_independently(self):
        long_a = make_result({"regulation": "a", "text": "a" * 2000}, score=0.9)
        long_b = make_result({"regulation": "b", "text": "b" * 2000}, score=0.8)
        context = format_context([long_a, long_b])
        for block in context.split("\n\n"):
            text_line = block.split("\n", 1)[1]
            assert len(text_line) == MAX_CHUNK_CHARS


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_embeds_question(self):
        result = make_result(
            {"regulation": "ai_act", "text": "some text"}, score=0.8
        )
        prompt = build_prompt("What are deployer obligations?", [result])
        assert "User question: What are deployer obligations?" in prompt

    def test_embeds_context(self):
        result = make_result(
            {"regulation": "ai_act", "article_num": "26", "text": "some text"},
            score=0.8,
        )
        prompt = build_prompt("question", [result])
        assert "[Source 1 | Regulation: ai_act | Article: 26]" in prompt
        assert "some text" in prompt

    def test_matches_rag_prompt_template_structure(self):
        result = make_result(
            {"regulation": "ai_act", "text": "text"}, score=0.8
        )
        prompt = build_prompt("q", [result])
        expected = RAG_PROMPT.format(
            context=format_context([result]), question="q"
        )
        assert prompt == expected

    def test_no_results_still_produces_valid_prompt(self):
        prompt = build_prompt("question with no matches", [])
        assert (
            "Retrieved context:\n\n\nUser question:"
            in prompt.replace("Retrieved context:\n", "Retrieved context:\n")
            or "User question: question with no matches" in prompt
        )

    def test_instructions_and_fallback_phrase_present(self):
        prompt = build_prompt("q", [])
        assert '"The corpus does not address this."' in prompt
        assert "[Regulation X, Article Y]" in prompt

    def test_context_with_curly_braces_does_not_break_formatting(self):
        # Regulatory text containing literal braces (e.g. from a formula or
        # citation) must not be mis-parsed by str.format on the template.
        result = make_result(
            {"regulation": "ai_act", "text": "See Annex {III} for details."},
            score=0.8,
        )
        prompt = build_prompt("q", [result])
        assert "See Annex {III} for details." in prompt
