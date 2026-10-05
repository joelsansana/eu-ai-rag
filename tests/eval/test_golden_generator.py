from unittest.mock import patch

import pytest

from safety_rag.eval.golden_generator import (
    GoldenGenerationError,
    generate_candidates,
)

CHUNK = {
    "content_hash": "abc123",
    "regulation": "ai_act",
    "celex": "32024R1689",
    "header": "Article 26",
    "text": "Deployers shall...",
}

VALID_JSON = """[
  {"question": "What must deployers do?", "difficulty": "easy", "answerable": true},
  {"question": "Who assigns human oversight?", "difficulty": "medium", "answerable": true},
  {"question": "What happens on a serious incident?", "difficulty": "hard", "answerable": true}
]"""


@patch("safety_rag.eval.golden_generator.generate")
def test_parses_valid_json_on_first_try(mock_generate):
    mock_generate.return_value = VALID_JSON
    result = generate_candidates(CHUNK)
    assert len(result) == 3
    assert mock_generate.call_count == 1


@patch("safety_rag.eval.golden_generator.generate")
def test_attaches_chunk_metadata_to_each_candidate(mock_generate):
    mock_generate.return_value = VALID_JSON
    result = generate_candidates(CHUNK)
    for c in result:
        assert c["content_hash"] == "abc123"
        assert c["regulation"] == "ai_act"
        assert c["celex"] == "32024R1689"


@patch("safety_rag.eval.golden_generator.generate")
def test_strips_markdown_fences(mock_generate):
    mock_generate.return_value = f"```json\n{VALID_JSON}\n```"
    result = generate_candidates(CHUNK)
    assert len(result) == 3


@patch("safety_rag.eval.golden_generator.generate")
def test_retries_once_on_invalid_json_then_succeeds(mock_generate):
    mock_generate.side_effect = ["not json at all", VALID_JSON]
    result = generate_candidates(CHUNK)
    assert len(result) == 3
    assert mock_generate.call_count == 2
    second_call_prompt = mock_generate.call_args_list[1].args[0]
    assert "Reminder" in second_call_prompt


@patch("safety_rag.eval.golden_generator.generate")
def test_raises_after_second_failure(mock_generate):
    mock_generate.side_effect = ["nope", "still nope"]
    with pytest.raises(GoldenGenerationError):
        generate_candidates(CHUNK)
    assert mock_generate.call_count == 2


@patch("safety_rag.eval.golden_generator.generate")
def test_passes_temperature_through(mock_generate):
    mock_generate.return_value = VALID_JSON
    generate_candidates(CHUNK, temperature=0.5)
    for call in mock_generate.call_args_list:
        assert call.kwargs.get("temperature") == 0.5
