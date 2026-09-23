from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI


def test_get_client_returns_openai_instance() -> None:
    from safety_rag.generation.llm import get_client

    client = get_client()

    assert isinstance(client, OpenAI)
    assert "minimax.io" in str(client.base_url)


def test_get_client_missing_api_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    from safety_rag.generation.llm import get_client

    with pytest.raises(KeyError, match="MINIMAX_API_KEY"):
        get_client()


def _fake_response(content: str | None) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def test_generate_calls_chat_completions_with_expected_args() -> None:
    from safety_rag.generation.llm import generate

    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat.completions.create.return_value = _fake_response("pong")

    with patch(
        "safety_rag.generation.llm.get_client",
        return_value=mock_client,
    ):
        result = generate("Reply with one word: pong.")

    assert result == "pong"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "MiniMax-M2"
    assert call_kwargs["messages"] == [
        {"role": "user", "content": "Reply with one word: pong."}
    ]


def test_generate_returns_empty_string_when_content_is_none() -> None:
    from safety_rag.generation.llm import generate

    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat.completions.create.return_value = _fake_response(None)

    with patch(
        "safety_rag.generation.llm.get_client",
        return_value=mock_client,
    ):
        result = generate("hi")

    assert result == ""
