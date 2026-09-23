from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer


def _fake_vectors(n: int, dim: int = 384) -> np.ndarray:
    return np.zeros((n, dim), dtype=np.float32)


def test_get_model_is_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("safety_rag.retrieval.embedder._MODEL", None)
    sentinel = MagicMock(spec=SentenceTransformer)
    monkeypatch.setattr(
        "safety_rag.retrieval.embedder.SentenceTransformer",
        lambda *args, **kwargs: sentinel,
    )
    from safety_rag.retrieval.embedder import get_model

    first = get_model()
    second = get_model()

    assert first is second
    assert first is sentinel


def test_embed_calls_encode_with_normalize_true() -> None:
    mock_model = MagicMock()
    mock_model.encode.return_value = _fake_vectors(1)

    with patch(
        "safety_rag.retrieval.embedder.get_model",
        return_value=mock_model,
    ):
        from safety_rag.retrieval.embedder import embed

        embed(["test"])

    mock_model.encode.assert_called_once()
    call_kwargs = mock_model.encode.call_args.kwargs
    assert call_kwargs["normalize_embeddings"] is True


def test_embed_returns_list_of_lists() -> None:
    mock_model = MagicMock()
    mock_model.encode.return_value = _fake_vectors(2)

    with patch(
        "safety_rag.retrieval.embedder.get_model",
        return_value=mock_model,
    ):
        from safety_rag.retrieval.embedder import embed

        result = embed(["a", "b"])

    assert isinstance(result, list)
    assert all(isinstance(v, list) for v in result)


def test_embed_returns_one_vector_per_input() -> None:
    mock_model = MagicMock()
    mock_model.encode.return_value = _fake_vectors(3)

    with patch(
        "safety_rag.retrieval.embedder.get_model",
        return_value=mock_model,
    ):
        from safety_rag.retrieval.embedder import embed

        result = embed(["a", "b", "c"])

    assert len(result) == 3


def test_embed_returns_384_dim_vectors() -> None:
    mock_model = MagicMock()
    mock_model.encode.return_value = _fake_vectors(2, dim=384)

    with patch(
        "safety_rag.retrieval.embedder.get_model",
        return_value=mock_model,
    ):
        from safety_rag.retrieval.embedder import embed

        result = embed(["a", "b"])

    assert len(result[0]) == 384
    assert len(result[1]) == 384
