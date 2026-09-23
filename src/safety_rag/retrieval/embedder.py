from sentence_transformers import SentenceTransformer

_MODEL = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _MODEL


def embed(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, normalize_embeddings=True).tolist()
