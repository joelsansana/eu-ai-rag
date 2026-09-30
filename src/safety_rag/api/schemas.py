from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    regulation: str | None = None  # "ai_act" | "nis2" | None
    article_num: int | None = None
    part: str | None = None  # "recital" | "article" | "annex"


class SourceChunk(BaseModel):
    score: float
    regulation: str
    part: str
    article_num: int | None = None
    recital_num: int | None = None
    annex_id: str | None = None
    chapter: str | None = None
    celex: str
    header: str
    text: str
    content_hash: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: int


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    regulation: str | None = None
    article_num: int | None = None
    part: str | None = None
