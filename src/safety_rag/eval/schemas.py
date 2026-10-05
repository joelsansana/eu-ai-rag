from __future__ import annotations

from pydantic import BaseModel, Field

UNANSWERABLE_CATEGORIES = {
    "national_transposition",
    "interplay_absent_instrument",
    "absent_standard",
    "lex_specialis_interplay",
    "absent_instrument",
    "post_enactment_content",
    "post_enactment_fact",
}


class CorpusSnapshot(BaseModel):
    ai_act_sha256: str
    nis2_sha256: str


class UnanswerableQuestion(BaseModel):
    q_id: str
    tier: str = "unanswerable"
    category: str
    question: str = Field(min_length=1)
    unanswerable_because: str
    anchor_terms: dict[str, list[int]] = Field(default_factory=dict)
    must_be_zero: list[str] = Field(default_factory=list)
    nearest_in_corpus: str | None = None
    corpus_snapshot: CorpusSnapshot
    verified_on: str
    verification_method: str
