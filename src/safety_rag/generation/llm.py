from __future__ import annotations

import os
import re

from openai import OpenAI


def get_client() -> OpenAI:
    return OpenAI(
        base_url="https://api.minimax.io/v1",
        api_key=os.environ["MINIMAX_API_KEY"],
    )


def generate(prompt: str) -> str:
    client = get_client()

    response = client.chat.completions.create(
        model="MiniMax-M2",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content or ""

    return _strip_thinking(raw)


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove any <think>...</think> reasoning block a model may emit
    inline in its response content (safety net even with thinking disabled)."""
    return _THINK_TAG_RE.sub("", text).strip()
