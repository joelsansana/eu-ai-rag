from __future__ import annotations

import os

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

    return response.choices[0].message.content or ""
