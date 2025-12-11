from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class OpenAIClient:
    """Thin wrapper around OpenAI SDK with sane defaults.

    - Loads OPENAI_API_KEY from environment or .env
    - Defaults to a fast and cost-effective model: gpt-5-nano
    """

    def __init__(self, api_key: Optional[str] = None, default_model: str = "gpt-5-nano"):
        load_dotenv(override=False)

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export it or add to a .env file at project root."
            )

        if OpenAI is None:
            raise RuntimeError(
                "openai package not found. Please install with: pip install openai"
            )

        self.client = OpenAI(api_key=key)
        self.default_model = default_model

    def generate(self, prompt: str, model: Optional[str] = None, temperature: float = 0.3) -> str:
        """Create a single-turn completion using Chat Completions API semantics."""
        mdl = model or self.default_model
        
        api_params = {
            "model": mdl,
            "messages": [
                {"role": "system", "content": "You are a helpful, precise writing assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        
        if "gpt-5" not in mdl.lower():
            api_params["temperature"] = temperature
        
        resp = self.client.chat.completions.create(**api_params)
        return resp.choices[0].message.content or ""


def get_openai_client(default_model: str = "gpt-5-nano") -> OpenAIClient:
    return OpenAIClient(default_model=default_model)
