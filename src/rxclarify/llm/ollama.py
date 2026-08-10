"""Local Ollama provider — the zero-cost development and smoke-test path.

Quality is well below Haiku 4.5; this exists so the pipeline can be built and
exercised without AWS credentials, not to produce publishable eval numbers.
"""

from __future__ import annotations

import time

import httpx

from rxclarify.config import get_settings
from rxclarify.llm.base import Completion


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    def complete(self, *, system: str, user: str, max_tokens: int) -> Completion:
        started = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": 0, "num_predict": max_tokens},
            },
            # Small local models on CPU/4GB GPU can be slow on a long context.
            timeout=300.0,
        )
        response.raise_for_status()
        payload = response.json()
        elapsed_ms = (time.perf_counter() - started) * 1000

        return Completion(
            text=(payload.get("message", {}).get("content") or "").strip(),
            model=self.model,
            input_tokens=payload.get("prompt_eval_count"),
            output_tokens=payload.get("eval_count"),
            latency_ms=elapsed_ms,
        )
