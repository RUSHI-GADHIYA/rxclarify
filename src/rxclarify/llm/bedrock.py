"""Claude on Amazon Bedrock via the Messages-API (Mantle) client.

Bedrock model IDs carry the `anthropic.` provider prefix, e.g.
`anthropic.claude-haiku-4-5`. Credentials resolve through the standard AWS
chain (env vars, shared profile, instance role), so nothing is passed here.

Haiku 4.5 predates the 4.6 parameter changes: it does not accept
`output_config.effort` or adaptive thinking, and it does accept `temperature`.
We pin temperature to 0 so eval runs are reproducible.
"""

from __future__ import annotations

import os
import time

from rxclarify.config import get_settings
from rxclarify.llm.base import Completion


class BedrockProvider:
    name = "bedrock"

    def __init__(self, model: str | None = None, region: str | None = None) -> None:
        from anthropic import AnthropicBedrockMantle

        settings = get_settings()
        self.model = model or settings.bedrock_model
        # AWS_REGION is the standard variable name, so it is read directly
        # rather than through the RX_-prefixed Settings class.
        self.region = region or os.environ.get("AWS_REGION") or "us-east-1"
        self._client = AnthropicBedrockMantle(aws_region=self.region)

    def complete(self, *, system: str, user: str, max_tokens: int) -> Completion:
        started = time.perf_counter()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        return Completion(
            text=text.strip(),
            model=self.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=elapsed_ms,
        )
