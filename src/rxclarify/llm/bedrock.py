"""Claude on Amazon Bedrock as a LangChain chat model.

Uses `ChatBedrockConverse` (Bedrock's Converse API) so Bedrock composes into the
same LCEL chain as OpenAI rather than needing a parallel code path. Credentials
resolve through the standard AWS chain — env vars, shared profile, instance
role — so nothing is passed explicitly here.

Bedrock model IDs carry the `anthropic.` provider prefix, e.g.
`anthropic.claude-haiku-4-5`. Model access is granted per-model *and*
per-region in the Bedrock console; a valid key with no grant still fails.
"""

from __future__ import annotations

import os

from langchain_aws import ChatBedrockConverse

from rxclarify.config import get_settings


def build_bedrock_chat(
    model: str | None = None,
    *,
    region: str | None = None,
    max_tokens: int | None = None,
) -> ChatBedrockConverse:
    settings = get_settings()
    return ChatBedrockConverse(
        model_id=model or settings.bedrock_model,
        # AWS_REGION is the standard variable name, so it is read directly
        # rather than through the RX_-prefixed Settings class.
        region_name=region or os.environ.get("AWS_REGION") or "us-east-1",
        temperature=0,
        max_tokens=max_tokens if max_tokens is not None else settings.max_tokens,
    )
