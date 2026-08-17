"""OpenAI chat model for the LCEL chain.

Reads OPENAI_API_KEY from the environment (langchain-openai's own convention);
the key is never passed through RxClarify's Settings so it cannot end up in a
config dump or a log line.

**Why reasoning effort is "none".** GPT-5-family models are reasoning models,
and langchain-openai deliberately drops `temperature` from the request payload
for them — silently, with no error. The one exception is reasoning effort
"none", which restores it. Answering from six supplied label excerpts is
extraction, not reasoning, so "none" is simultaneously the cheapest setting,
the fastest, and the only one under which `temperature=0` is actually honoured.

**Why there is no seed.** Setting `reasoning` routes the request through
OpenAI's Responses API rather than Chat Completions. `Responses.create()`
accepts `temperature` but has no `seed` parameter at all, so passing one is a
TypeError at call time — note that it still appears in the payload dict, so a
payload inspection alone will not catch this. Chat Completions does accept
`seed`, but langchain-openai strips `temperature` on that path for this model
family, which is the worse trade: temperature dominates determinism and seed
was only ever best-effort. Eval runs are therefore near-deterministic, not
bit-reproducible; record the model version alongside results.

Verified against langchain-openai 1.5.1 / openai 3.1.0.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from rxclarify.config import get_settings

# Effort levels above this cause langchain-openai to strip temperature/seed.
DETERMINISTIC_EFFORT = "none"


def build_openai_chat(
    model: str | None = None,
    *,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    settings = get_settings()
    effort = settings.openai_reasoning_effort

    kwargs: dict = {
        "model": model or settings.openai_model,
        "max_tokens": max_tokens if max_tokens is not None else settings.max_tokens,
        "reasoning": {"effort": effort},
        "timeout": 120,
        "max_retries": 3,
    }
    if effort == DETERMINISTIC_EFFORT:
        # Only honoured at this effort level; sending it otherwise is a no-op
        # that reads like determinism without providing it.
        kwargs["temperature"] = 0

    return ChatOpenAI(**kwargs)
