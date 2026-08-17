"""Provider wiring.

The important assertions here are about `temperature`. langchain-openai
silently drops it for GPT-5-family models at any reasoning effort above
"none" — no error, no warning — and `temperature=0` is the only determinism
control available on this path (the Responses API has no `seed`). If either
behaviour regresses, eval runs stop being comparable and nothing else in the
suite would notice.
"""

import pytest

from rxclarify.llm.factory import PROVIDERS, get_chat_model, provider_name
from rxclarify.llm.openai import build_openai_chat


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")


def _payload(chat):
    return chat._get_request_payload([("human", "hi")])


def test_default_effort_none_keeps_temperature():
    payload = _payload(build_openai_chat())
    assert payload["temperature"] == 0
    assert payload["reasoning"] == {"effort": "none"}


def test_seed_is_never_sent():
    """Responses.create() has no `seed` parameter — sending one is a TypeError.

    It still shows up in the payload dict, so only a real call (or this test)
    catches it.
    """
    import inspect

    from openai.resources.responses import Responses

    assert "seed" not in inspect.signature(Responses.create).parameters
    assert "seed" not in _payload(build_openai_chat())


def test_higher_effort_would_drop_temperature_so_we_do_not_send_it(monkeypatch):
    """Guard against pretending to be deterministic when the API ignores us."""
    from rxclarify import config

    monkeypatch.setenv("RX_OPENAI_REASONING_EFFORT", "low")
    config.get_settings.cache_clear()
    try:
        payload = _payload(build_openai_chat())
        assert "temperature" not in payload
    finally:
        config.get_settings.cache_clear()


def test_max_tokens_is_forwarded():
    # Setting `reasoning` routes langchain-openai through the Responses API,
    # where the output cap is `max_output_tokens` (the Chat Completions path
    # calls it `max_completion_tokens`).
    payload = _payload(build_openai_chat(max_tokens=256))
    assert payload.get("max_output_tokens") == 256


def test_reasoning_effort_routes_through_the_responses_api():
    payload = _payload(build_openai_chat())
    assert "input" in payload and "messages" not in payload


def test_factory_returns_a_chat_model_for_openai():
    chat = get_chat_model("openai")
    assert chat.model_name


def test_factory_rejects_an_unknown_provider():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        get_chat_model("ollama")


def test_provider_name_normalises_case():
    assert provider_name("OpenAI") == "openai"


def test_providers_tuple_is_the_supported_set():
    assert PROVIDERS == ("openai", "bedrock")
