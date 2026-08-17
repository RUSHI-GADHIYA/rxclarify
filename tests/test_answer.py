"""Generation-path tests.

These drive the real LCEL chain — real retriever interface, real
ChatPromptTemplate, real citation validation — with only the chat model faked.
No network, no database, no API key.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rxclarify.generate.answer import Answer, answer_question, is_refusal, parse_citations
from rxclarify.generate.chain import build_chain
from rxclarify.generate.prompt import NO_CONTEXT_SENTINEL, REFUSAL_TOKEN, build_prompt
from rxclarify.retrieval.base import RetrievedChunk
from rxclarify.retrieval.langchain_retriever import to_document


def _chunk(marker: int, text: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=marker,
        label_id=1,
        drug="Coumadin",
        section="drug_interactions",
        text=text,
        score=0.9,
        marker=marker,
    )


class FakeRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunks: list = []
    k: int = 6

    def _get_relevant_documents(self, query, *, run_manager) -> list[Document]:
        return [to_document(c) for c in self.chunks[: self.k]]


def _model(text: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


def _run(model_text: str, chunks: list[RetrievedChunk], **kwargs) -> Answer:
    return answer_question(
        "q",
        retriever=FakeRetriever(chunks=chunks),
        chat_model=_model(model_text),
        **kwargs,
    )


def test_parse_citations_splits_valid_from_invalid():
    valid, invalid = parse_citations("Yes [C1], but see [C7] and again [C1].", {1, 2, 3})
    assert valid == [1]
    assert invalid == [7]


def test_parse_citations_is_case_and_space_tolerant():
    valid, _ = parse_citations("supported [c2] and [ C3 ]", {2, 3})
    assert valid == [2, 3]


def test_is_refusal_detects_the_token():
    assert is_refusal(f"{REFUSAL_TOKEN}\nNo interaction data was retrieved.")
    assert not is_refusal("Fluconazole increases warfarin exposure [C1].")


def test_answer_question_flags_hallucinated_citation():
    result = _run("Avoid the combination [C1][C9].", [_chunk(1), _chunk(2)])

    assert result.cited_markers == [1]
    assert result.invalid_markers == [9]
    assert not result.refused


def test_answer_question_marks_refusal():
    result = _run(f"{REFUSAL_TOKEN}\nThe excerpts do not cover pediatric dosing.", [_chunk(1)])

    assert result.refused
    assert result.cited_markers == []
    # A refusal citing nothing is correct, not a warning-worthy uncited answer.
    assert not result.uncited


def test_uncited_flags_a_non_refusal_answer_with_no_citations():
    result = _run("Warfarin interacts with many drugs.", [_chunk(1)])

    assert result.uncited
    assert result.invalid_markers == []


def test_cited_chunks_resolves_markers_back_to_chunks():
    chunks = [_chunk(1, "first"), _chunk(2, "second")]
    answer = Answer(question="q", text="[C2]", chunks=chunks, cited_markers=[2])
    assert [c.text for c in answer.cited_chunks()] == ["second"]


def test_top_k_limits_the_documents_reaching_the_model():
    result = _run("[C1]", [_chunk(i) for i in range(1, 6)], top_k=3)
    assert len(result.chunks) == 3


def test_usage_metadata_is_carried_onto_the_answer():
    """Token counts drive the cost-per-query metric, so the chain must not drop them."""
    message = AIMessage(
        content="Yes [C1].",
        usage_metadata={"input_tokens": 1200, "output_tokens": 42, "total_tokens": 1242},
    )
    result = answer_question(
        "q",
        retriever=FakeRetriever(chunks=[_chunk(1)]),
        chat_model=GenericFakeChatModel(messages=iter([message])),
    )
    assert result.input_tokens == 1200
    assert result.output_tokens == 42
    assert result.latency_ms is not None


def test_chain_returns_docs_alongside_the_message():
    """Citation validation needs the exact excerpts the model was shown."""
    chain = build_chain(FakeRetriever(chunks=[_chunk(1)]), _model("[C1]"))
    out = chain.invoke("does it interact?")

    assert sorted(out) == ["docs", "message", "question"]
    assert out["question"] == "does it interact?"
    assert out["docs"][0].metadata["marker"] == 1


def test_empty_retrieval_still_produces_a_well_formed_prompt():
    rendered = build_prompt().format(context=NO_CONTEXT_SENTINEL, question="q")
    assert NO_CONTEXT_SENTINEL in rendered
    assert REFUSAL_TOKEN in rendered


def test_empty_retrieval_is_handled_end_to_end():
    result = _run(f"{REFUSAL_TOKEN} nothing retrieved.", [])
    assert result.refused
    assert result.chunks == []


@pytest.mark.parametrize("hostile", ["{context}", "dose {1-2} mg", "}{"])
def test_label_text_with_braces_is_not_treated_as_a_template_variable(hostile):
    """SPL dose tables contain braces; they must never be interpolated."""
    chain = build_chain(FakeRetriever(chunks=[_chunk(1, hostile)]), _model("[C1]"))
    out = chain.invoke("q")
    assert out["docs"][0].page_content == hostile
