"""Generation-path tests. No network, no database, no real model."""

from rxclarify.generate.answer import Answer, answer_question, is_refusal, parse_citations
from rxclarify.generate.prompt import REFUSAL_TOKEN, build_user_prompt
from rxclarify.llm.base import Completion
from rxclarify.retrieval.base import RetrievedChunk


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


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self.chunks[:top_k]


class FakeProvider:
    name = "fake"

    def __init__(self, text: str) -> None:
        self.text = text
        self.last_user_prompt: str | None = None

    def complete(self, *, system: str, user: str, max_tokens: int) -> Completion:
        self.last_user_prompt = user
        return Completion(text=self.text, model="fake-1", input_tokens=10, output_tokens=5)


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
    retriever = FakeRetriever([_chunk(1), _chunk(2)])
    provider = FakeProvider("Avoid the combination [C1][C9].")

    result = answer_question("q", retriever=retriever, provider=provider, top_k=2)

    assert result.cited_markers == [1]
    assert result.invalid_markers == [9]
    assert not result.refused
    assert result.model == "fake-1"


def test_answer_question_marks_refusal():
    retriever = FakeRetriever([_chunk(1)])
    provider = FakeProvider(f"{REFUSAL_TOKEN}\nThe excerpts do not cover pediatric dosing.")

    result = answer_question("q", retriever=retriever, provider=provider, top_k=1)

    assert result.refused
    assert result.cited_markers == []
    # A refusal citing nothing is correct, not a warning-worthy uncited answer.
    assert not result.uncited


def test_uncited_flags_a_non_refusal_answer_with_no_citations():
    retriever = FakeRetriever([_chunk(1)])
    provider = FakeProvider("Warfarin interacts with many drugs.")

    result = answer_question("q", retriever=retriever, provider=provider, top_k=1)

    assert result.uncited
    assert result.invalid_markers == []


def test_cited_chunks_resolves_markers_back_to_chunks():
    chunks = [_chunk(1, "first"), _chunk(2, "second")]
    answer = Answer(question="q", text="[C2]", chunks=chunks, cited_markers=[2])
    assert [c.text for c in answer.cited_chunks()] == ["second"]


def test_top_k_is_passed_through_to_the_retriever():
    retriever = FakeRetriever([_chunk(i) for i in range(1, 6)])
    provider = FakeProvider("[C1]")

    answer_question("q", retriever=retriever, provider=provider, top_k=3)

    assert retriever.calls == [("q", 3)]


def test_empty_retrieval_still_demands_the_refusal_shape():
    prompt = build_user_prompt("q", [])
    assert REFUSAL_TOKEN in prompt
    assert "no excerpts were retrieved" in prompt


def test_context_blocks_are_numbered_for_citation():
    prompt = build_user_prompt("Any interaction?", [_chunk(1, "alpha"), _chunk(2, "beta")])
    assert "[C1] Coumadin — drug_interactions" in prompt
    assert "[C2] Coumadin — drug_interactions" in prompt
    assert "alpha" in prompt and "beta" in prompt
    assert prompt.rstrip().endswith("QUESTION: Any interaction?")
