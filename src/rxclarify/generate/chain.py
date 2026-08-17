"""The LCEL retrieval-augmented generation chain.

Shape:

    question ──► RunnableParallel ──► {question, docs}
                   ├─ question: passthrough
                   └─ docs:     PgVectorRetriever
                          │
                          └─► assign(message = format ▸ prompt ▸ chat_model)

    result: {"question": str, "docs": [Document], "message": AIMessage}

The chain deliberately stops at the chat model rather than piping through
`StrOutputParser`. The raw `AIMessage` carries `usage_metadata`, which is where
per-query token counts and therefore cost come from — and cost per query is one
of the numbers this project exists to report. A string parser would discard it.

`docs` is carried through to the output because citation validation needs to
know exactly which excerpts the model was shown.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from rxclarify.generate.prompt import build_prompt, format_context
from rxclarify.retrieval.langchain_retriever import to_chunk


def _format_inputs(payload: dict) -> dict:
    docs: list[Document] = payload["docs"]
    return {
        "context": format_context([to_chunk(d) for d in docs]),
        "question": payload["question"],
    }


def build_chain(retriever: BaseRetriever, chat_model: BaseChatModel) -> Runnable:
    generate = (
        RunnableLambda(_format_inputs).with_config(run_name="format_context")
        | build_prompt()
        | chat_model
    )

    return RunnableParallel(
        question=RunnablePassthrough(),
        docs=retriever,
    ).assign(message=generate)
