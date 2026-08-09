import time

from src.config import TOP_K

from src.retrieval import (
    retrieve,
)

from src.llm import (
    generate_answer,
    stream_answer,
)


def ask_clara(
    question: str,
    top_k: int = TOP_K,
    document_id: str | None = None,
) -> dict:
    """
    Execute the complete CLARA RAG pipeline.

    Flow
    ----
    question
        ↓
    embeddings
        ↓
    Qdrant
        ↓
    Top-K chunks
        ↓
    context
        ↓
    AMALIA
        ↓
    answer + sources

    Parameters
    ----------
    question : str
        Natural-language user question.

    top_k : int
        Number of legal chunks to retrieve.

    document_id : str | None
        Optional document filter, for example
        "CIRS" or "CIVA".

    Returns
    -------
    dict
        Complete RAG result containing the generated
        answer, retrieved sources, filters, retrieval
        settings, and execution times.

    Raises
    ------
    ValueError
        If the provided question is empty.
    """

    if not question or not question.strip():
        raise ValueError(
            "A pergunta não pode estar vazia."
        )

    # =====================================================
    # Retrieval
    # =====================================================

    retrieval_start = time.perf_counter()

    retrieval_result = retrieve(
        question=question,
        top_k=top_k,
        document_id=document_id,
    )

    retrieval_seconds = (
        time.perf_counter()
        -
        retrieval_start
    )

    context = retrieval_result[
        "context"
    ]

    sources = retrieval_result[
        "sources"
    ]

    # =====================================================
    # Generation
    # =====================================================

    generation_start = time.perf_counter()

    answer = generate_answer(
        question=question,
        context=context,
    )

    generation_seconds = (
        time.perf_counter()
        -
        generation_start
    )

    # =====================================================
    # Final result
    # =====================================================

    total_seconds = (
        retrieval_seconds
        +
        generation_seconds
    )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "document_filter": document_id,
        "top_k": top_k,
        "retrieval_seconds": round(
            retrieval_seconds,
            3,
        ),
        "generation_seconds": round(
            generation_seconds,
            3,
        ),
        "total_seconds": round(
            total_seconds,
            3,
        ),
    }


def stream_clara(
    question: str,
    top_k: int = TOP_K,
    document_id: str | None = None,
) -> dict:
    """
    Execute the CLARA RAG pipeline with streaming generation.

    Retrieval is completed first. The retrieved legal
    context is then passed to AMALIA, whose response is
    returned progressively through a generator.

    Parameters
    ----------
    question : str
        Natural-language user question.

    top_k : int
        Number of legal chunks to retrieve.

    document_id : str | None
        Optional document filter.

    Returns
    -------
    dict
        Result containing the streaming generator,
        retrieved sources, retrieval configuration,
        and retrieval execution time.

    Raises
    ------
    ValueError
        If the provided question is empty.
    """

    if not question or not question.strip():
        raise ValueError(
            "A pergunta não pode estar vazia."
        )

    # =====================================================
    # Retrieval
    # =====================================================

    retrieval_start = time.perf_counter()

    retrieval_result = retrieve(
        question=question,
        top_k=top_k,
        document_id=document_id,
    )

    retrieval_seconds = (
        time.perf_counter()
        -
        retrieval_start
    )

    context = retrieval_result[
        "context"
    ]

    sources = retrieval_result[
        "sources"
    ]

    # =====================================================
    # Streaming generation
    # =====================================================

    answer_stream = stream_answer(
        question=question,
        context=context,
    )

    return {
        "question": question,
        "stream": answer_stream,
        "sources": sources,
        "document_filter": document_id,
        "top_k": top_k,
        "retrieval_seconds": round(
            retrieval_seconds,
            3,
        ),
    }


def chat(
    question: str,
    document_id: str | None = None,
) -> str:
    """
    Provide a simplified CLARA interface.

    This helper returns only the generated answer text,
    without exposing retrieval metadata or execution times.

    Parameters
    ----------
    question : str
        User question.

    document_id : str | None
        Optional document filter.

    Returns
    -------
    str
        Answer generated by CLARA.
    """

    result = ask_clara(
        question=question,
        document_id=document_id,
    )

    return result[
        "answer"
    ]