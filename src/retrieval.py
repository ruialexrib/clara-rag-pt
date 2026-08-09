import requests

from src.config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    QDRANT_TIMEOUT,
    TOP_K,
)

from src.embeddings import generate_embedding


def check_qdrant_service() -> bool:
    """
    Check whether Qdrant is reachable and the configured
    CLARA collection exists.

    Returns
    -------
    bool
        True if Qdrant is reachable and the configured
        collection exists. False otherwise.
    """

    try:
        response = requests.get(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}",
            timeout=10,
        )

        response.raise_for_status()

        return True

    except requests.RequestException:
        return False


def vector_search(
    question: str,
    top_k: int = TOP_K,
    document_id: str | None = None,
) -> list[dict]:
    """
    Perform semantic vector search in Qdrant.

    The user question is converted into an embedding using
    the configured Ollama embedding model. The resulting
    vector is then compared with the legal chunks stored in
    the Qdrant collection.

    Parameters
    ----------
    question : str
        Natural-language user question.

    top_k : int
        Maximum number of legal chunks to retrieve.

    document_id : str | None
        Optional filter used to restrict retrieval to a
        specific legal document, such as "CIRS" or "CIVA".

    Returns
    -------
    list[dict]
        Ranked search results returned by Qdrant.

    Raises
    ------
    ValueError
        If the question is empty or `top_k` is not positive.

    requests.HTTPError
        If the Qdrant API request fails.
    """

    if not question or not question.strip():
        raise ValueError(
            "A pergunta não pode estar vazia."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k deve ser superior a zero."
        )

    # Generate the semantic representation of the question.
    query_vector = generate_embedding(
        question
    )

    search_payload = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True,
        "with_vector": False,
    }

    # Optionally restrict retrieval to a specific legal document.
    if document_id is not None:
        search_payload["filter"] = {
            "must": [
                {
                    "key": "document_id",
                    "match": {
                        "value": document_id
                    },
                }
            ]
        }

    response = requests.post(
        (
            f"{QDRANT_URL}/collections/"
            f"{QDRANT_COLLECTION}/points/search"
        ),
        json=search_payload,
        timeout=QDRANT_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    return data.get(
        "result",
        []
    )


def build_context(
    results: list[dict],
) -> str:
    """
    Build the legal context that will be sent to the LLM.

    Each retrieved chunk is enriched with its legal source
    metadata, including the document, article, article title,
    and page range.

    Parameters
    ----------
    results : list[dict]
        Ranked search results returned by Qdrant.

    Returns
    -------
    str
        Formatted legal context ready to be included in
        the grounded LLM prompt.
    """

    if not results:
        return ""

    context_blocks = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        payload = result.get(
            "payload",
            {}
        )

        document_id = payload.get(
            "document_id",
            "Documento desconhecido",
        )

        article = payload.get(
            "article",
            "Artigo desconhecido",
        )

        article_title = payload.get(
            "article_title"
        )

        page_start = payload.get(
            "page_start"
        )

        page_end = payload.get(
            "page_end"
        )

        text = payload.get(
            "text",
            "",
        )

        # Build a human-readable legal source reference.
        reference_lines = [
            f"[Fonte {rank}]",
            f"Documento: {document_id}",
            f"Artigo: {article}",
        ]

        if article_title:
            reference_lines.append(
                f"Epígrafe: {article_title}"
            )

        if (
            page_start is not None
            and page_end is not None
        ):
            if page_start == page_end:
                reference_lines.append(
                    f"Página: {page_start}"
                )

            else:
                reference_lines.append(
                    f"Páginas: "
                    f"{page_start}-{page_end}"
                )

        reference = "\n".join(
            reference_lines
        )

        block = (
            f"{reference}\n\n"
            f"{text}"
        )

        context_blocks.append(
            block
        )

    separator = (
        "\n\n"
        + "-" * 80
        + "\n\n"
    )

    return separator.join(
        context_blocks
    )


def get_sources(
    results: list[dict],
) -> list[dict]:
    """
    Extract source metadata from Qdrant search results.

    This function removes the vector-search response details
    that are not required by the application and returns a
    concise source representation suitable for API responses
    and user-facing source presentation.

    Parameters
    ----------
    results : list[dict]
        Ranked search results returned by Qdrant.

    Returns
    -------
    list[dict]
        Structured metadata for the retrieved legal sources.
    """

    sources = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        payload = result.get(
            "payload",
            {}
        )

        sources.append({
            "rank": rank,
            "score": result.get(
                "score"
            ),
            "document_id": payload.get(
                "document_id"
            ),
            "article": payload.get(
                "article"
            ),
            "article_title": payload.get(
                "article_title"
            ),
            "page_start": payload.get(
                "page_start"
            ),
            "page_end": payload.get(
                "page_end"
            ),
            "chunk_id": payload.get(
                "chunk_id"
            ),
        })

    return sources


def retrieve(
    question: str,
    top_k: int = TOP_K,
    document_id: str | None = None,
) -> dict:
    """
    Execute the complete CLARA retrieval pipeline.

    Flow
    ----
    question
        ↓
    embedding
        ↓
    Qdrant
        ↓
    Top-K legal chunks
        ↓
    context + sources

    Parameters
    ----------
    question : str
        Natural-language user question.

    top_k : int
        Number of legal chunks to retrieve.

    document_id : str | None
        Optional legal-document filter.

    Returns
    -------
    dict
        Retrieval result containing the raw Qdrant results,
        formatted legal context, and structured source metadata.
    """

    results = vector_search(
        question=question,
        top_k=top_k,
        document_id=document_id,
    )

    context = build_context(
        results
    )

    sources = get_sources(
        results
    )

    return {
        "question": question,
        "document_filter": document_id,
        "top_k": top_k,
        "results": results,
        "context": context,
        "sources": sources,
    }