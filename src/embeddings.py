import requests

from src.config import (
    OLLAMA_URL,
    EMBEDDING_MODEL,
    OLLAMA_TIMEOUT,
)


def generate_embedding(
    text: str,
    model: str = EMBEDDING_MODEL,
) -> list[float]:
    """
    Generate a semantic embedding for the provided text using Ollama.

    Parameters
    ----------
    text : str
        Text to be converted into a semantic vector.

    model : str
        Embedding model available in Ollama.

    Returns
    -------
    list[float]
        Generated embedding vector.

    Raises
    ------
    ValueError
        If the provided text is empty.

    RuntimeError
        If Ollama does not return an embedding.

    requests.HTTPError
        If the Ollama API request fails.
    """

    if not text or not text.strip():
        raise ValueError(
            "Não é possível gerar um embedding "
            "para um texto vazio."
        )

    # Generate the embedding using the Ollama API.
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": model,
            "input": text.strip(),
        },
        timeout=OLLAMA_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    embeddings = data.get("embeddings")

    if not embeddings:
        raise RuntimeError(
            "O Ollama não devolveu nenhum embedding."
        )

    # Ollama returns a list of embeddings because the API
    # also supports multiple input texts. CLARA sends one
    # text at a time, so only the first vector is required.
    return embeddings[0]


def check_embedding_service() -> bool:
    """
    Check whether Ollama is available and the configured
    embedding model is installed.

    Returns
    -------
    bool
        True if Ollama is reachable and the configured
        embedding model is available. False otherwise.
    """

    try:
        # Retrieve the list of models currently available in Ollama.
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=10,
        )

        response.raise_for_status()

        models = response.json().get(
            "models",
            []
        )

        installed_models = [
            model.get("name", "")
            for model in models
        ]

        # Accept both the exact model name and tagged variants,
        # such as "bge-m3:latest".
        return any(
            name == EMBEDDING_MODEL
            or name.startswith(
                f"{EMBEDDING_MODEL}:"
            )
            for name in installed_models
        )

    except requests.RequestException:
        return False