import os


# =========================================================
# Ollama configuration
# =========================================================

# Base URL of the local Ollama service.
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

# Embedding model used both for corpus indexing and query retrieval.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "bge-m3"
)

# Quantised AMALIA model used for grounded answer generation.
AMALIA_MODEL = os.getenv(
    "AMALIA_MODEL",
    "hf.co/ruialexrib/AMALIA-9B-0626-SFT-GGUF:Q3_K_M"
)


# =========================================================
# Qdrant configuration
# =========================================================

# Base URL of the Qdrant vector database.
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

# Name of the Qdrant collection containing the CLARA legal corpus.
QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "clara_bge_m3"
)


# =========================================================
# Retrieval configuration
# =========================================================

# Number of legal chunks retrieved for each user question.
TOP_K = int(
    os.getenv(
        "TOP_K",
        "5"
    )
)


# =========================================================
# LLM generation configuration
# =========================================================

# Low temperature is used to favour more deterministic responses.
TEMPERATURE = float(
    os.getenv(
        "TEMPERATURE",
        "0.1"
    )
)

# Maximum time, in seconds, allowed for Ollama requests.
OLLAMA_TIMEOUT = int(
    os.getenv(
        "OLLAMA_TIMEOUT",
        "300"
    )
)

# Maximum time, in seconds, allowed for Qdrant requests.
QDRANT_TIMEOUT = int(
    os.getenv(
        "QDRANT_TIMEOUT",
        "120"
    )
)


# =========================================================
# CLARA application metadata
# =========================================================

APP_NAME = "CLARA"

APP_DESCRIPTION = (
    "CLARA (Consulta de Legislação Assistida por Recuperação Aumentada) "
    "is an application for querying Portuguese legislation using "
    "Retrieval-Augmented Generation. It combines semantic retrieval "
    "over an indexed legal corpus with a locally hosted language model "
    "to generate answers grounded in Portuguese legal codes and regulations."
)

APP_VERSION = "0.1.0"