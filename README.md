<div align="center">

# CLARA — Portuguese Legislation RAG Assistant

### Consulta de Legislação Assistida por Recuperação Aumentada

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-DC244C)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20Models-black)](https://ollama.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Portuguese legislation · Local RAG · Semantic retrieval · Grounded answers**

</div>

---

## About

**CLARA** (*Consulta de Legislação Assistida por Recuperação Aumentada*) is a local Retrieval-Augmented Generation system for natural-language consultation of Portuguese legislation.

The pipeline uses **BGE-M3** for embeddings, **Qdrant** for vector retrieval, **AMALIA-9B** for grounded answer generation, **FastAPI** for the API, and **Open WebUI** for the chat interface. The complete system runs locally without external LLM APIs.

---

## Architecture

```text
Open WebUI
    │
    ▼
CLARA FastAPI
    │
    ├──► BGE-M3 / Ollama ──► Qdrant
    │                            │
    │                            ▼
    │                     Top-K Legal Chunks
    │                            │
    └────────► AMALIA-9B ◄───────┘
                    │
                    ▼
              Grounded Answer
```

---

## Technology Stack

| Technology | Purpose |
| --- | --- |
| **Python** | Core application and RAG pipeline |
| **FastAPI** | Native and OpenAI-compatible API |
| **BGE-M3** | Multilingual legal embeddings |
| **Qdrant** | Vector storage and retrieval |
| **AMALIA-9B** | European Portuguese answer generation |
| **Ollama** | Local model runtime |
| **Open WebUI** | Conversational user interface |
| **Docker Compose** | Application orchestration |

---

## Legal Corpus

The current corpus includes Portuguese tax and fiscal legislation:

- CIMI — Código do Imposto Municipal sobre Imóveis
- CIRC — Código do Imposto sobre o Rendimento das Pessoas Coletivas
- CIRS — Código do Imposto sobre o Rendimento das Pessoas Singulares
- CIS — Código do Imposto do Selo
- CIVA — Código do Imposto sobre o Valor Acrescentado
- EBF — Estatuto dos Benefícios Fiscais
- LGT — Lei Geral Tributária
- OE2026 — Orçamento do Estado 2026
- RGIT — Regime Geral das Infrações Tributárias
- RITI — Regime do IVA nas Transações Intracomunitárias

---

## Repository Structure

```text
clara-rag-pt/
├── app/                 # FastAPI application
├── data/
│   ├── raw/             # Source legal PDFs
│   ├── processed/       # Extracted, parsed and chunked documents
│   ├── embeddings/      # Generated BGE-M3 embeddings
│   └── evaluation/      # Retrieval evaluation results
├── notebooks/           # Ingestion, indexing and evaluation pipeline
├── src/                 # Core RAG implementation
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── test_rag.py
```

---

## Getting Started

### Requirements

- Git
- Python 3.14
- Docker / Docker Desktop
- Ollama
- Jupyter Notebook or JupyterLab
- NVIDIA GPU recommended for local AMALIA-9B inference

### Clone

```bash
git clone https://github.com/ruialexrib/clara-rag-pt.git
cd clara-rag-pt
```

### Models

```bash
ollama pull bge-m3
ollama run hf.co/ruialexrib/AMALIA-9B-0626-SFT-GGUF:Q3_K_M
```

### Environment

Create `.env` in the repository root:

```env
WEBUI_SECRET_KEY=your-secret-key
```

Generate a random value with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Do not commit `.env`.

---

## Build the Legal Corpus

Run the notebooks sequentially:

```text
01_pdf_extraction.ipynb
        ↓
02_document_parsing.ipynb
        ↓
03_document_chunking.ipynb
        ↓
04_embedding_generation.ipynb
        ↓
05_qdrant_indexing.ipynb
        ↓
06_vector_search.ipynb
        ↓
07_retrieval_evaluation.ipynb
        ↓
08_rag_generation_amalia.ipynb
```

The first five notebooks rebuild the searchable corpus. Notebooks `06`–`08` test semantic retrieval, evaluate retrieval quality, and validate the complete RAG pipeline.

---

## Run CLARA

Start Qdrant before indexing if the collection does not yet exist:

```bash
docker compose up -d qdrant
```

After running `05_qdrant_indexing.ipynb`, start the complete stack:

```bash
docker compose up -d --build
```

The stack contains `clara-qdrant`, `clara-api`, and `clara-open-webui`.

### Local Services

| Service | Address |
| --- | --- |
| Open WebUI | `localhost:3000` |
| FastAPI Swagger | `localhost:8000/docs` |
| API health check | `localhost:8000/health` |
| Qdrant | `localhost:6333` |

---

## API

Native CLARA endpoint:

```text
POST /chat
```

Example request:

```json
{
  "question": "Como são tributados os rendimentos prediais?",
  "top_k": 5,
  "document_id": null
}
```

CLARA also exposes an OpenAI-compatible interface:

```text
GET  /v1/models
POST /v1/chat/completions
```

Streaming responses are supported.

---

## Configuration

| Setting | Value |
| --- | --- |
| Embedding model | `bge-m3` |
| LLM | `hf.co/ruialexrib/AMALIA-9B-0626-SFT-GGUF:Q3_K_M` |
| Qdrant collection | `clara_bge_m3` |
| Top-K | `5` |
| Temperature | `0.1` |

Ollama runs on the host and is accessed from the API container through `host.docker.internal:11434`.

---

## Grounding

CLARA is instructed to answer exclusively from the legal context retrieved from Qdrant. Retrieved chunks preserve source document, article, article title, and page-range metadata.

If the retrieved context is insufficient, the model is instructed not to complete the answer using external knowledge.

---

## Stop the Application

```bash
docker compose down
```

Qdrant and Open WebUI data are stored in persistent Docker volumes. Avoid `docker compose down -v` unless you intentionally want to delete those volumes.

---

## Disclaimer

CLARA is an experimental system developed for research, educational, and technical demonstration purposes. Generated answers may contain errors or omissions and do not constitute legal advice.

Legally relevant information should always be verified against the official and currently applicable version of the legislation.

---

## Author

**Rui Ribeiro**

---

## License

This project is licensed under the [MIT License](LICENSE).
