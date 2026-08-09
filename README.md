# CLARA — Portuguese Legislation RAG Assistant

**CLARA** (*Consulta de Legislação Assistida por Recuperação Aumentada*) is a local Retrieval-Augmented Generation (RAG) system for natural-language consultation of Portuguese legislation.

The pipeline uses **BGE-M3** for embeddings, **Qdrant** for vector retrieval, **AMALIA-9B** for grounded answer generation, **FastAPI** for the API, and **Open WebUI** for the chat interface. The system runs locally and does not require external LLM APIs.

## Architecture

```text
Open WebUI
    │
    ▼
CLARA FastAPI
    │
    ├──► BGE-M3 (Ollama) ──► Qdrant
    │                            │
    │                            ▼
    │                     Top-K legal chunks
    │                            │
    └────────► AMALIA-9B ◄───────┘
                    │
                    ▼
             Grounded answer
```

## Legal Corpus

The current corpus includes:

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

## Requirements

To reproduce the project you need:

- Git
- Python 3.14
- Docker / Docker Desktop
- Ollama
- Jupyter Notebook or JupyterLab
- An NVIDIA GPU is recommended for local AMALIA-9B inference

## 1. Clone the Repository

```bash
git clone https://github.com/ruialexrib/clara-rag-pt.git
cd clara-rag-pt
```

## 2. Install the Ollama Models

CLARA expects Ollama to run on the host machine.

Install the embedding model:

```bash
ollama pull bge-m3
```

Install the quantised AMALIA-9B model:

```bash
ollama run hf.co/ruialexrib/AMALIA-9B-0626-SFT-GGUF:Q3_K_M
```

Check that both models are available:

```bash
ollama list
```

## 3. Configure the Environment

Create a `.env` file in the repository root:

```env
WEBUI_SECRET_KEY=your-secret-key
```

A random key can be generated with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Do not commit the `.env` file.

## 4. Build the Legal Corpus

The notebooks implement the complete ingestion and evaluation pipeline and should be executed sequentially:

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

The first five notebooks are the essential steps for rebuilding the searchable corpus:

1. extract text from the legal PDFs;
2. identify articles and document structure;
3. generate article-aware chunks;
4. generate BGE-M3 embeddings;
5. create and populate the Qdrant collection.

Notebooks 06–08 are used to test semantic retrieval, evaluate retrieval quality, and test the complete RAG pipeline.

> Qdrant must be running before executing the indexing notebook.

## 5. Start the Docker Services

Start Qdrant first if the collection still needs to be created:

```bash
docker compose up -d qdrant
```

Run `05_qdrant_indexing.ipynb` to create and populate the `clara_bge_m3` collection.

Then start the complete application:

```bash
docker compose up -d --build
```

The stack contains:

```text
clara-qdrant
clara-api
clara-open-webui
```

## 6. Access CLARA

Open WebUI:

```text
http://localhost:3000
```

FastAPI Swagger:

```text
http://localhost:8000/docs
```

API health check:

```text
http://localhost:8000/health
```

Qdrant:

```text
http://localhost:6333
```

## API

The native CLARA endpoint is:

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

CLARA also exposes an OpenAI-compatible interface used by Open WebUI:

```text
GET  /v1/models
POST /v1/chat/completions
```

Streaming responses are supported.

## Main Configuration

The default runtime configuration is:

```text
Embedding model:     bge-m3
LLM:                 hf.co/ruialexrib/AMALIA-9B-0626-SFT-GGUF:Q3_K_M
Qdrant collection:   clara_bge_m3
Top-K:               5
Temperature:         0.1
```

Ollama runs on the host and is accessed from the CLARA API container through `host.docker.internal:11434`.

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

## Grounding

CLARA is instructed to answer exclusively from the legal context retrieved from Qdrant. Retrieved chunks preserve metadata such as the source document, article, article title and page range.

If the retrieved context is insufficient, the model is instructed not to complete the answer using external knowledge.

## Stopping the Application

```bash
docker compose down
```

Qdrant and Open WebUI data are stored in persistent Docker volumes. Avoid `docker compose down -v` unless you intentionally want to delete those volumes.

## Disclaimer

CLARA is an experimental system developed for research, educational and technical demonstration purposes. Generated answers may contain errors or omissions and do not constitute legal advice.

Legally relevant information should always be verified against the official and currently applicable version of the legislation.

## Author

**Rui Ribeiro**

## License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.
