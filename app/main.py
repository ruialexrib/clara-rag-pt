import json
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.rag import (
    ask_clara,
    stream_clara,
)

from src.config import (
    APP_NAME,
    APP_DESCRIPTION,
    APP_VERSION,
)


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)


# =========================================================
# Request and response schemas
# =========================================================

class ChatRequest(BaseModel):
    """
    Request model for the native CLARA chat endpoint.
    """

    question: str
    top_k: int = 5
    document_id: str | None = None


class ChatResponse(BaseModel):
    """
    Response model returned by the native CLARA chat endpoint.
    """

    question: str
    answer: str
    sources: list[dict]
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float


class OpenAIMessage(BaseModel):
    """
    Message structure compatible with the OpenAI chat format.
    """

    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    """
    Request structure for the OpenAI-compatible chat endpoint.
    """

    model: str
    messages: list[OpenAIMessage]
    stream: bool = False


# =========================================================
# Health endpoint
# =========================================================

@app.get("/health")
def health():
    """
    Return the current CLARA API status.

    Returns
    -------
    dict
        Basic service health information.
    """

    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
    }


# =========================================================
# OpenAI-compatible model discovery
# =========================================================

@app.get("/v1/models")
def list_models():
    """
    Expose CLARA as an OpenAI-compatible model.

    This endpoint allows clients such as Open WebUI
    to discover the CLARA model automatically.

    Returns
    -------
    dict
        OpenAI-compatible model list.
    """

    return {
        "object": "list",
        "data": [
            {
                "id": "clara",
                "object": "model",
                "created": int(
                    datetime.now().timestamp()
                ),
                "owned_by": "clara-rag",
            }
        ],
    }


# =========================================================
# Native CLARA chat endpoint
# =========================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    """
    Execute the complete CLARA RAG pipeline.

    This endpoint exposes the native CLARA response format,
    including the generated answer, retrieved legal sources,
    and execution times.

    Parameters
    ----------
    request : ChatRequest
        User question and optional retrieval configuration.

    Returns
    -------
    ChatResponse
        Complete CLARA RAG result.
    """

    try:
        result = ask_clara(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# =========================================================
# OpenAI-compatible chat completions endpoint
# =========================================================

@app.post("/v1/chat/completions")
def openai_chat_completions(
    request: OpenAIChatRequest,
):
    """
    Provide an OpenAI-compatible chat completions endpoint.

    This endpoint allows OpenAI-compatible clients, including
    Open WebUI, to interact with the CLARA RAG pipeline.

    Both standard and streaming responses are supported.

    Parameters
    ----------
    request : OpenAIChatRequest
        OpenAI-compatible chat request.

    Returns
    -------
    dict | StreamingResponse
        Standard OpenAI-compatible response or SSE stream.
    """

    try:

        # -------------------------------------------------
        # Validate the requested model
        # -------------------------------------------------

        if request.model != "clara":
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Modelo '{request.model}' "
                    "não encontrado."
                ),
            )

        # -------------------------------------------------
        # Extract the most recent user message
        # -------------------------------------------------

        user_messages = [
            message
            for message in request.messages
            if message.role == "user"
        ]

        if not user_messages:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Não foi encontrada nenhuma "
                    "mensagem do utilizador."
                ),
            )

        question = user_messages[-1].content

        if not question.strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    "A mensagem do utilizador "
                    "não pode estar vazia."
                ),
            )

        # =================================================
        # Streaming response
        # =================================================

        if request.stream:

            rag_result = stream_clara(
                question=question,
            )

            answer_stream = rag_result[
                "stream"
            ]

            def event_stream():
                """
                Convert the AMALIA stream into
                OpenAI-compatible Server-Sent Events.
                """

                created = int(
                    time.time()
                )

                # -----------------------------------------
                # Initial assistant role event
                # -----------------------------------------

                first_chunk = {
                    "id": "chatcmpl-clara",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "clara",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant"
                            },
                            "finish_reason": None,
                        }
                    ],
                }

                yield (
                    "data: "
                    + json.dumps(
                        first_chunk,
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

                # -----------------------------------------
                # Stream AMALIA-generated text fragments
                # -----------------------------------------

                for token in answer_stream:

                    content_chunk = {
                        "id": "chatcmpl-clara",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "clara",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": token
                                },
                                "finish_reason": None,
                            }
                        ],
                    }

                    yield (
                        "data: "
                        + json.dumps(
                            content_chunk,
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )

                # -----------------------------------------
                # Final completion event
                # -----------------------------------------

                final_chunk = {
                    "id": "chatcmpl-clara",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "clara",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }

                yield (
                    "data: "
                    + json.dumps(
                        final_chunk,
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

                # OpenAI-compatible end-of-stream marker.
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # =================================================
        # Non-streaming response
        # =================================================

        result = ask_clara(
            question=question,
        )

        return {
            "id": "chatcmpl-clara",
            "object": "chat.completion",
            "created": int(
                time.time()
            ),
            "model": "clara",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result[
                            "answer"
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )