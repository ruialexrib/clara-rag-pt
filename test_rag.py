from src.rag import ask_clara


# =========================================================
# Test question
# =========================================================

question = "Como são tributados os rendimentos prediais?"

print(f"Pergunta: {question}")
print("-" * 80)


# =========================================================
# Execute the complete CLARA RAG pipeline
# =========================================================

result = ask_clara(
    question=question
)


# =========================================================
# Generated answer
# =========================================================

print("\nResposta:")
print(result["answer"])


# =========================================================
# Retrieved legal sources
# =========================================================

print("\nFontes:")

for source in result["sources"]:
    print(
        f"- {source['document_id']} | "
        f"{source['article']} | "
        f"{source['article_title']} | "
        f"score={source['score']:.4f}"
    )


# =========================================================
# Execution times
# =========================================================

print("\nTempos:")

print(
    f"Retrieval: {result['retrieval_seconds']} s"
)

print(
    f"Geração: {result['generation_seconds']} s"
)

print(
    f"Total: {result['total_seconds']} s"
)