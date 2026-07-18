"""Create this service's .env without printing or hard-coding credentials."""

from __future__ import annotations

import secrets
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "ChinaTravel-main" / ".env"
TARGET = ROOT / ".env"


def value(source: dict[str, str | None], name: str, default: str = "") -> str:
    return str(source.get(name) or default).strip()


def main() -> None:
    source = dotenv_values(SOURCE) if SOURCE.exists() else {}
    existing = dotenv_values(TARGET) if TARGET.exists() else {}
    service_key = value(existing, "SERVICE_API_KEY") or secrets.token_urlsafe(32)
    values = {
        "APP_HOST": "127.0.0.1",
        "APP_PORT": "8100",
        "REQUIRE_SERVICE_API_KEY": value(existing, "REQUIRE_SERVICE_API_KEY", "false"),
        "SERVICE_API_KEY": service_key,
        "MAX_UPLOAD_MB": "10",
        "LLM_API_URL": value(existing, "LLM_API_URL", "https://api.deepseek.com/v1"),
        "LLM_API_KEY": value(existing, "LLM_API_KEY") or value(source, "OPENAI_API_KEY"),
        "LLM_MODEL": value(existing, "LLM_MODEL", "deepseek-v4-flash"),
        "LLM_TIMEOUT_SECONDS": "45",
        "BGE_M3_API_URL": value(existing, "BGE_M3_API_URL") or value(source, "BGE_M3_API_URL"),
        "BGE_M3_API_KEY": value(existing, "BGE_M3_API_KEY") or value(source, "BGE_M3_API_KEY"),
        "BGE_M3_MODEL": value(existing, "BGE_M3_MODEL") or value(source, "BGE_M3_MODEL", "BAAI/bge-m3"),
        "BGE_M3_VECTOR_SIZE": value(existing, "BGE_M3_VECTOR_SIZE") or value(source, "BGE_M3_VECTOR_SIZE", "1024"),
        "BGE_M3_TIMEOUT_SECONDS": "60",
        "QDRANT_URL": value(existing, "QDRANT_URL") or value(source, "QDRANT_URL", "http://127.0.0.1:6333"),
        "QDRANT_API_KEY": value(existing, "QDRANT_API_KEY") or value(source, "QDRANT_API_KEY"),
        "QDRANT_COLLECTION": value(existing, "QDRANT_COLLECTION") or value(source, "QDRANT_COLLECTION", "chinatravel_safety_knowledge"),
        "RAG_MAX_ROUNDS": "2",
        "RAG_MAX_SUBQUERIES": "2",
        "RAG_RETRIEVE_TOP_K": "8",
        "RAG_CONTEXT_TOP_K": "4",
        "RAG_RETRIEVE_THRESHOLD": value(existing, "RAG_RETRIEVE_THRESHOLD") or value(source, "RAG_SCORE_THRESHOLD", "0.40"),
        "RAG_ACCEPT_THRESHOLD": "0.55",
        "RAG_REQUEST_TIMEOUT_SECONDS": "20",
        "DEEPSEEK_CACHE_HIT_COST_PER_1M_TOKENS_CNY": value(existing, "DEEPSEEK_CACHE_HIT_COST_PER_1M_TOKENS_CNY", "0.02"),
        "DEEPSEEK_CACHE_MISS_COST_PER_1M_TOKENS_CNY": value(existing, "DEEPSEEK_CACHE_MISS_COST_PER_1M_TOKENS_CNY", "1"),
        "DEEPSEEK_OUTPUT_COST_PER_1M_TOKENS_CNY": value(existing, "DEEPSEEK_OUTPUT_COST_PER_1M_TOKENS_CNY", "2"),
        "EMBEDDING_COST_PER_1M_TOKENS_CNY": value(existing, "EMBEDDING_COST_PER_1M_TOKENS_CNY", "0"),
        "EMBEDDING_PRICING_CONFIGURED": value(existing, "EMBEDDING_PRICING_CONFIGURED", "true"),
        "EMBEDDING_PRICING_SOURCE": value(existing, "EMBEDDING_PRICING_SOURCE", "SiliconFlow 官方价格"),
        "RAGAAI_ENABLED": value(existing, "RAGAAI_ENABLED", "false"),
        "RAGAAI_CATALYST_ACCESS_KEY": value(existing, "RAGAAI_CATALYST_ACCESS_KEY"),
        "RAGAAI_CATALYST_SECRET_KEY": value(existing, "RAGAAI_CATALYST_SECRET_KEY"),
        "RAGAAI_CATALYST_BASE_URL": value(existing, "RAGAAI_CATALYST_BASE_URL", "https://catalyst.raga.ai/api"),
        "RAGAAI_PROJECT_NAME": value(existing, "RAGAAI_PROJECT_NAME", "ChinaTravel-AgenticRAG"),
        "RAGAAI_DATASET_NAME": value(existing, "RAGAAI_DATASET_NAME", "chinatravel-rag-traces"),
    }
    TARGET.write_text(
        "\n".join(f"{key}={item}" for key, item in values.items()) + "\n",
        encoding="utf-8",
    )
    missing = [
        name
        for name in ("LLM_API_KEY", "BGE_M3_API_URL", "BGE_M3_API_KEY")
        if not values[name]
    ]
    print(f"Created {TARGET}; no credential values were printed.")
    if missing:
        print("Still required: " + ", ".join(missing))
    else:
        print("Reused the LLM, BGE-M3 and Qdrant settings from ChinaTravel.")


if __name__ == "__main__":
    main()
