from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

RAG_CATEGORIES = {
    "child_ticket",
    "elderly_ticket",
    "student_ticket",
    "flight_safety",
    "highspeed_rail_safety",
    "attraction_notice",
}


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("APP_HOST", "127.0.0.1")
    port: int = int(os.getenv("APP_PORT", "8100"))
    require_service_api_key: bool = _bool("REQUIRE_SERVICE_API_KEY", False)
    service_api_key: str = os.getenv("SERVICE_API_KEY", "")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))

    llm_api_url: str = os.getenv("LLM_API_URL", "https://api.deepseek.com/v1").rstrip("/")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-v4-flash")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))

    embedding_api_url: str = os.getenv("BGE_M3_API_URL", "").rstrip("/")
    embedding_api_key: str = os.getenv("BGE_M3_API_KEY", "")
    embedding_model: str = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
    embedding_vector_size: int = int(os.getenv("BGE_M3_VECTOR_SIZE", "1024"))
    embedding_timeout_seconds: float = float(os.getenv("BGE_M3_TIMEOUT_SECONDS", "60"))

    qdrant_url: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "chinatravel_safety_knowledge")

    max_rounds: int = min(2, max(1, int(os.getenv("RAG_MAX_ROUNDS", "2"))))
    max_subqueries: int = min(2, max(1, int(os.getenv("RAG_MAX_SUBQUERIES", "2"))))
    retrieve_top_k: int = int(os.getenv("RAG_RETRIEVE_TOP_K", "8"))
    context_top_k: int = int(os.getenv("RAG_CONTEXT_TOP_K", "4"))
    retrieve_threshold: float = float(os.getenv("RAG_RETRIEVE_THRESHOLD", "0.40"))
    accept_threshold: float = float(os.getenv("RAG_ACCEPT_THRESHOLD", "0.55"))
    request_timeout_seconds: float = float(os.getenv("RAG_REQUEST_TIMEOUT_SECONDS", "30"))
    llm_cache_hit_cost_per_million_cny: float = float(
        os.getenv("DEEPSEEK_CACHE_HIT_COST_PER_1M_TOKENS_CNY", "0.02")
    )
    llm_cache_miss_cost_per_million_cny: float = float(
        os.getenv("DEEPSEEK_CACHE_MISS_COST_PER_1M_TOKENS_CNY", "1")
    )
    llm_output_cost_per_million_cny: float = float(
        os.getenv("DEEPSEEK_OUTPUT_COST_PER_1M_TOKENS_CNY", "2")
    )
    embedding_cost_per_million_cny: float = float(
        os.getenv("EMBEDDING_COST_PER_1M_TOKENS_CNY", "0")
    )
    embedding_pricing_configured: bool = _bool("EMBEDDING_PRICING_CONFIGURED", True)
    embedding_pricing_source: str = os.getenv(
        "EMBEDDING_PRICING_SOURCE", "SiliconFlow 官方价格"
    )

    raga_enabled: bool = _bool("RAGAAI_ENABLED", False)
    raga_access_key: str = os.getenv("RAGAAI_CATALYST_ACCESS_KEY", "")
    raga_secret_key: str = os.getenv("RAGAAI_CATALYST_SECRET_KEY", "")
    raga_base_url: str = os.getenv("RAGAAI_CATALYST_BASE_URL", "https://catalyst.raga.ai/api")
    raga_project_name: str = os.getenv("RAGAAI_PROJECT_NAME", "ChinaTravel-AgenticRAG")
    raga_dataset_name: str = os.getenv("RAGAAI_DATASET_NAME", "chinatravel-rag-traces")

    data_dir: Path = PROJECT_ROOT / "data"
    kb_dir: Path = PROJECT_ROOT / "data" / "kb"
    registry_path: Path = PROJECT_ROOT / "data" / "documents.sqlite3"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        for category in RAG_CATEGORIES:
            (self.kb_dir / category).mkdir(parents=True, exist_ok=True)

    def validate_query_runtime(self) -> None:
        missing = []
        if not self.llm_api_key:
            missing.append("LLM_API_KEY")
        if not self.embedding_api_url:
            missing.append("BGE_M3_API_URL")
        if not self.embedding_api_key:
            missing.append("BGE_M3_API_KEY")
        if missing:
            raise RuntimeError("缺少运行配置: " + ", ".join(missing))

    def validate_index_runtime(self) -> None:
        missing = []
        if not self.embedding_api_url:
            missing.append("BGE_M3_API_URL")
        if not self.embedding_api_key:
            missing.append("BGE_M3_API_KEY")
        if missing:
            raise RuntimeError("缺少运行配置: " + ", ".join(missing))


settings = Settings()
