from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.config import RAG_CATEGORIES


class QueryRequest(BaseModel):
    request_id: str | None = Field(default=None, max_length=80)
    session_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=2000)
    category: str

    @field_validator("category")
    @classmethod
    def valid_category(cls, value: str) -> str:
        if value not in RAG_CATEGORIES:
            raise ValueError("category 必须是六类知识之一")
        return value


class SourceItem(BaseModel):
    document_id: str | None = None
    title: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    updated_at: str | None = None
    score: float = 0.0


class QueryMeta(BaseModel):
    retrieval_rounds: int = 0
    max_retrieval_rounds: int = 0
    retrieved_chunks: int = 0
    accepted_chunks: int = 0
    rewritten: bool = False
    verified: bool = False
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_input_tokens: int = 0
    cache_miss_input_tokens: int = 0
    cache_usage_reported: bool = False
    estimated_cost_cny: float = 0.0
    cost_configured: bool = False
    pricing_model: str = ""
    pricing_currency: str = "CNY"


class TraceEvent(BaseModel):
    event_id: str = ""
    event_type: Literal["agent", "tool", "llm", "system"] = "agent"
    stage: str
    label: str | None = None
    status: Literal["running", "completed", "failed", "fallback"]
    progress: int = Field(default=0, ge=0, le=100)
    started_at: str | None = None
    completed_at: str | None = None
    latency_ms: int
    details: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    found: bool
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    trace_id: str
    meta: QueryMeta
    trace: list[TraceEvent] = Field(default_factory=list)


class QueryJobCreated(BaseModel):
    job_id: str
    status: Literal["queued"]
    progress: int = Field(ge=0, le=100)
    poll_url: str


class QueryJobError(BaseModel):
    type: str
    message: str


class QueryJob(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    current_stage: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    result: QueryResponse | None = None
    error: QueryJobError | None = None


class DocumentItem(BaseModel):
    document_id: str
    title: str
    category: str
    source_name: str
    source_url: str
    updated_at: str
    original_filename: str
    normalized_path: str
    content_hash: str
    chunk_count: int
    status: str
    created_at: str


class UploadResponse(BaseModel):
    document: DocumentItem
    message: str


class IngestionJobCreated(BaseModel):
    job_id: str
    status: str
    progress: int
    poll_url: str


class IngestionEvent(BaseModel):
    event_id: str
    kind: str
    name: str
    label: str
    status: str
    progress: int
    started_at: str
    completed_at: str | None = None
    latency_ms: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None


class IngestionJob(BaseModel):
    job_id: str
    status: str
    progress: int
    current_stage: str
    filename: str
    size_bytes: int
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    events: list[IngestionEvent] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    result: DocumentItem | None = None
    error: dict[str, Any] | None = None
