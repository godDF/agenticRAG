from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agents import RagAgents
from app.clients import EmbeddingClient, LlmClient
from app.config import settings
from app.documents import DocumentService
from app.local_trace import trace_store
from app.ingestion import IngestionReporter, ingestion_jobs
from app.observability import catalyst_tracing
from app.query_jobs import query_jobs
from app.registry import DocumentRegistry
from app.retrieval import VectorStore
from app.schemas import (
    DocumentItem,
    IngestionJob,
    IngestionJobCreated,
    QueryJob,
    QueryJobCreated,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.service import AgenticRagService

logger = logging.getLogger(__name__)

registry = DocumentRegistry(settings.registry_path)
embedding = EmbeddingClient(settings)
llm = LlmClient(settings)
vectors = VectorStore(settings, embedding)
agents = RagAgents(settings, llm)
rag_service = AgenticRagService(settings, agents, vectors)
document_service = DocumentService(settings, registry, vectors)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_directories()
    await asyncio.to_thread(registry.initialize)
    catalyst_tracing.initialize(settings)
    yield


app = FastAPI(
    title="ChinaTravel Agentic RAG",
    version="0.1.0",
    description="Agentic retrieval, document ingestion and observable grounded answers.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=settings.data_dir.parent / "frontend"), name="static")


def require_service_key(authorization: Annotated[str | None, Header()] = None) -> None:
    if not settings.require_service_api_key:
        return
    if not settings.service_api_key:
        raise HTTPException(503, "服务端尚未配置 SERVICE_API_KEY")
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if supplied != settings.service_api_key:
        raise HTTPException(401, "服务访问密钥无效")


@app.get("/")
async def root():
    return {
        "service": "ChinaTravel Agentic RAG",
        "version": "0.1.0",
        "docs": "/docs",
        "workbench": "/ui",
        "port": settings.port,
    }


@app.get("/ui", include_in_schema=False)
async def knowledge_workbench():
    return FileResponse(settings.data_dir.parent / "frontend" / "index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "python_service": "agenticRAG",
        "ragaai_enabled": catalyst_tracing.enabled,
        "ragaai_error": catalyst_tracing.error,
        "auth_required": settings.require_service_api_key,
        "llm_model": settings.llm_model,
        "pricing": {
            "currency": "CNY",
            "per_million_tokens": {
                "cache_hit_input": settings.llm_cache_hit_cost_per_million_cny,
                "cache_miss_input": settings.llm_cache_miss_cost_per_million_cny,
                "output": settings.llm_output_cost_per_million_cny,
            },
        },
        "embedding_model": settings.embedding_model,
        "embedding_pricing": {
            "currency": "CNY",
            "per_million_tokens": settings.embedding_cost_per_million_cny,
            "configured": settings.embedding_pricing_configured,
            "source": settings.embedding_pricing_source,
            "free": (
                settings.embedding_pricing_configured
                and settings.embedding_cost_per_million_cny == 0
            ),
        },
    }


@app.get("/ready")
async def ready():
    ok, detail = await vectors.ready()
    if not ok:
        return JSONResponse(status_code=503, content={"status": "not_ready", "qdrant": detail})
    return {"status": "ready", "qdrant": "ok"}


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_service_key)],
)
async def query_knowledge(request: QueryRequest):
    try:
        settings.validate_query_runtime()
        with catalyst_tracing.request_trace():
            return await asyncio.wait_for(
                rag_service.answer(
                    query=request.query,
                    category=request.category,
                    session_id=request.session_id,
                    request_id=request.request_id,
                ),
                timeout=settings.request_timeout_seconds,
            )
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "Agentic RAG 查询超时") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Agentic RAG query failed")
        raise HTTPException(503, f"Agentic RAG 暂不可用：{type(exc).__name__}") from exc


async def _run_query_job(job_id: str, request: QueryRequest) -> None:
    await query_jobs.start(job_id)

    async def publish_event(event: dict) -> None:
        await query_jobs.publish(job_id, event)

    try:
        with catalyst_tracing.request_trace():
            result = await asyncio.wait_for(
                rag_service.answer(
                    query=request.query,
                    category=request.category,
                    session_id=request.session_id,
                    request_id=job_id,
                    event_sink=publish_event,
                ),
                timeout=settings.request_timeout_seconds,
            )
        await query_jobs.complete(job_id, result)
    except asyncio.TimeoutError:
        await query_jobs.fail(
            job_id,
            error_type="query_timeout",
            message=f"Agentic RAG 查询超过 {settings.request_timeout_seconds:g} 秒",
        )
    except Exception as exc:
        logger.exception("Background Agentic RAG query failed: job_id=%s", job_id)
        await query_jobs.fail(
            job_id,
            error_type=type(exc).__name__,
            message="Agentic RAG 查询失败，请检查 8100 服务日志",
        )


@app.post(
    "/api/v1/query-jobs",
    response_model=QueryJobCreated,
    status_code=202,
    dependencies=[Depends(require_service_key)],
)
async def create_query_job(request: QueryRequest):
    try:
        settings.validate_query_runtime()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    job = await query_jobs.create()
    query_jobs.schedule(_run_query_job(job["job_id"], request))
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "poll_url": f"/api/v1/query-jobs/{job['job_id']}",
    }


@app.get(
    "/api/v1/query-jobs/{job_id}",
    response_model=QueryJob,
    dependencies=[Depends(require_service_key)],
)
async def get_query_job(job_id: str):
    job = await query_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "RAG 查询任务不存在或已过期")
    return job


@app.post(
    "/api/v1/documents",
    response_model=UploadResponse,
    dependencies=[Depends(require_service_key)],
)
async def upload_document(
    file: Annotated[UploadFile, File()],
    category: Annotated[str, Form()],
    updated_at: Annotated[str, Form(min_length=1, max_length=32)],
    title: Annotated[str | None, Form(max_length=200)] = None,
    source_name: Annotated[str | None, Form(max_length=200)] = None,
    source_url: Annotated[str, Form(max_length=1000)] = "",
):
    try:
        settings.validate_index_runtime()
        data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        record = await document_service.upload(
            filename=file.filename or "upload",
            data=data,
            title=title,
            category=category,
            source_name=source_name,
            source_url=source_url,
            updated_at=updated_at,
        )
        return {"document": record, "message": "文档已解析、切分并写入向量库"}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Document indexing failed")
        raise HTTPException(503, f"文档索引失败：{type(exc).__name__}") from exc
    finally:
        await file.close()


async def _run_ingestion_job(job_id: str, payload: dict) -> None:
    await ingestion_jobs.start(job_id)
    reporter = IngestionReporter(ingestion_jobs, job_id)
    coordinator_event = await reporter.start(
        kind="agent",
        name="ingestion_coordinator",
        label="知识库摄取流程开始",
        progress=7,
        details={
            "mode": "deterministic_workflow",
            "llm_used": False,
            "note": "上传索引不需要 LLM，避免额外 Token 与不确定性",
        },
    )
    try:
        record = await document_service.upload(**payload, reporter=reporter)
        await reporter.finish(
            coordinator_event,
            progress=99,
            details={"document_id": record["document_id"], "status": "indexed"},
        )
        await ingestion_jobs.complete(job_id, record)
    except Exception as exc:
        logger.exception("Background document ingestion failed: job_id=%s", job_id)
        await ingestion_jobs.fail(job_id, exc)


@app.post(
    "/api/v1/document-jobs",
    response_model=IngestionJobCreated,
    status_code=202,
    dependencies=[Depends(require_service_key)],
)
async def create_document_job(
    file: Annotated[UploadFile, File()],
    category: Annotated[str, Form()],
    updated_at: Annotated[str, Form(min_length=1, max_length=32)],
    title: Annotated[str | None, Form(max_length=200)] = None,
    source_name: Annotated[str | None, Form(max_length=200)] = None,
    source_url: Annotated[str, Form(max_length=1000)] = "",
):
    try:
        settings.validate_index_runtime()
        data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"文件不能超过 {settings.max_upload_mb} MB")
        job = await ingestion_jobs.create(file.filename or "upload", len(data))
        ingestion_jobs.schedule(
            _run_ingestion_job(
                job["job_id"],
                {
                    "filename": file.filename or "upload",
                    "data": data,
                    "title": title,
                    "category": category,
                    "source_name": source_name,
                    "source_url": source_url,
                    "updated_at": updated_at,
                },
            )
        )
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "poll_url": f"/api/v1/document-jobs/{job['job_id']}",
        }
    finally:
        await file.close()


@app.get(
    "/api/v1/document-jobs/{job_id}",
    response_model=IngestionJob,
    dependencies=[Depends(require_service_key)],
)
async def get_document_job(job_id: str):
    job = await ingestion_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "上传任务不存在或已过期")
    return job


@app.get(
    "/api/v1/documents",
    response_model=list[DocumentItem],
    dependencies=[Depends(require_service_key)],
)
async def list_documents():
    return await document_service.list()


@app.delete(
    "/api/v1/documents/{document_id}",
    response_model=DocumentItem,
    dependencies=[Depends(require_service_key)],
)
async def delete_document(document_id: str):
    try:
        return await document_service.delete(document_id)
    except KeyError as exc:
        raise HTTPException(404, "文档不存在") from exc


@app.post(
    "/api/v1/documents/{document_id}/reindex",
    response_model=DocumentItem,
    dependencies=[Depends(require_service_key)],
)
async def reindex_document(document_id: str):
    try:
        settings.validate_index_runtime()
        return await document_service.reindex(document_id)
    except KeyError as exc:
        raise HTTPException(404, "文档不存在") from exc
    except Exception as exc:
        raise HTTPException(503, f"重新索引失败：{type(exc).__name__}") from exc


@app.get(
    "/api/v1/traces/{trace_id}",
    dependencies=[Depends(require_service_key)],
)
async def get_trace(trace_id: str):
    item = await trace_store.get(trace_id)
    if item is None:
        raise HTTPException(404, "Trace 不存在或已过期")
    return item
