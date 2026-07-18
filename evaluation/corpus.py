from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from qdrant_client import models

from app.chunk_identity import build_chunk_uid, normalize_chunk_text
from app.clients import EmbeddingClient
from app.config import RAG_CATEGORIES, Settings, settings
from app.documents import chunk_markdown
from app.retrieval import VectorStore
from evaluation.common import MANIFESTS_DIR, ensure_evaluation_directories, sha256_file, write_json


DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "ChinaTravel-main" / "kb"
DEFAULT_EVAL_COLLECTION = "chinatravel_safety_eval_v1"


def parse_markdown(path: Path, kb_dir: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8-sig")
    metadata: dict[str, Any] = {}
    body = raw.strip()
    if raw.lstrip().startswith("---"):
        try:
            _, header, body = raw.lstrip().split("---", 2)
            metadata = yaml.safe_load(header) or {}
        except (ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"Markdown YAML 无法解析: {path}") from exc

    folder_category = path.parent.name
    category = str(metadata.get("category") or folder_category).strip()
    if category not in RAG_CATEGORIES:
        raise ValueError(f"未知知识分类 {category}: {path}")
    if category != folder_category:
        raise ValueError(
            f"目录分类与YAML分类不一致: {path} ({folder_category} != {category})"
        )
    title = str(metadata.get("title") or path.stem).strip()
    source_url = str(metadata.get("source_url") or "").strip()
    relative_path = path.relative_to(kb_dir).as_posix()
    document_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:32]
    return {
        "document_id": document_id,
        "title": title,
        "category": category,
        "source_name": str(metadata.get("source_name") or "").strip(),
        "source_url": source_url,
        "updated_at": str(metadata.get("updated_at") or "").strip(),
        "file_path": f"kb/{relative_path}",
        "body": body.strip(),
        "file_sha256": sha256_file(path),
    }


def build_corpus_manifest(kb_dir: Path, version: str) -> dict[str, Any]:
    kb_dir = kb_dir.resolve()
    if not kb_dir.exists():
        raise FileNotFoundError(f"知识库目录不存在: {kb_dir}")
    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for path in sorted(kb_dir.glob("**/*.md")):
        document = parse_markdown(path, kb_dir)
        body = document.pop("body")
        document_chunks = []
        for index, content in enumerate(chunk_markdown(body)):
            chunk_uid = build_chunk_uid(
                category=document["category"],
                title=document["title"],
                source_url=document["source_url"],
                content=content,
            )
            chunk = {
                **{key: value for key, value in document.items() if key != "file_sha256"},
                "chunk_uid": chunk_uid,
                "chunk_index": index,
                "content": content,
                "content_hash": hashlib.sha256(
                    normalize_chunk_text(content).encode("utf-8")
                ).hexdigest(),
                "untrusted_content": True,
            }
            chunks.append(chunk)
            document_chunks.append(chunk_uid)
        documents.append({**document, "chunk_uids": document_chunks})

    if not documents or not chunks:
        raise ValueError("知识库没有可用于评测的 Markdown Chunk")
    corpus_hash = hashlib.sha256(
        "\n".join(sorted(chunk["chunk_uid"] for chunk in chunks)).encode("utf-8")
    ).hexdigest()
    return {
        "dataset_version": version,
        "kb_dir": str(kb_dir),
        "corpus_hash": corpus_hash,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "documents": documents,
        "chunks": chunks,
    }


async def index_manifest(
    manifest: dict[str, Any],
    *,
    collection_name: str = DEFAULT_EVAL_COLLECTION,
    recreate: bool = True,
    runtime_settings: Settings = settings,
) -> dict[str, Any]:
    config = replace(runtime_settings, qdrant_collection=collection_name)
    config.validate_index_runtime()
    embedding = EmbeddingClient(config)
    vectors = VectorStore(config, embedding)
    exists = await asyncio.to_thread(vectors.client.collection_exists, collection_name)
    if exists and recreate:
        await asyncio.to_thread(vectors.client.delete_collection, collection_name)
    await vectors.ensure_collection()

    chunks = list(manifest["chunks"])
    texts = [chunk["content"] for chunk in chunks]
    embeddings: list[list[float]] = []
    embedding_tokens = 0
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        batch_vectors, usage = await embedding.embed_with_usage(texts[start : start + batch_size])
        embeddings.extend(batch_vectors)
        embedding_tokens += int(usage.get("total_tokens", 0))

    points = []
    for chunk, vector in zip(chunks, embeddings):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_uid"]))
        points.append(models.PointStruct(id=point_id, vector=vector, payload=chunk))
    await asyncio.to_thread(
        vectors.client.upsert,
        collection_name=collection_name,
        points=points,
        wait=True,
    )
    return {
        "collection": collection_name,
        "point_count": len(points),
        "embedding_tokens": embedding_tokens,
        "corpus_hash": manifest["corpus_hash"],
    }


def save_manifest(manifest: dict[str, Any], version: str) -> Path:
    ensure_evaluation_directories()
    path = MANIFESTS_DIR / f"kb_{version}.json"
    write_json(path, manifest)
    return path

