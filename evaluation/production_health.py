from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from app.chunk_identity import chunk_uid_from_payload, normalize_chunk_text
from app.config import RAG_CATEGORIES, settings
from evaluation.common import MANIFESTS_DIR, RUNS_DIR, read_json, write_csv, write_json


def _scroll_all(client: QdrantClient, collection: str) -> list[dict[str, Any]]:
    offset = None
    rows = []
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = dict(point.payload or {})
            rows.append(
                {
                    "point_id": str(point.id),
                    "chunk_uid": chunk_uid_from_payload(payload),
                    **payload,
                }
            )
        if offset is None:
            break
    return rows


def audit_production(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=15,
        check_compatibility=False,
        trust_env=False,
    )
    collection = settings.qdrant_collection
    if not client.collection_exists(collection):
        raise RuntimeError(f"生产Collection不存在: {collection}")
    points = _scroll_all(client, collection)
    clean_uids = {item["chunk_uid"] for item in manifest["chunks"]}
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_content_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    issues = []
    for point in points:
        uid = str(point.get("chunk_uid") or "")
        content = str(point.get("content") or "")
        category = str(point.get("category") or "")
        by_uid[uid].append(point)
        content_hash = hashlib.sha256(normalize_chunk_text(content).encode("utf-8")).hexdigest()
        by_content_hash[content_hash].append(point)
        if category not in RAG_CATEGORIES:
            issues.append(
                {
                    "point_id": point["point_id"],
                    "issue": "invalid_category",
                    "category": category,
                    "title": point.get("title", ""),
                    "detail": "Payload category不属于六类知识",
                }
            )
        declared = re.search(
            r"(?:^|\n)category\s*:\s*([a-z_]+)", content, flags=re.IGNORECASE
        )
        if declared and declared.group(1) != category:
            issues.append(
                {
                    "point_id": point["point_id"],
                    "issue": "category_mismatch",
                    "category": category,
                    "title": point.get("title", ""),
                    "detail": f"内容声明{declared.group(1)}，Payload为{category}",
                }
            )
        missing = [
            field
            for field in ("title", "category", "content", "source_url")
            if not str(point.get(field) or "").strip()
        ]
        if missing:
            issues.append(
                {
                    "point_id": point["point_id"],
                    "issue": "missing_metadata",
                    "category": category,
                    "title": point.get("title", ""),
                    "detail": "缺少字段: " + ", ".join(missing),
                }
            )

    duplicate_uid_groups = [items for items in by_uid.values() if len(items) > 1]
    duplicate_content_groups = [items for items in by_content_hash.values() if len(items) > 1]
    for group in duplicate_content_groups:
        for point in group[1:]:
            issues.append(
                {
                    "point_id": point["point_id"],
                    "issue": "duplicate_content",
                    "category": point.get("category", ""),
                    "title": point.get("title", ""),
                    "detail": f"与{group[0]['point_id']}内容重复",
                }
            )
    production_uids = set(by_uid)
    report = {
        "collection": collection,
        "point_count": len(points),
        "category_counts": dict(Counter(str(item.get("category") or "") for item in points)),
        "clean_manifest_chunk_count": len(clean_uids),
        "clean_chunk_coverage": (
            len(clean_uids & production_uids) / len(clean_uids) if clean_uids else None
        ),
        "clean_chunks_present": len(clean_uids & production_uids),
        "clean_chunks_missing": len(clean_uids - production_uids),
        "extra_production_chunks": len(production_uids - clean_uids),
        "duplicate_uid_groups": len(duplicate_uid_groups),
        "duplicate_content_groups": len(duplicate_content_groups),
        "issue_count": len(issues),
        "issue_counts": dict(Counter(item["issue"] for item in issues)),
    }
    return report, issues


def save_audit(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    report, issues = audit_production(manifest)
    write_json(run_dir / "production_health.json", report)
    write_csv(run_dir / "production_health_issues.csv", issues)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="检查当前生产Qdrant知识库健康度")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    manifest = read_json(MANIFESTS_DIR / f"kb_{args.version}.json")
    output = RUNS_DIR / args.run_id if args.run_id else RUNS_DIR / "production-health"
    output.mkdir(parents=True, exist_ok=True)
    report = save_audit(output, manifest)
    print(f"生产知识库审计完成: {output}")
    print(report)


if __name__ == "__main__":
    main()

