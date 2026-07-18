from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import RAG_CATEGORIES
from evaluation.common import DATASETS_DIR, MANIFESTS_DIR, read_json, read_jsonl


EXPECTED_PER_CATEGORY = {
    "single_hop": 8,
    "paraphrase": 4,
    "multi_context": 4,
    "unanswerable": 2,
    "noise_conflict": 2,
}


def validate(
    dataset: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    allow_pending: bool = False,
) -> list[str]:
    errors: list[str] = []
    known_chunks = {item["chunk_uid"] for item in manifest["chunks"]}
    ids = [str(item.get("case_id", "")) for item in dataset]
    if len(dataset) != 120:
        errors.append(f"样本总数应为120，实际为{len(dataset)}")
    if len(ids) != len(set(ids)):
        errors.append("case_id存在重复")
    normalized_questions = [str(item.get("question", "")).strip().lower() for item in dataset]
    if len(normalized_questions) != len(set(normalized_questions)):
        errors.append("问题文本存在完全重复")

    for category in sorted(RAG_CATEGORIES):
        rows = [item for item in dataset if item.get("category") == category]
        if len(rows) != 20:
            errors.append(f"{category} 应有20条，实际{len(rows)}条")
        counts = Counter(item.get("question_type") for item in rows)
        if dict(counts) != EXPECTED_PER_CATEGORY:
            errors.append(f"{category} 类型分布错误: {dict(counts)}")
        if sum(1 for item in rows if item.get("split") == "dev") != 4:
            errors.append(f"{category} dev应为4条")
        if sum(1 for item in rows if item.get("smoke")) != 5:
            errors.append(f"{category} smoke应为5条")

    required = {
        "case_id", "category", "question_type", "question", "expected_found",
        "reference_answer", "reference_claims", "forbidden_claims",
        "relevant_chunk_uids", "reference_contexts", "review_status",
    }
    for item in dataset:
        case_id = item.get("case_id", "<unknown>")
        missing = sorted(required - set(item))
        if missing:
            errors.append(f"{case_id} 缺少字段: {missing}")
            continue
        if not str(item["question"]).strip() or not str(item["reference_answer"]).strip():
            errors.append(f"{case_id} 问题或参考答案为空")
        unknown = set(item["relevant_chunk_uids"]) - known_chunks
        if unknown:
            errors.append(f"{case_id} 引用了未知Chunk: {sorted(unknown)}")
        if item["expected_found"] and not item["relevant_chunk_uids"]:
            errors.append(f"{case_id} 可回答问题缺少相关Chunk")
        if not item["expected_found"] and item["relevant_chunk_uids"]:
            errors.append(f"{case_id} 不可回答问题不应绑定相关Chunk")
        if not allow_pending and item["review_status"] != "approved":
            errors.append(f"{case_id} 尚未人工审核通过")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="校验RAG评测集")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    dataset_path = args.dataset or DATASETS_DIR / f"chinatravel_rag_{args.version}.jsonl"
    manifest_path = args.manifest or MANIFESTS_DIR / f"kb_{args.version}.json"
    errors = validate(
        read_jsonl(dataset_path),
        read_json(manifest_path),
        allow_pending=args.allow_pending,
    )
    if errors:
        print("评测集校验失败：")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"评测集校验通过: {dataset_path}")


if __name__ == "__main__":
    main()

