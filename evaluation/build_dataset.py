from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.clients import LlmClient
from app.config import RAG_CATEGORIES, settings
from evaluation import RANDOM_SEED
from evaluation.common import (
    DATASETS_DIR,
    ensure_evaluation_directories,
    write_csv,
    write_json,
    write_jsonl,
)
from evaluation.corpus import (
    DEFAULT_EVAL_COLLECTION,
    DEFAULT_KB_DIR,
    build_corpus_manifest,
    index_manifest,
    save_manifest,
)


QUESTION_DISTRIBUTION = (
    ["single_hop"] * 8
    + ["paraphrase"] * 4
    + ["multi_context"] * 4
    + ["unanswerable"] * 2
    + ["noise_conflict"] * 2
)

DIFFICULTY = {
    "single_hop": "easy",
    "paraphrase": "medium",
    "multi_context": "hard",
    "unanswerable": "hard",
    "noise_conflict": "hard",
}


def _blueprints_for_category(
    category: str,
    chunks: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not chunks:
        raise ValueError(f"分类没有Chunk: {category}")
    shuffled = list(chunks)
    rng.shuffle(shuffled)
    blueprints = []
    for index, question_type in enumerate(QUESTION_DISTRIBUTION, start=1):
        if question_type == "unanswerable":
            anchors: list[dict[str, Any]] = []
        elif question_type == "multi_context":
            first = shuffled[(index - 1) % len(shuffled)]
            second = shuffled[index % len(shuffled)]
            anchors = [first] if first["chunk_uid"] == second["chunk_uid"] else [first, second]
        else:
            anchors = [shuffled[(index - 1) % len(shuffled)]]
        blueprints.append(
            {
                "blueprint_id": f"{category}_{index:03d}",
                "category": category,
                "question_type": question_type,
                "difficulty": DIFFICULTY[question_type],
                "anchors": anchors,
            }
        )
    return blueprints


def build_blueprints(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in manifest["chunks"]:
        grouped[chunk["category"]].append(chunk)
    rng = random.Random(RANDOM_SEED)
    blueprints = []
    for category in sorted(RAG_CATEGORIES):
        blueprints.extend(_blueprints_for_category(category, grouped[category], rng))
    return blueprints


def _category_digest(chunks: list[dict[str, Any]], max_chars: int = 12000) -> str:
    blocks = []
    used = 0
    for chunk in chunks:
        block = (
            f"CHUNK_UID={chunk['chunk_uid']}\n"
            f"标题={chunk['title']}\n内容={chunk['content']}"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _generation_prompt(
    category: str,
    blueprints: list[dict[str, Any]],
    category_chunks: list[dict[str, Any]],
) -> str:
    specifications = []
    for blueprint in blueprints:
        anchors = blueprint["anchors"]
        specifications.append(
            {
                "blueprint_id": blueprint["blueprint_id"],
                "question_type": blueprint["question_type"],
                "anchor_chunk_uids": [item["chunk_uid"] for item in anchors],
                "anchor_evidence": [item["content"] for item in anchors],
            }
        )
    digest = _category_digest(category_chunks)
    return f"""你是ChinaTravel旅行规则RAG评测集设计员。当前类别固定为 {category}。
请严格依据给出的知识片段生成评测问题，不得补充外部规则。

规则：
1. single_hop：直接由一个锚点片段回答。
2. paraphrase：使用不照抄原文、尽量不含核心原词的自然表达，答案仍由锚点支持。
3. multi_context：问题需要综合所有锚点片段，不得只靠一个片段完整回答。
4. unanswerable：生成该类别相关但下方完整摘要没有答案的问题；参考答案明确说明知识库信息不足。
5. noise_conflict：加入容易与相近规则混淆的条件，但正确答案只能来自锚点。
6. reference_claims拆成可独立核验的短事实；forbidden_claims列出1至2条容易产生但证据不支持的错误结论。
7. 不得输出Markdown，只返回JSON对象。

输出结构：
{{"items":[{{"blueprint_id":"...","question":"...","reference_answer":"...","reference_claims":["..."],"forbidden_claims":["..."]}}]}}

本批规格：
{json.dumps(specifications, ensure_ascii=False)}

当前分类知识摘要（用于确认不可回答问题确实缺失）：
{digest}"""


def _normalize_generated_item(
    blueprint: dict[str, Any],
    generated: dict[str, Any],
    version: str,
    category_index: int,
) -> dict[str, Any]:
    anchors = blueprint["anchors"]
    question_type = blueprint["question_type"]
    question = str(generated.get("question") or "").strip()
    answer = str(generated.get("reference_answer") or "").strip()
    claims = generated.get("reference_claims") or []
    forbidden = generated.get("forbidden_claims") or []
    if not question or not answer or not isinstance(claims, list):
        raise ValueError(f"生成样本字段缺失: {blueprint['blueprint_id']}")
    expected_found = question_type != "unanswerable"
    split = "dev" if category_index <= 4 else "test"
    smoke = category_index in {1, 9, 13, 17, 19}
    return {
        "case_id": blueprint["blueprint_id"],
        "dataset_version": version,
        "split": split,
        "smoke": smoke,
        "category": blueprint["category"],
        "question_type": question_type,
        "difficulty": blueprint["difficulty"],
        "question": question[:2000],
        "expected_found": expected_found,
        "reference_answer": answer,
        "reference_claims": [str(item).strip() for item in claims if str(item).strip()],
        "forbidden_claims": [
            str(item).strip() for item in forbidden if str(item).strip()
        ],
        "relevant_chunk_uids": [item["chunk_uid"] for item in anchors],
        "relevant_document_ids": sorted({item["document_id"] for item in anchors}),
        "reference_contexts": [item["content"] for item in anchors],
        "source_urls": sorted({item["source_url"] for item in anchors if item["source_url"]}),
        "review_status": "pending",
        "reviewer": "",
        "review_notes": "",
    }


async def generate_dataset(
    manifest: dict[str, Any],
    *,
    version: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    settings.validate_query_runtime()
    llm = LlmClient(settings)
    blueprints = build_blueprints(manifest)
    chunks_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in manifest["chunks"]:
        chunks_by_category[chunk["category"]].append(chunk)
    generated_rows: list[dict[str, Any]] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    for category in sorted(RAG_CATEGORIES):
        category_blueprints = [item for item in blueprints if item["category"] == category]
        for start in range(0, len(category_blueprints), 5):
            batch = category_blueprints[start : start + 5]
            data, result = await llm.complete_json(
                _generation_prompt(category, batch, chunks_by_category[category])
            )
            usage["input_tokens"] += result.input_tokens
            usage["output_tokens"] += result.output_tokens
            items = data.get("items")
            if not isinstance(items, list):
                raise RuntimeError(f"{category} 评测集生成结果缺少 items")
            by_id = {
                str(item.get("blueprint_id")): item
                for item in items
                if isinstance(item, dict)
            }
            for blueprint in batch:
                generated = by_id.get(blueprint["blueprint_id"])
                if generated is None:
                    raise RuntimeError(f"模型漏掉规格: {blueprint['blueprint_id']}")
                category_index = int(blueprint["blueprint_id"].rsplit("_", 1)[1])
                generated_rows.append(
                    _normalize_generated_item(
                        blueprint, generated, version, category_index
                    )
                )
    return generated_rows, usage


def _review_rows(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in dataset:
        rows.append(
            {
                "case_id": item["case_id"],
                "category": item["category"],
                "question_type": item["question_type"],
                "split": item["split"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "reference_claims": json.dumps(item["reference_claims"], ensure_ascii=False),
                "forbidden_claims": json.dumps(item["forbidden_claims"], ensure_ascii=False),
                "relevant_chunk_uids": json.dumps(item["relevant_chunk_uids"], ensure_ascii=False),
                "review_status": item["review_status"],
                "reviewer": item["reviewer"],
                "review_notes": item["review_notes"],
            }
        )
    return rows


async def async_main(args: argparse.Namespace) -> None:
    ensure_evaluation_directories()
    manifest = build_corpus_manifest(args.kb_dir, args.version)
    manifest_path = save_manifest(manifest, args.version)
    index_result = None
    if not args.no_index:
        index_result = await index_manifest(
            manifest,
            collection_name=args.collection,
            recreate=not args.keep_existing,
        )
    if args.no_generate:
        print(f"知识快照已保存: {manifest_path}")
        if index_result:
            print(json.dumps(index_result, ensure_ascii=False, indent=2))
        return

    dataset, usage = await generate_dataset(manifest, version=args.version)
    dataset_path = DATASETS_DIR / f"chinatravel_rag_{args.version}.jsonl"
    review_path = DATASETS_DIR / f"review_sheet_{args.version}.csv"
    write_jsonl(dataset_path, dataset)
    write_csv(review_path, _review_rows(dataset))
    write_json(
        DATASETS_DIR / f"generation_meta_{args.version}.json",
        {
            "dataset_version": args.version,
            "seed": RANDOM_SEED,
            "sample_count": len(dataset),
            "distribution": dict(Counter(item["question_type"] for item in dataset)),
            "review_status": "pending_human_review",
            "llm_usage": usage,
            "index_result": index_result,
            "manifest_path": str(manifest_path),
        },
    )
    print(f"候选评测集已生成: {dataset_path}")
    print(f"人工审核表已生成: {review_path}")
    print("全部样本当前均为 pending，人工审核通过前 full 评测会拒绝执行。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建ChinaTravel RAG评测语料与候选数据集")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR)
    parser.add_argument("--collection", default=DEFAULT_EVAL_COLLECTION)
    parser.add_argument("--no-index", action="store_true", help="只生成清单，不写入Qdrant")
    parser.add_argument("--keep-existing", action="store_true", help="不重建评测Collection")
    parser.add_argument("--no-generate", action="store_true", help="只建立语料快照和索引")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))

