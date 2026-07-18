from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evaluation.common import RUNS_DIR, read_json, read_jsonl, write_jsonl
from evaluation.metrics import retrieval_views


SYSTEM_DIRS = {
    "baseline": "traditional_rag",
    "agentic": "agentic_rag",
    "production": "agentic_rag_production",
}


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "recall_at_1": "retrieval.final.recall@1",
        "recall_at_3": "retrieval.final.recall@3",
        "recall_at_5": "retrieval.final.recall@5",
        "precision_at_1": "retrieval.final.precision@1",
        "precision_at_3": "retrieval.final.precision@3",
        "precision_at_5": "retrieval.final.precision@5",
        "mrr": "retrieval.final.mrr",
        "ndcg_at_5": "retrieval.final.ndcg@5",
        "hitrate_at_5": "retrieval.final.hitrate@5",
        "faithfulness": "ragas.faithfulness",
        "answer_relevancy": "ragas.answer_relevancy",
        "context_precision": "ragas.context_precision",
        "context_recall": "ragas.context_recall",
        "answer_correctness": "ragas.answer_correctness",
        "f1": "answer.token_f1",
        "found_accuracy": "decision.found_accuracy",
    }
    return {name: metrics.get(key) for name, key in mapping.items()}


def export_run(run_id: str, *, run_dir: Path | None = None) -> None:
    run_dir = run_dir or RUNS_DIR / run_id
    config = read_json(run_dir / "config.json")
    dataset_path = config.get("dataset_path")
    if not dataset_path:
        return
    dataset = read_jsonl(Path(dataset_path))
    cases = {item["case_id"]: item for item in dataset}
    raw_rows = read_jsonl(run_dir / "raw_results.jsonl")
    score_rows = read_jsonl(run_dir / "scores.jsonl")
    scores = {
        (item["case_id"], item["system"], int(item.get("repetition", 1))): item
        for item in score_rows
    }
    selected_case_ids = {
        item["case_id"] for item in raw_rows if int(item.get("repetition", 1)) == 1
    }
    testset_rows = []
    for case_id in [item["case_id"] for item in dataset if item["case_id"] in selected_case_ids]:
        case = cases[case_id]
        testset_rows.append(
            {
                "case_id": case_id,
                "question": case["question"],
                "difficulty": case["difficulty"],
                "question_type": case["question_type"],
                "category": case["category"],
                "ground_truth_answer": case["reference_answer"],
                "item_name": case["category"],
                "relevant_docs": case["relevant_chunk_uids"],
                "expected_found": case["expected_found"],
            }
        )

    for system, directory in SYSTEM_DIRS.items():
        system_raw = [
            item
            for item in raw_rows
            if item["system"] == system and int(item.get("repetition", 1)) == 1
        ]
        if not system_raw:
            continue
        output_dir = run_dir / directory
        write_jsonl(output_dir / "testset_real.jsonl", testset_rows)
        result_rows = []
        for raw in system_raw:
            case = cases[raw["case_id"]]
            score = scores.get((raw["case_id"], system, 1), {})
            metrics = score.get("metrics") or {}
            result = raw.get("result") or {}
            final_hits = retrieval_views(result.get("evaluation_events") or []).get("final") or []
            meta = result.get("meta") or {}
            result_rows.append(
                {
                    "case_id": raw["case_id"],
                    "question": case["question"],
                    "ground_truth_answer": case["reference_answer"],
                    "generated_answer": result.get("answer", ""),
                    "found": result.get("found", False),
                    "retrieved_chunk_ids": [item.get("chunk_uid") for item in final_hits],
                    "relevant_docs": case["relevant_chunk_uids"],
                    "difficulty": case["difficulty"],
                    "question_type": case["question_type"],
                    "category": case["category"],
                    "metrics": _public_metrics(metrics),
                    "timing": {
                        "total_s": round(float(meta.get("latency_ms", 0) or 0) / 1000, 3)
                    },
                    "usage": {
                        "input_tokens": int(meta.get("input_tokens", 0) or 0),
                        "output_tokens": int(meta.get("output_tokens", 0) or 0),
                        "estimated_cost_cny": float(meta.get("estimated_cost_cny", 0) or 0),
                    },
                    "trace_id": result.get("trace_id"),
                    "status": raw.get("status"),
                    "error": raw.get("error"),
                }
            )
        write_jsonl(output_dir / "eval_results.jsonl", result_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出兼容旧评测项目的JSONL结果")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    export_run(args.run_id)
    print(f"兼容格式已导出: {RUNS_DIR / args.run_id}")


if __name__ == "__main__":
    main()
