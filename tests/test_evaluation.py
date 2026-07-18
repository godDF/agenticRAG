import json
from collections import Counter
from pathlib import Path

from app.chunk_identity import build_chunk_uid, chunk_uid_from_payload
from evaluation.build_dataset import build_blueprints
from evaluation.metrics import retrieval_metrics, retrieval_views, score_result, token_f1
from evaluation.common import read_json, write_json, write_jsonl
from evaluation.validate_dataset import EXPECTED_PER_CATEGORY, validate


def _manifest():
    chunks = []
    categories = [
        "attraction_notice",
        "child_ticket",
        "elderly_ticket",
        "flight_safety",
        "highspeed_rail_safety",
        "student_ticket",
    ]
    for category in categories:
        for index in range(4):
            content = f"{category} 规则内容 {index}"
            chunks.append(
                {
                    "chunk_uid": build_chunk_uid(
                        category=category,
                        title=f"{category}标题",
                        source_url="https://example.com",
                        content=content,
                    ),
                    "document_id": f"{category}-doc-{index}",
                    "category": category,
                    "title": f"{category}标题",
                    "source_url": "https://example.com",
                    "content": content,
                }
            )
    return {"chunks": chunks}


def test_chunk_uid_is_stable_for_whitespace_and_prefers_existing_value():
    first = build_chunk_uid(
        category="student_ticket",
        title="学生票",
        source_url="https://example.com",
        content="规则\n  内容",
    )
    second = build_chunk_uid(
        category="student_ticket",
        title="学生票",
        source_url="https://example.com",
        content="规则 内容",
    )
    assert first == second
    assert chunk_uid_from_payload({"chunk_uid": "fixed"}) == "fixed"


def test_blueprints_have_exact_120_case_distribution():
    blueprints = build_blueprints(_manifest())
    assert len(blueprints) == 120
    by_category = {}
    for category in {item["category"] for item in blueprints}:
        rows = [item for item in blueprints if item["category"] == category]
        by_category[category] = Counter(item["question_type"] for item in rows)
        assert len(rows) == 20
    assert all(dict(counts) == EXPECTED_PER_CATEGORY for counts in by_category.values())


def test_retrieval_metrics_and_views_use_stable_chunk_ids():
    hits = [
        {"chunk_uid": "noise", "document_id": "x"},
        {"chunk_uid": "gold", "document_id": "d1"},
    ]
    values = retrieval_metrics(hits, ["gold"], ["d1"])
    assert values["recall@1"] == 0
    assert values["recall@3"] == 1
    assert values["precision@3"] == 1 / 3
    assert values["mrr"] == 0.5
    views = retrieval_views(
        [
            {"stage": "retrieval", "round": 1, "fused_hits": hits},
            {
                "stage": "evidence_grading",
                "round": 1,
                "accepted_hits": [hits[1]],
            },
        ]
    )
    assert views["accepted"][0]["chunk_uid"] == "gold"


def test_score_result_captures_decision_and_performance():
    case = {
        "case_id": "student_ticket_001",
        "category": "student_ticket",
        "question_type": "single_hop",
        "difficulty": "easy",
        "expected_found": True,
        "reference_answer": "学生票需要核验资质。",
        "reference_claims": ["需要核验资质"],
        "forbidden_claims": ["不需要证件"],
        "relevant_chunk_uids": ["gold"],
        "relevant_document_ids": ["d1"],
    }
    result = {
        "system": "agentic",
        "found": True,
        "answer": "学生票需要核验资质。[1]",
        "trace_id": "t1",
        "evaluation_events": [
            {
                "stage": "retrieval",
                "round": 1,
                "fused_hits": [{"chunk_uid": "gold", "document_id": "d1"}],
            },
            {
                "stage": "evidence_grading",
                "round": 1,
                "accepted_hits": [
                    {"chunk_uid": "gold", "document_id": "d1", "content": "需要核验资质"}
                ],
            },
        ],
        "meta": {
            "retrieval_rounds": 1,
            "rewritten": False,
            "verified": True,
            "latency_ms": 100,
            "input_tokens": 10,
            "output_tokens": 5,
            "estimated_cost_cny": 0.001,
        },
    }
    scored = score_result(case, result)
    assert scored["metrics"]["retrieval.final.recall@1"] == 1
    assert scored["metrics"]["decision.found_accuracy"] == 1
    assert scored["metrics"]["answer.forbidden_claim_rate"] == 0
    assert token_f1("学生票需要核验资质", "需要核验资质") > 0.5


def test_validator_rejects_pending_review_even_when_shape_is_valid():
    manifest = _manifest()
    blueprints = build_blueprints(manifest)
    rows = []
    per_category_index = Counter()
    for blueprint in blueprints:
        category = blueprint["category"]
        per_category_index[category] += 1
        index = per_category_index[category]
        anchors = blueprint["anchors"]
        expected_found = blueprint["question_type"] != "unanswerable"
        rows.append(
            {
                "case_id": blueprint["blueprint_id"],
                "category": category,
                "question_type": blueprint["question_type"],
                "question": f"{blueprint['blueprint_id']}问题",
                "expected_found": expected_found,
                "reference_answer": "参考答案",
                "reference_claims": ["事实"],
                "forbidden_claims": [],
                "relevant_chunk_uids": [item["chunk_uid"] for item in anchors],
                "reference_contexts": [item["content"] for item in anchors],
                "review_status": "pending",
                "split": "dev" if index <= 4 else "test",
                "smoke": index in {1, 9, 13, 17, 19},
            }
        )
    assert validate(rows, manifest, allow_pending=True) == []
    errors = validate(rows, manifest, allow_pending=False)
    assert len([error for error in errors if "尚未人工审核" in error]) == 120


def test_report_builds_agentic_vs_baseline_tables(tmp_path, monkeypatch):
    import evaluation.build_report as report_module

    monkeypatch.setattr(report_module, "RUNS_DIR", tmp_path)
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    write_json(
        run_dir / "config.json",
        {
            "run_id": "run-1",
            "dataset_version": "v1",
            "suite": "smoke",
            "systems": ["baseline", "agentic"],
        },
    )
    common = {
        "case_id": "student_ticket_001",
        "category": "student_ticket",
        "question_type": "single_hop",
        "difficulty": "easy",
        "repetition": 1,
    }
    write_jsonl(
        run_dir / "scores.jsonl",
        [
            {
                **common,
                "system": "baseline",
                "metrics": {
                    "retrieval.final.recall@5": 0.5,
                    "retrieval.final.precision@5": 0.2,
                    "retrieval.final.mrr": 0.5,
                    "performance.latency_ms": 100,
                    "performance.error": 0,
                },
            },
            {
                **common,
                "system": "agentic",
                "metrics": {
                    "retrieval.final.recall@5": 1.0,
                    "retrieval.final.precision@5": 0.4,
                    "retrieval.final.mrr": 1.0,
                    "retrieval.round1.recall@5": 0.5,
                    "retrieval.all_rounds.recall@5": 1.0,
                    "performance.latency_ms": 200,
                    "performance.error": 0,
                },
            },
        ],
    )
    write_jsonl(
        run_dir / "raw_results.jsonl",
        [
            {"case_id": "student_ticket_001", "system": "baseline", "status": "completed"},
            {"case_id": "student_ticket_001", "system": "agentic", "status": "completed"},
        ],
    )
    report_path = report_module.build_report("run-1")
    assert report_path.exists()
    assert (run_dir / "comparison_overall.csv").exists()
    assert "Agentic RAG" in report_path.read_text(encoding="utf-8")
    summary = read_json(run_dir / "summary.json")
    assert summary["systems"]["agentic"]["retrieval.final.recall@5"]["mean"] == 1.0
