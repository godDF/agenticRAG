from __future__ import annotations

import math
import random
import re
from collections import Counter
from statistics import mean
from typing import Any, Iterable

from evaluation import RANDOM_SEED


KS = (1, 3, 5)


def _dedupe_hits(hits: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for hit in hits:
        uid = str(hit.get("chunk_uid", ""))
        if not uid or uid in seen:
            continue
        seen.add(uid)
        result.append(hit)
    return result


def retrieval_views(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    retrievals = [item for item in events if item.get("stage") == "retrieval"]
    views: dict[str, list[dict[str, Any]]] = {}
    all_rounds = []
    for event in retrievals:
        round_number = int(event.get("round", 0) or 0)
        hits = _dedupe_hits(event.get("fused_hits") or [])
        views[f"round{round_number}"] = hits
        all_rounds.extend(hits)
    views["all_rounds"] = _dedupe_hits(all_rounds)
    views["final"] = (
        _dedupe_hits(retrievals[-1].get("fused_hits") or []) if retrievals else []
    )
    gradings = [item for item in events if item.get("stage") == "evidence_grading"]
    views["accepted"] = (
        _dedupe_hits(gradings[-1].get("accepted_hits") or []) if gradings else []
    )
    return views


def _dcg(relevances: list[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevances))


def retrieval_metrics(
    hits: list[dict[str, Any]],
    relevant_chunk_uids: list[str],
    relevant_document_ids: list[str],
) -> dict[str, float | None]:
    relevant = set(relevant_chunk_uids)
    relevant_documents = set(relevant_document_ids)
    if not relevant:
        result: dict[str, float | None] = {
            f"recall@{k}": None for k in KS
        }
        result.update({f"precision@{k}": None for k in KS})
        result.update({f"hitrate@{k}": None for k in KS})
        result.update({f"ndcg@{k}": None for k in KS})
        result.update({f"document_recall@{k}": None for k in KS})
        result.update({f"matches@{k}": None for k in KS})
        result["relevant_count"] = None
        result["mrr"] = None
        return result

    result = {"relevant_count": float(len(relevant))}
    ranked_uids = [str(item.get("chunk_uid", "")) for item in hits]
    ranked_docs = [str(item.get("document_id", "")) for item in hits]
    for k in KS:
        top_uids = ranked_uids[:k]
        matches = sum(1 for uid in top_uids if uid in relevant)
        result[f"matches@{k}"] = float(matches)
        result[f"recall@{k}"] = matches / len(relevant)
        result[f"precision@{k}"] = matches / k
        result[f"hitrate@{k}"] = float(matches > 0)
        relevances = [1 if uid in relevant else 0 for uid in top_uids]
        ideal = [1] * min(k, len(relevant))
        result[f"ndcg@{k}"] = _dcg(relevances) / _dcg(ideal) if ideal else 0.0
        if relevant_documents:
            matched_docs = len(set(ranked_docs[:k]) & relevant_documents)
            result[f"document_recall@{k}"] = matched_docs / len(relevant_documents)
        else:
            result[f"document_recall@{k}"] = None
    first_rank = next(
        (index for index, uid in enumerate(ranked_uids, start=1) if uid in relevant),
        None,
    )
    result["mrr"] = 1.0 / first_rank if first_rank else 0.0
    return result


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", str(text).lower())


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(_tokens(prediction))
    expected = Counter(_tokens(reference))
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _claim_coverage(answer: str, claims: list[str]) -> float | None:
    if not claims:
        return None
    covered = sum(token_f1(answer, claim) >= 0.55 for claim in claims)
    return covered / len(claims)


def _forbidden_rate(answer: str, claims: list[str]) -> float | None:
    if not claims:
        return None
    normalized = re.sub(r"\s+", "", answer).lower()
    hits = 0
    for claim in claims:
        claim_normalized = re.sub(r"\s+", "", claim).lower()
        if claim_normalized and claim_normalized in normalized:
            hits += 1
    return hits / len(claims)


def _citation_metrics(answer: str, context_count: int) -> tuple[float | None, float | None]:
    citations = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    validity = (
        sum(1 <= value <= context_count for value in citations) / len(citations)
        if citations
        else None
    )
    sentences = [
        item.strip()
        for item in re.split(r"[。！？\n]+", answer)
        if len(item.strip()) >= 8 and "具体规则可能变化" not in item
    ]
    coverage = (
        sum(bool(re.search(r"\[\d+\]", sentence)) for sentence in sentences)
        / len(sentences)
        if sentences
        else None
    )
    return validity, coverage


def score_result(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    events = result.get("evaluation_events") or []
    views = retrieval_views(events)
    metrics: dict[str, Any] = {}
    for view_name, hits in views.items():
        for name, value in retrieval_metrics(
            hits,
            case.get("relevant_chunk_uids") or [],
            case.get("relevant_document_ids") or [],
        ).items():
            metrics[f"retrieval.{view_name}.{name}"] = value

    answer = str(result.get("answer") or "")
    found = bool(result.get("found"))
    expected_found = bool(case.get("expected_found"))
    accepted = views.get("accepted") or []
    citation_validity, citation_coverage = _citation_metrics(answer, len(accepted))
    meta = result.get("meta") or {}
    metrics.update(
        {
            "answer.token_f1": token_f1(answer, str(case.get("reference_answer") or "")),
            "answer.required_claim_coverage": _claim_coverage(
                answer, case.get("reference_claims") or []
            ),
            "answer.forbidden_claim_rate": _forbidden_rate(
                answer, case.get("forbidden_claims") or []
            ),
            "answer.citation_validity": citation_validity,
            "answer.citation_coverage": citation_coverage,
            "decision.found_accuracy": float(found == expected_found),
            "decision.correct_abstention": (
                float(not found) if not expected_found else None
            ),
            "decision.false_answer": float(found) if not expected_found else None,
            "decision.false_rejection": float(not found) if expected_found else None,
            "agentic.rewritten": float(bool(meta.get("rewritten"))),
            "agentic.retrieval_rounds": float(meta.get("retrieval_rounds", 0) or 0),
            "agentic.verified": float(bool(meta.get("verified"))),
            "performance.latency_ms": float(meta.get("latency_ms", 0) or 0),
            "performance.input_tokens": float(meta.get("input_tokens", 0) or 0),
            "performance.output_tokens": float(meta.get("output_tokens", 0) or 0),
            "performance.estimated_cost_cny": float(
                meta.get("estimated_cost_cny", 0) or 0
            ),
            "performance.error": 0.0,
        }
    )
    round1_hit = metrics.get("retrieval.round1.hitrate@8")
    all_hit = metrics.get("retrieval.all_rounds.hitrate@8")
    metrics["agentic.second_round_recovery"] = (
        float(round1_hit == 0 and all_hit == 1)
        if round1_hit is not None and all_hit is not None
        else None
    )
    return {
        "case_id": case["case_id"],
        "system": result["system"],
        "category": case["category"],
        "question_type": case["question_type"],
        "difficulty": case["difficulty"],
        "expected_found": expected_found,
        "found": found,
        "trace_id": result.get("trace_id"),
        "metrics": metrics,
    }


def score_error(case: dict[str, Any], system: str, error: str) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "system": system,
        "category": case["category"],
        "question_type": case["question_type"],
        "difficulty": case["difficulty"],
        "expected_found": bool(case.get("expected_found")),
        "found": False,
        "trace_id": None,
        "error": error,
        "metrics": {"performance.error": 1.0},
    }


def metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = []
    for row in rows:
        value = (row.get("metrics") or {}).get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_ci(
    values: list[float],
    *,
    seed: int = RANDOM_SEED,
    iterations: int = 1000,
) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "count": 0}
    rng = random.Random(seed)
    sample_means = [
        mean(rng.choice(values) for _ in range(len(values))) for _ in range(iterations)
    ]
    return {
        "mean": mean(values),
        "ci95_low": percentile(sample_means, 0.025),
        "ci95_high": percentile(sample_means, 0.975),
        "count": len(values),
    }
