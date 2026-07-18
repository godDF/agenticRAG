from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from evaluation import RANDOM_SEED
from evaluation.common import RUNS_DIR, read_json, read_jsonl, write_csv, write_json, write_jsonl
from evaluation.metrics import bootstrap_ci, metric_values, percentile


CORE_METRICS = [
    ("Recall@1", "retrieval.final.recall@1", True),
    ("Recall@3", "retrieval.final.recall@3", True),
    ("Recall@5", "retrieval.final.recall@5", True),
    ("Precision@1", "retrieval.final.precision@1", True),
    ("Precision@3", "retrieval.final.precision@3", True),
    ("Precision@5", "retrieval.final.precision@5", True),
    ("HitRate@5", "retrieval.final.hitrate@5", True),
    ("MRR", "retrieval.final.mrr", True),
    ("nDCG@5", "retrieval.final.ndcg@5", True),
    ("Faithfulness", "ragas.faithfulness", True),
    ("Answer Relevancy", "ragas.answer_relevancy", True),
    ("Context Precision", "ragas.context_precision", True),
    ("Context Recall", "ragas.context_recall", True),
    ("Answer Correctness", "ragas.answer_correctness", True),
    ("必要事实覆盖率", "answer.required_claim_coverage", True),
    ("引用有效率", "answer.citation_validity", True),
    ("正确拒答率", "decision.correct_abstention", True),
    ("错误作答率", "decision.false_answer", False),
]

PERFORMANCE_METRICS = [
    ("平均耗时(ms)", "performance.latency_ms", False, "mean"),
    ("P95耗时(ms)", "performance.latency_ms", False, "p95"),
    ("平均输入Token", "performance.input_tokens", False, "mean"),
    ("平均输出Token", "performance.output_tokens", False, "mean"),
    ("平均单次成本(CNY)", "performance.estimated_cost_cny", False, "mean"),
    ("错误率", "performance.error", False, "mean"),
]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _group(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]):
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return grouped


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = metric_values(rows, metric)
    return mean(values) if values else None


def _paired_delta_ci(
    baseline: list[dict[str, Any]],
    agentic: list[dict[str, Any]],
    metric: str,
    iterations: int = 1000,
) -> dict[str, Any]:
    def by_case(rows):
        values: dict[tuple[str, int], float] = {}
        for row in rows:
            value = (row.get("metrics") or {}).get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[(row["case_id"], int(row.get("repetition", 1)))] = float(value)
        return values

    base_map = by_case(baseline)
    agent_map = by_case(agentic)
    keys = sorted(set(base_map) & set(agent_map))
    differences = [agent_map[key] - base_map[key] for key in keys]
    if not differences:
        return {"delta": None, "ci95_low": None, "ci95_high": None, "paired_count": 0}
    rng = random.Random(RANDOM_SEED)
    samples = [
        mean(rng.choice(differences) for _ in differences) for _ in range(iterations)
    ]
    return {
        "delta": mean(differences),
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
        "paired_count": len(differences),
    }


def _comparison_row(
    label: str,
    metric: str,
    higher_is_better: bool,
    baseline: list[dict[str, Any]],
    agentic: list[dict[str, Any]],
    *,
    statistic: str = "mean",
) -> dict[str, Any]:
    base_values = metric_values(baseline, metric)
    agent_values = metric_values(agentic, metric)
    if statistic == "p95":
        base_value = percentile(base_values, 0.95)
        agent_value = percentile(agent_values, 0.95)
        delta_info = {
            "delta": (agent_value - base_value) if base_value is not None and agent_value is not None else None,
            "ci95_low": None,
            "ci95_high": None,
            "paired_count": min(len(base_values), len(agent_values)),
        }
    else:
        base_value = mean(base_values) if base_values else None
        agent_value = mean(agent_values) if agent_values else None
        delta_info = _paired_delta_ci(baseline, agentic, metric)
    delta = delta_info["delta"]
    if base_value in (None, 0) or delta is None:
        relative = None
    else:
        relative = (
            (agent_value - base_value) / base_value
            if higher_is_better
            else (base_value - agent_value) / base_value
        )
    winner = "N/A"
    if base_value is not None and agent_value is not None:
        if math.isclose(base_value, agent_value, rel_tol=1e-9, abs_tol=1e-12):
            winner = "持平"
        elif (agent_value > base_value) == higher_is_better:
            winner = "Agentic RAG"
        else:
            winner = "Traditional RAG"
    low, high = delta_info["ci95_low"], delta_info["ci95_high"]
    significant = bool(low is not None and high is not None and (low > 0 or high < 0))
    return {
        "指标": label,
        "metric_key": metric,
        "Traditional RAG": base_value,
        "Agentic RAG": agent_value,
        "绝对变化(Agentic-Baseline)": delta,
        "相对改善": relative,
        "变化95%CI下限": low,
        "变化95%CI上限": high,
        "配对样本数": delta_info["paired_count"],
        "统计显著": significant,
        "优胜系统": winner,
        "方向": "越高越好" if higher_is_better else "越低越好",
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in (row.get("metrics") or {}).items()
            if isinstance(value, (int, float))
        }
    )
    result = {}
    for name in metric_names:
        result[name] = bootstrap_ci(metric_values(rows, name))
    latencies = metric_values(rows, "performance.latency_ms")
    result["performance.latency_percentiles"] = {
        "p50": percentile(latencies, 0.50),
        "p90": percentile(latencies, 0.90),
        "p95": percentile(latencies, 0.95),
    }
    return result


def _micro_retrieval(rows: list[dict[str, Any]], view: str, k: int) -> dict[str, Any]:
    matches = metric_values(rows, f"retrieval.{view}.matches@{k}")
    relevant = metric_values(rows, f"retrieval.{view}.relevant_count")
    return {
        "recall": sum(matches) / sum(relevant) if relevant and sum(relevant) else None,
        "precision": sum(matches) / (len(matches) * k) if matches else None,
        "cases": len(matches),
    }


def _markdown_table(rows: list[dict[str, Any]], performance: bool = False) -> str:
    header = "| 指标 | Traditional RAG | Agentic RAG | 绝对变化 | 相对改善 | 95% CI | 显著 | 优胜系统 |\n"
    divider = "|---|---:|---:|---:|---:|---:|:---:|---|\n"
    lines = []
    for row in rows:
        digits = 2 if "耗时" in row["指标"] or "Token" in row["指标"] else 6 if "成本" in row["指标"] else 4
        relative = row["相对改善"]
        ci = (
            f"[{_fmt(row['变化95%CI下限'])}, {_fmt(row['变化95%CI上限'])}]"
            if row["变化95%CI下限"] is not None
            else "N/A"
        )
        lines.append(
            "| {label} | {base} | {agent} | {delta} | {relative} | {ci} | {sig} | {winner} |".format(
                label=row["指标"],
                base=_fmt(row["Traditional RAG"], digits),
                agent=_fmt(row["Agentic RAG"], digits),
                delta=_fmt(row["绝对变化(Agentic-Baseline)"], digits),
                relative=(f"{relative * 100:.2f}%" if relative is not None else "N/A"),
                ci=ci,
                sig="是" if row["统计显著"] else "否",
                winner=row["优胜系统"],
            )
        )
    return header + divider + "\n".join(lines)


def build_report(run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    scores_path = run_dir / "scores.jsonl"
    rows = read_jsonl(scores_path)
    if not rows:
        raise ValueError(f"没有可汇总的评分: {scores_path}")
    config = read_json(run_dir / "config.json")
    by_system = _group(rows, lambda item: item["system"])
    summary = {
        "run_id": run_id,
        "config": config,
        "systems": {name: _summary(items) for name, items in by_system.items()},
        "micro_retrieval": {
            name: {
                f"{view}@{k}": _micro_retrieval(items, view, k)
                for view in ("round1", "final", "all_rounds", "accepted")
                for k in (1, 3, 5)
            }
            for name, items in by_system.items()
        },
    }
    write_json(run_dir / "summary.json", summary)

    summary_rows = []
    for system_name, metrics in summary["systems"].items():
        for metric_name, stats in metrics.items():
            if not isinstance(stats, dict) or "mean" not in stats:
                continue
            summary_rows.append({"system": system_name, "metric": metric_name, **stats})
    write_csv(run_dir / "summary.csv", summary_rows)

    baseline = by_system.get("baseline", [])
    agentic = by_system.get("agentic", [])
    overall_rows = [
        _comparison_row(label, metric, higher, baseline, agentic)
        for label, metric, higher in CORE_METRICS
    ]
    performance_rows = [
        _comparison_row(label, metric, higher, baseline, agentic, statistic=statistic)
        for label, metric, higher, statistic in PERFORMANCE_METRICS
    ]
    write_csv(run_dir / "comparison_overall.csv", overall_rows + performance_rows)

    category_rows = []
    for category in sorted({row["category"] for row in rows}):
        base_group = [row for row in baseline if row["category"] == category]
        agent_group = [row for row in agentic if row["category"] == category]
        for label, metric, higher in (
            ("Recall@5", "retrieval.final.recall@5", True),
            ("MRR", "retrieval.final.mrr", True),
            ("Faithfulness", "ragas.faithfulness", True),
        ):
            category_rows.append(
                {"category": category, **_comparison_row(label, metric, higher, base_group, agent_group)}
            )
    write_csv(run_dir / "comparison_by_category.csv", category_rows)

    type_rows = []
    for question_type in sorted({row["question_type"] for row in rows}):
        base_group = [row for row in baseline if row["question_type"] == question_type]
        agent_group = [row for row in agentic if row["question_type"] == question_type]
        for label, metric, higher in (
            ("Recall@5", "retrieval.final.recall@5", True),
            ("MRR", "retrieval.final.mrr", True),
            ("Faithfulness", "ragas.faithfulness", True),
            ("正确拒答率", "decision.correct_abstention", True),
        ):
            type_rows.append(
                {"question_type": question_type, **_comparison_row(label, metric, higher, base_group, agent_group)}
            )
    write_csv(run_dir / "comparison_by_question_type.csv", type_rows)

    round_rows = []
    for view, label in (
        ("round1", "第一轮"),
        ("round2", "第二轮"),
        ("all_rounds", "两轮并集"),
        ("final", "最终一轮"),
        ("accepted", "最终接受证据"),
    ):
        round_rows.append(
            {
                "阶段": label,
                "Recall@5": _mean_metric(agentic, f"retrieval.{view}.recall@5"),
                "Precision@5": _mean_metric(agentic, f"retrieval.{view}.precision@5"),
                "MRR": _mean_metric(agentic, f"retrieval.{view}.mrr"),
                "HitRate@5": _mean_metric(agentic, f"retrieval.{view}.hitrate@5"),
            }
        )
    write_csv(run_dir / "comparison_agentic_rounds.csv", round_rows)

    raw_rows = read_jsonl(run_dir / "raw_results.jsonl")
    failed = [item for item in raw_rows if item.get("status") != "completed"]
    write_jsonl(run_dir / "failed_cases.jsonl", failed)

    markdown = [
        f"# ChinaTravel RAG评测报告\n",
        f"- Run ID：`{run_id}`",
        f"- 数据集：`{config.get('dataset_version')}`",
        f"- Suite：`{config.get('suite')}`",
        f"- 系统：{', '.join(config.get('systems', []))}",
        f"- 失败任务：{len(failed)}\n",
        "## 核心质量对比\n",
        _markdown_table(overall_rows),
        "\n## 性能与成本对比\n",
        _markdown_table(performance_rows, performance=True),
        "\n## Agentic检索阶段\n",
        "| 阶段 | Recall@5 | Precision@5 | MRR | HitRate@5 |\n|---|---:|---:|---:|---:|",
    ]
    markdown.extend(
        f"| {row['阶段']} | {_fmt(row['Recall@5'])} | {_fmt(row['Precision@5'])} | {_fmt(row['MRR'])} | {_fmt(row['HitRate@5'])} |"
        for row in round_rows
    )
    markdown.extend(
        [
            "\n## 说明\n",
            "- 相对改善已按指标方向计算；耗时、成本和错误率越低越好。",
            "- Baseline为0时相对改善显示N/A，避免误导性百分比。",
            "- 统计显著性基于同一问题的成对Bootstrap 95%置信区间。",
            "- RAGAS未执行时，对应生成质量指标显示N/A。",
        ]
    )
    report_text = "\n".join(markdown) + "\n"
    (run_dir / "comparison.md").write_text(report_text, encoding="utf-8")
    (run_dir / "report.md").write_text(report_text, encoding="utf-8")
    from evaluation.export_results import export_run

    export_run(run_id, run_dir=run_dir)
    return run_dir / "report.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="生成RAG评测汇总和对比表")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    path = build_report(args.run_id)
    print(f"评测报告已生成: {path}")


if __name__ == "__main__":
    main()
