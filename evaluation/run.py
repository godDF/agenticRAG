from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any

from app.config import settings
from evaluation import RANDOM_SEED
from evaluation.common import (
    DATASETS_DIR,
    MANIFESTS_DIR,
    RUNS_DIR,
    append_jsonl,
    ensure_evaluation_directories,
    read_json,
    read_jsonl,
    safe_environment_snapshot,
    utc_run_id,
    write_json,
)
from evaluation.metrics import score_error, score_result
from evaluation.systems import create_systems
from evaluation.validate_dataset import validate


TRANSIENT_ERROR_MARKERS = ("429", "502", "503", "timeout", "timed out", "连接")


def _versions() -> dict[str, str]:
    names = [
        "fastapi", "qdrant-client", "openai", "httpx", "numpy",
        "langchain", "langchain-core", "ragaai-catalyst", "ragas",
    ]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _select_cases(dataset: list[dict[str, Any]], suite: str) -> list[dict[str, Any]]:
    if suite == "all":
        return list(dataset)
    if suite in {"smoke", "stability"}:
        return [item for item in dataset if item.get("smoke")]
    if suite == "dev":
        return [item for item in dataset if item.get("split") == "dev"]
    if suite == "full":
        return [item for item in dataset if item.get("split") == "test"]
    raise ValueError(f"未知suite: {suite}")


def _should_retry(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in TRANSIENT_ERROR_MARKERS)


async def _execute_one(system, case: dict[str, Any], timeout: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts = []
    last_error: Exception | None = None
    for attempt in range(1, 4):
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(system.answer(case), timeout=timeout)
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "completed",
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                }
            )
            return result, attempts
        except Exception as exc:
            last_error = exc
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            if attempt >= 3 or not _should_retry(exc):
                break
            await asyncio.sleep(2 ** (attempt - 1))
    assert last_error is not None
    raise RuntimeError(json.dumps({"attempts": attempts, "final_error": str(last_error)}, ensure_ascii=False))


async def async_main(args: argparse.Namespace) -> None:
    ensure_evaluation_directories()
    dataset_path = args.dataset or DATASETS_DIR / f"chinatravel_rag_{args.version}.jsonl"
    manifest_path = args.manifest or MANIFESTS_DIR / f"kb_{args.version}.json"
    dataset = read_jsonl(dataset_path)
    manifest = read_json(manifest_path)
    errors = validate(dataset, manifest, allow_pending=False)
    if errors:
        print("评测集未达到正式运行要求：")
        for error in errors[:30]:
            print(f"- {error}")
        raise SystemExit(1)
    cases = _select_cases(dataset, args.suite)
    repetitions = 3 if args.suite == "stability" else max(1, args.repetitions)
    system_names = [item.strip() for item in args.systems.split(",") if item.strip()]
    systems = create_systems(system_names, eval_collection=args.eval_collection)

    run_id = args.run_id or utc_run_id()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw_results.jsonl"
    scores_path = run_dir / "scores.jsonl"
    completed = set()
    if args.resume:
        for row in read_jsonl(raw_path):
            completed.add((row.get("case_id"), row.get("system"), row.get("repetition", 1)))
    elif raw_path.exists() or scores_path.exists():
        raise FileExistsError(f"运行目录已有结果，请使用 --resume: {run_dir}")

    config = {
        "run_id": run_id,
        "dataset_version": args.version,
        "dataset_path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "corpus_hash": manifest.get("corpus_hash"),
        "suite": args.suite,
        "case_count": len(cases),
        "systems": system_names,
        "repetitions": repetitions,
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "seed": RANDOM_SEED,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": _versions(),
        "runtime": safe_environment_snapshot(),
        "evaluation_collection": args.eval_collection,
        "production_collection": settings.qdrant_collection,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(run_dir / "config.json", config)
    if "production" in system_names:
        from evaluation.production_health import save_audit

        config["production_health"] = await asyncio.to_thread(
            save_audit, run_dir, manifest
        )
        write_json(run_dir / "config.json", config)

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    write_lock = asyncio.Lock()

    async def worker(system_name: str, case: dict[str, Any], repetition: int) -> None:
        key = (case["case_id"], system_name, repetition)
        if key in completed:
            return
        async with semaphore:
            started = time.perf_counter()
            try:
                result, attempts = await _execute_one(
                    systems[system_name], case, args.timeout
                )
                raw = {
                    "case_id": case["case_id"],
                    "system": system_name,
                    "repetition": repetition,
                    "status": "completed",
                    "question": case["question"],
                    "category": case["category"],
                    "question_type": case["question_type"],
                    "attempts": attempts,
                    "wall_latency_ms": int((time.perf_counter() - started) * 1000),
                    "result": result,
                }
                scored = {**score_result(case, result), "repetition": repetition}
            except Exception as exc:
                raw = {
                    "case_id": case["case_id"],
                    "system": system_name,
                    "repetition": repetition,
                    "status": "failed",
                    "question": case["question"],
                    "category": case["category"],
                    "question_type": case["question_type"],
                    "wall_latency_ms": int((time.perf_counter() - started) * 1000),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                }
                scored = {
                    **score_error(case, system_name, str(exc)),
                    "repetition": repetition,
                }
            async with write_lock:
                append_jsonl(raw_path, raw)
                append_jsonl(scores_path, scored)
                completed.add(key)
                print(
                    f"[{len(completed):>4}] {system_name:<10} "
                    f"{case['case_id']} r{repetition} {raw['status']}"
                )

    tasks = []
    for repetition in range(1, repetitions + 1):
        for case in cases:
            for system_name in system_names:
                tasks.append(worker(system_name, case, repetition))
    await asyncio.gather(*tasks)
    config["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    config["completed_items"] = len(completed)
    write_json(run_dir / "config.json", config)

    from evaluation.build_report import build_report

    build_report(run_id)
    print(f"评测完成，结果目录: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行ChinaTravel RAG离线评测")
    parser.add_argument("--version", default="v1")
    parser.add_argument(
        "--suite",
        choices=["smoke", "dev", "full", "all", "stability"],
        default="smoke",
    )
    parser.add_argument("--systems", default="baseline,agentic")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--eval-collection", default="chinatravel_safety_eval_v1")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=45.0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
