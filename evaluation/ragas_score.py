from __future__ import annotations

import argparse
import asyncio
import ctypes
import os
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from evaluation.common import DATASETS_DIR, RUNS_DIR, append_jsonl, read_jsonl, write_jsonl
from evaluation.metrics import retrieval_views


os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


def _prepare_windows_pyarrow() -> None:
    """Load Conda's Arrow runtime before importing RAGAS on Windows.

    Some Windows environments do not resolve ``arrow.dll`` transitively from
    ``pyarrow.lib.pyd`` even though the DLL exists in ``Library/bin``.
    Preloading it keeps RAGAS isolated from the ChinaTravel runtime environment.
    """
    if sys.platform != "win32":
        return
    library_bin = Path(sys.prefix) / "Library" / "bin"
    arrow_dlls = ["arrow.dll", "arrow_acero.dll", "arrow_dataset.dll"]
    if (library_bin / arrow_dlls[0]).exists():
        os.add_dll_directory(str(library_bin))
        for name in arrow_dlls:
            dll = library_bin / name
            if dll.exists():
                ctypes.CDLL(str(dll))


def _openai_base_url(url: str) -> str:
    """Normalize provider roots for OpenAI-compatible SDK clients."""
    normalized = url.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


def _score_value(result: Any) -> float:
    value = getattr(result, "value", result)
    return float(value)


async def _build_scorers():
    _prepare_windows_pyarrow()
    try:
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "当前环境未安装ragas==0.4.3；请在独立agenticRAG-eval环境运行"
        ) from exc

    llm_client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_api_url,
        timeout=90,
        max_retries=2,
    )
    embedding_client = AsyncOpenAI(
        api_key=settings.embedding_api_key,
        base_url=_openai_base_url(settings.embedding_api_url),
        timeout=90,
        max_retries=2,
    )
    judge = llm_factory(
        settings.llm_model,
        provider="openai",
        client=llm_client,
        temperature=0,
        max_tokens=8192,
        extra_body={"thinking": {"type": "disabled"}},
    )
    embeddings = embedding_factory(
        "openai",
        model=settings.embedding_model,
        client=embedding_client,
    )
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
        "answer_correctness": FactualCorrectness(llm=judge, mode="f1"),
    }


async def _score_case(scorers, case: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    result = raw["result"]
    answer = str(result.get("answer") or "")
    events = result.get("evaluation_events") or []
    accepted = retrieval_views(events).get("accepted") or []
    contexts = [str(item.get("content") or "") for item in accepted if item.get("content")]
    reference = str(case.get("reference_answer") or "")
    question = str(case.get("question") or "")
    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}

    calls = {
        "faithfulness": {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        },
        "answer_relevancy": {"user_input": question, "response": answer},
        "context_precision": {
            "user_input": question,
            "reference": reference,
            "retrieved_contexts": contexts,
        },
        "context_recall": {
            "user_input": question,
            "reference": reference,
            "retrieved_contexts": contexts,
        },
        "answer_correctness": {"response": answer, "reference": reference},
    }
    for name, kwargs in calls.items():
        if not contexts and name in {"faithfulness", "context_precision", "context_recall"}:
            scores[name] = None
            continue
        try:
            scores[name] = _score_value(await scorers[name].ascore(**kwargs))
        except Exception as exc:
            scores[name] = None
            errors[name] = f"{type(exc).__name__}: {exc}"[:1000]
    return {
        "case_id": raw["case_id"],
        "system": raw["system"],
        "repetition": int(raw.get("repetition", 1)),
        "scores": scores,
        "errors": errors,
    }


async def async_main(args: argparse.Namespace) -> None:
    settings.validate_query_runtime()
    run_dir = RUNS_DIR / args.run_id
    raw_rows = [item for item in read_jsonl(run_dir / "raw_results.jsonl") if item.get("status") == "completed"]
    dataset_path = args.dataset or DATASETS_DIR / f"chinatravel_rag_{args.version}.jsonl"
    cases = {item["case_id"]: item for item in read_jsonl(dataset_path)}
    output_path = run_dir / "ragas_scores.jsonl"
    completed = {
        (item["case_id"], item["system"], int(item.get("repetition", 1)))
        for item in read_jsonl(output_path)
    }
    scorers = await _build_scorers()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    write_lock = asyncio.Lock()

    async def worker(raw: dict[str, Any]) -> None:
        key = (raw["case_id"], raw["system"], int(raw.get("repetition", 1)))
        if key in completed:
            return
        async with semaphore:
            row = await _score_case(scorers, cases[raw["case_id"]], raw)
            async with write_lock:
                append_jsonl(output_path, row)
                completed.add(key)
                print(f"RAGAS {raw['system']} {raw['case_id']} 完成")

    await asyncio.gather(*(worker(raw) for raw in raw_rows))
    ragas_rows = {
        (item["case_id"], item["system"], int(item.get("repetition", 1))): item
        for item in read_jsonl(output_path)
    }
    deterministic = read_jsonl(run_dir / "scores.jsonl")
    for row in deterministic:
        key = (row["case_id"], row["system"], int(row.get("repetition", 1)))
        ragas_row = ragas_rows.get(key)
        if ragas_row:
            for name, value in ragas_row["scores"].items():
                row.setdefault("metrics", {})[f"ragas.{name}"] = value
            if ragas_row["errors"]:
                row["ragas_errors"] = ragas_row["errors"]
    write_jsonl(run_dir / "scores.jsonl", deterministic)

    from evaluation.build_report import build_report

    build_report(args.run_id)
    print(f"RAGAS评分和报告已更新: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在独立环境中为评测结果补充RAGAS指标")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--concurrency", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
