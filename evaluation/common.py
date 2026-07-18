from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "evaluation"
DATASETS_DIR = EVALUATION_ROOT / "datasets"
MANIFESTS_DIR = EVALUATION_ROOT / "manifests"
RUNS_DIR = EVALUATION_ROOT / "runs"


def ensure_evaluation_directories() -> None:
    for path in (DATASETS_DIR, MANIFESTS_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def utc_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit = "no-git"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass
    return f"{stamp}_{commit}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是有效 JSON") from exc
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_environment_snapshot() -> dict[str, str]:
    keys = [
        "LLM_MODEL",
        "BGE_M3_MODEL",
        "BGE_M3_VECTOR_SIZE",
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "RAG_RETRIEVE_TOP_K",
        "RAG_CONTEXT_TOP_K",
        "RAG_RETRIEVE_THRESHOLD",
        "RAG_ACCEPT_THRESHOLD",
        "RAG_MAX_ROUNDS",
        "RAG_MAX_SUBQUERIES",
    ]
    return {key: os.getenv(key, "") for key in keys}

