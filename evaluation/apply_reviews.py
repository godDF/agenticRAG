from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluation.common import DATASETS_DIR, read_jsonl, write_jsonl


JSON_FIELDS = {
    "reference_claims",
    "forbidden_claims",
    "relevant_chunk_uids",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="把人工审核CSV合并回评测JSONL")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--review-sheet", type=Path)
    parser.add_argument(
        "--approve-all",
        action="store_true",
        help="用户已逐条审核后，将全部样本标记为approved",
    )
    parser.add_argument("--reviewer", default="user_reviewed")
    args = parser.parse_args()
    dataset_path = args.dataset or DATASETS_DIR / f"chinatravel_rag_{args.version}.jsonl"
    review_path = args.review_sheet or DATASETS_DIR / f"review_sheet_{args.version}.csv"
    dataset = read_jsonl(dataset_path)
    if args.approve_all:
        for item in dataset:
            item["review_status"] = "approved"
            item["reviewer"] = args.reviewer
        write_jsonl(dataset_path, dataset)
        if review_path.exists():
            with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = list(reader.fieldnames or [])
            for row in rows:
                row["review_status"] = "approved"
                row["reviewer"] = args.reviewer
            with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        print(f"审核结果已合并: {dataset_path}，approved={len(dataset)}/{len(dataset)}")
        return
    by_id = {item["case_id"]: item for item in dataset}
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    unknown = sorted({row.get("case_id", "") for row in rows} - set(by_id))
    if unknown:
        raise ValueError(f"审核表包含未知case_id: {unknown}")
    for row in rows:
        item = by_id[row["case_id"]]
        for field in ("question", "reference_answer", "review_status", "reviewer", "review_notes"):
            if field in row:
                item[field] = str(row[field]).strip()
        for field in JSON_FIELDS:
            if field in row:
                value = json.loads(row[field] or "[]")
                if not isinstance(value, list):
                    raise ValueError(f"{row['case_id']} 的 {field} 必须是JSON数组")
                item[field] = value
    write_jsonl(dataset_path, dataset)
    approved = sum(item.get("review_status") == "approved" for item in dataset)
    print(f"审核结果已合并: {dataset_path}，approved={approved}/{len(dataset)}")


if __name__ == "__main__":
    main()
