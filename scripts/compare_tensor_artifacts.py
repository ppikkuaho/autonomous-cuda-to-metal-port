#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT, ensure_dir, write_json


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if p.is_dir():
        p = p / "tensor_summaries.jsonl"
    return p


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["name"]] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two tensor_summaries.jsonl artifacts.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--stat-atol", type=float, default=1e-3)
    args = parser.parse_args()

    left_path = resolve(args.left)
    right_path = resolve(args.right)
    left = load_jsonl(left_path)
    right = load_jsonl(right_path)

    names = sorted(set(left) | set(right))
    comparisons = []
    failures = 0
    for name in names:
        lrow = left.get(name)
        rrow = right.get(name)
        result = {"name": name, "status": "PASS", "differences": []}
        if lrow is None or rrow is None:
            result["status"] = "FAIL"
            result["differences"].append("missing on one side")
        else:
            for key in ["shape", "dtype"]:
                if lrow.get(key) != rrow.get(key):
                    result["differences"].append(f"{key}: {lrow.get(key)} != {rrow.get(key)}")
            for key in ["min", "max", "mean", "std", "zero_fraction"]:
                if lrow.get(key) is None or rrow.get(key) is None:
                    continue
                if abs(float(lrow[key]) - float(rrow[key])) > args.stat_atol:
                    result["differences"].append(f"{key}: {lrow[key]} != {rrow[key]}")
            if result["differences"]:
                result["status"] = "FAIL"
        if result["status"] != "PASS":
            failures += 1
        comparisons.append(result)

    report = {
        "left": str(left_path),
        "right": str(right_path),
        "count": len(comparisons),
        "failures": failures,
        "comparisons": comparisons,
    }
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        ensure_dir(out)
        write_json(out / "tensor_compare_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

