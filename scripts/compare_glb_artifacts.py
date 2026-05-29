#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT, ensure_dir, write_json
from validate_glb import validate


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if p.is_dir():
        p = p / "outputs" / "output.glb"
    return p


def rel_delta(left: float, right: float) -> float:
    denom = max(abs(left), abs(right), 1.0)
    return abs(left - right) / denom


def bbox_extent(report: dict) -> list[float] | None:
    bounds = report.get("bounds") or []
    if not bounds:
        return None
    first = bounds[0]
    return [float(first[1][i]) - float(first[0][i]) for i in range(3)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two GLB artifacts by structural mesh stats.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--count-rtol", type=float, default=0.15)
    parser.add_argument("--extent-rtol", type=float, default=0.05)
    args = parser.parse_args()

    left_path = resolve(args.left)
    right_path = resolve(args.right)
    left = validate(left_path)
    right = validate(right_path)
    comparisons = []

    for key in ["vertex_count", "face_count"]:
        status = "PASS"
        detail = {}
        if key not in left or key not in right:
            status = "FAIL"
            detail["reason"] = "missing count"
        else:
            delta = rel_delta(float(left[key]), float(right[key]))
            detail["relative_delta"] = delta
            status = "PASS" if delta <= args.count_rtol else "WARN"
        comparisons.append({"name": key, "status": status, **detail})

    left_extent = bbox_extent(left)
    right_extent = bbox_extent(right)
    if left_extent is None or right_extent is None:
        comparisons.append({"name": "bbox_extent", "status": "FAIL", "reason": "missing bounds"})
    else:
        deltas = [rel_delta(l, r) for l, r in zip(left_extent, right_extent)]
        comparisons.append({
            "name": "bbox_extent",
            "status": "PASS" if max(deltas) <= args.extent_rtol else "WARN",
            "relative_deltas": deltas,
        })

    report = {
        "left_path": str(left_path),
        "right_path": str(right_path),
        "left": left,
        "right": right,
        "comparisons": comparisons,
        "thresholds": {
            "count_rtol": args.count_rtol,
            "extent_rtol": args.extent_rtol,
        },
        "status": (
            "FAIL" if any(c["status"] == "FAIL" for c in comparisons)
            else "WARN" if any(c["status"] == "WARN" for c in comparisons)
            else "PASS"
        ),
    }
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        ensure_dir(out)
        write_json(out / "glb_compare_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
