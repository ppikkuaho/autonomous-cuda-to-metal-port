#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from _common import ROOT, ensure_dir, write_json


PATTERNS: list[tuple[str, str, str, str]] = [
    (r"\.cuda\s*\(", "hardcoded_cuda_device", ".cuda(", "Direct CUDA device move."),
    (r"torch\.cuda\.", "cuda_cache_or_memory", "torch.cuda", "CUDA runtime/cache/memory call."),
    (r"device\s*=\s*[\"']cuda[\"']", "hardcoded_cuda_device", "device=cuda", "Hardcoded CUDA device kwarg."),
    (r"to\s*\(\s*[\"']cuda[\"']\s*\)", "hardcoded_cuda_device", ".to(cuda)", "Hardcoded CUDA device move."),
    (r"PYTORCH_CUDA_ALLOC_CONF", "cuda_env", "PYTORCH_CUDA_ALLOC_CONF", "CUDA allocator env var."),
    (r"NATTEN_CUDA_ARCH", "cuda_extension_import", "NATTEN_CUDA_ARCH", "CUDA build env var."),
    (r"\bnatten\b", "cuda_extension_import", "natten", "NATTEN dependency."),
    (r"\bflash_attn\b|flash_attn_3|flash_attn_4", "attention_backend", "flash_attn", "FlashAttention dependency."),
    (r"\bxformers\b", "attention_backend", "xformers", "xFormers attention dependency."),
    (r"\bflex_gemm\b", "sparse_conv_backend", "flex_gemm", "Sparse conv/grid-sample backend."),
    (r"\bcumesh\b", "mesh_cleanup_backend", "cumesh", "Mesh cleanup backend."),
    (r"\bo_voxel\b", "mesh_extraction_backend", "o_voxel", "Voxel/mesh/GLB backend."),
    (r"\bnvdiffrast\b", "texture_baking_backend", "nvdiffrast", "CUDA rasterization backend."),
    (r"\bnvdiffrec\b|nvdiffrec_render", "texture_baking_backend", "nvdiffrec", "CUDA texture/render backend."),
    (r"ATTN_BACKEND|SPARSE_ATTN_BACKEND", "attention_backend", "attention env", "Attention backend switch."),
    (r"SPARSE_CONV_BACKEND", "sparse_conv_backend", "sparse conv env", "Sparse conv backend switch."),
    (r"FLEX_GEMM", "sparse_conv_backend", "FLEX_GEMM", "flex_gemm tuning env/config."),
    (r"grid_sample", "projection_math", "grid_sample", "Feature sampling/projection op."),
    (r"torch\.bmm|torch\.linalg|linalg\.inv", "projection_math", "linear algebra", "Projection/camera linear algebra."),
    (r"\bMoGe\b|\bmoge\b", "camera_estimation", "MoGe", "Camera estimation model/path."),
    (r"DinoV3|DINOv3|dinov3", "dino_feature_extraction", "DINOv3", "Image feature extractor."),
    (r"\bNAF\b|\bnaf\b", "naf_upsampling", "NAF", "NAF feature upsampler."),
]

IGNORE_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "artifacts"}
TEXT_SUFFIXES = {
    ".py",
    ".txt",
    ".md",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".sh",
    ".cfg",
    ".ini",
}


def iter_files(root: Path):
    for path in root.rglob("*"):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name in {"README", "LICENSE"}):
            yield path


def scan(root: Path) -> list[dict]:
    compiled = [(re.compile(pattern), category, symbol, risk) for pattern, category, symbol, risk in PATTERNS]
    rows: list[dict] = []
    for path in iter_files(root):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            for regex, category, symbol, risk in compiled:
                if regex.search(line):
                    rows.append(
                        {
                            "file": str(path.relative_to(root)),
                            "line": line_no,
                            "category": category,
                            "symbol": symbol,
                            "risk": risk,
                            "text": line.strip(),
                        }
                    )
    return rows


def write_outputs(rows: list[dict], out: Path) -> None:
    ensure_dir(out)
    (out / "cuda_refs.txt").write_text(
        "\n".join(
            f"{row['file']}:{row['line']}: [{row['category']}] {row['text']}" for row in rows
        )
        + ("\n" if rows else "")
    )
    with (out / "scan_results.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (out / "scan_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "line", "category", "symbol", "risk", "text"])
        writer.writeheader()
        writer.writerows(rows)

    high_risk = [r for r in rows if r["category"] not in {"projection_math", "dino_feature_extraction", "camera_estimation", "naf_upsampling"}]
    (out / "high_risk_ops.txt").write_text(
        "\n".join(
            f"{row['file']}:{row['line']}: [{row['category']}] {row['text']}" for row in high_risk
        )
        + ("\n" if high_risk else "")
    )

    counts = Counter(row["category"] for row in rows)
    write_json(
        out / "scan_summary.json",
        {
            "total_matches": len(rows),
            "by_category": dict(sorted(counts.items())),
            "high_risk_matches": len(high_risk),
        },
    )

    matrix = [
        "# Compatibility Matrix Seed",
        "",
        "| Original dependency / assumption | Location | Purpose | Mac strategy | Test | Status | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    seen = set()
    for row in rows:
        key = (row["symbol"], row["category"])
        if key in seen:
            continue
        seen.add(key)
        matrix.append(
            f"| `{row['symbol']}` | `{row['file']}:{row['line']}` | {row['risk']} | TBD | TBD | planned | {row['category']} |"
        )
    (out / "compatibility_matrix_seed.md").write_text("\n".join(matrix) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan a source tree for CUDA/backend assumptions.")
    parser.add_argument("target", help="Source tree to scan.")
    parser.add_argument("--out", default="artifacts/runs/m0_static_scan", help="Output artifact directory.")
    parser.add_argument("--fail-unguarded", action="store_true", help="Return nonzero if high-risk CUDA/backend matches exist.")
    args = parser.parse_args()

    target = Path(args.target)
    if not target.is_absolute():
        target = ROOT / target
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    rows = scan(target)
    write_outputs(rows, out)
    summary = json.loads((out / "scan_summary.json").read_text())
    print(f"scanned={target}")
    print(f"wrote={out}")
    print(f"total_matches={summary['total_matches']}")
    print(f"high_risk_matches={summary['high_risk_matches']}")
    for category, count in summary["by_category"].items():
        print(f"{category}={count}")
    if args.fail_unguarded and summary["high_risk_matches"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

