#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _common import ROOT, ensure_dir, write_json
from compare_sparse_artifacts import row_keys


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def load_sparse(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "coords" not in payload or "feats" not in payload:
        raise ValueError(f"expected sparse artifact with coords/feats: {path}")
    return {
        "path": str(path),
        "stage": payload.get("stage"),
        "name": payload.get("name"),
        "coords": payload["coords"].detach().cpu().to(torch.long),
        "feats": payload["feats"].detach().cpu().to(torch.float32),
    }


def stats(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float64)
    flat = arr.reshape(-1)
    finite = np.isfinite(flat)
    out: dict[str, Any] = {
        "shape": list(arr.shape),
        "numel": int(arr.size),
        "nan_count": int(np.isnan(flat).sum()),
        "inf_count": int(np.isinf(flat).sum()),
    }
    if finite.any():
        data = flat[finite]
        out.update(
            {
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "p01": float(np.percentile(data, 1)),
                "p50": float(np.percentile(data, 50)),
                "p99": float(np.percentile(data, 99)),
            }
        )
    return out


def rgb_stats(rgb: np.ndarray) -> dict[str, Any]:
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    if rgb.size == 0:
        return {"numel": 0}
    flat = rgb.reshape(-1, 3)
    gray = flat.mean(axis=1, keepdims=True)
    chroma = np.linalg.norm(flat - gray, axis=1)
    luma = flat @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    return {
        "numel": int(flat.shape[0]),
        "rgb_mean": flat.mean(axis=0).tolist(),
        "rgb_std": flat.std(axis=0).tolist(),
        "rgb_min": flat.min(axis=0).tolist(),
        "rgb_max": flat.max(axis=0).tolist(),
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "chroma_mean": float(chroma.mean()),
        "chroma_p95": float(np.percentile(chroma, 95)),
        "non_gray_fraction": float((chroma > 0.03).mean()),
        "near_white_fraction": float((flat.mean(axis=1) > 0.96).mean()),
        "near_black_fraction": float((flat.mean(axis=1) < 0.04).mean()),
    }


def affine_channels(left: np.ndarray, right: np.ndarray) -> list[dict[str, Any]]:
    x = np.asarray(left, dtype=np.float64).reshape(-1, left.shape[-1])
    y = np.asarray(right, dtype=np.float64).reshape(-1, right.shape[-1])
    rows = []
    for channel in range(x.shape[1]):
        xc = x[:, channel]
        yc = y[:, channel]
        var = float(np.var(xc))
        if var <= 1e-12:
            slope = 0.0
            intercept = float(yc.mean())
        else:
            slope = float(np.cov(xc, yc, bias=True)[0, 1] / var)
            intercept = float(yc.mean() - slope * xc.mean())
        pred = slope * xc + intercept
        ss_res = float(((yc - pred) ** 2).sum())
        ss_tot = float(((yc - yc.mean()) ** 2).sum())
        rows.append(
            {
                "channel": channel,
                "slope": slope,
                "intercept": intercept,
                "mae": float(np.abs(yc - xc).mean()),
                "fit_mae": float(np.abs(pred - yc).mean()),
                "rmse": float(np.sqrt(np.mean((yc - xc) ** 2))),
                "r2": None if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot),
            }
        )
    return rows


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(rgb)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear @ matrix.T
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz = xyz / white
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    lab = np.empty_like(f)
    lab[:, 0] = 116 * f[:, 1] - 16
    lab[:, 1] = 500 * (f[:, 0] - f[:, 1])
    lab[:, 2] = 200 * (f[:, 1] - f[:, 2])
    return lab


def color_delta(left_rgb: np.ndarray, right_rgb: np.ndarray, max_rows: int) -> dict[str, Any]:
    left = np.clip(np.asarray(left_rgb, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
    right = np.clip(np.asarray(right_rgb, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
    if left.shape[0] > max_rows:
        idx = np.linspace(0, left.shape[0] - 1, max_rows, dtype=np.int64)
        left = left[idx]
        right = right[idx]
        sampled = True
    else:
        sampled = False
    delta_rgb = right - left
    left_gray = left.mean(axis=1, keepdims=True)
    right_gray = right.mean(axis=1, keepdims=True)
    left_chroma = np.linalg.norm(left - left_gray, axis=1)
    right_chroma = np.linalg.norm(right - right_gray, axis=1)
    left_lab = rgb_to_lab(left)
    right_lab = rgb_to_lab(right)
    delta_e = np.linalg.norm(right_lab - left_lab, axis=1)
    return {
        "rows": int(left.shape[0]),
        "sampled": sampled,
        "rgb_mean_delta": delta_rgb.mean(axis=0).tolist(),
        "rgb_mae": np.abs(delta_rgb).mean(axis=0).tolist(),
        "rgb_rmse": np.sqrt((delta_rgb**2).mean(axis=0)).tolist(),
        "chroma_mean_left": float(left_chroma.mean()),
        "chroma_mean_right": float(right_chroma.mean()),
        "chroma_mean_ratio": float(right_chroma.mean() / left_chroma.mean()) if left_chroma.mean() > 1e-12 else None,
        "delta_e_mean": float(delta_e.mean()),
        "delta_e_p50": float(np.percentile(delta_e, 50)),
        "delta_e_p95": float(np.percentile(delta_e, 95)),
        "affine_right_from_left": affine_channels(left, right),
    }


def align(left: dict[str, Any], right: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    left_coords = left["coords"]
    right_coords = right["coords"]
    if left_coords.ndim != 2 or right_coords.ndim != 2 or left_coords.shape[1] != right_coords.shape[1]:
        raise ValueError(f"coordinate shape mismatch: {tuple(left_coords.shape)} vs {tuple(right_coords.shape)}")
    shared, left_idx, right_idx = np.intersect1d(row_keys(left_coords), row_keys(right_coords), return_indices=True)
    union_count = left_coords.shape[0] + right_coords.shape[0] - len(shared)
    report = {
        "shared_count": int(len(shared)),
        "left_count": int(left_coords.shape[0]),
        "right_count": int(right_coords.shape[0]),
        "left_only_count": int(left_coords.shape[0] - len(shared)),
        "right_only_count": int(right_coords.shape[0] - len(shared)),
        "jaccard": float(len(shared) / union_count) if union_count else 1.0,
        "left_coverage": float(len(shared) / left_coords.shape[0]) if left_coords.shape[0] else 1.0,
        "right_coverage": float(len(shared) / right_coords.shape[0]) if right_coords.shape[0] else 1.0,
    }
    return left["feats"][torch.from_numpy(left_idx).long()], right["feats"][torch.from_numpy(right_idx).long()], report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare aligned sparse PBR/color tensor artifacts.")
    parser.add_argument("--left", required=True, help="Reference sparse .pt, usually CUDA.")
    parser.add_argument("--right", required=True, help="Candidate sparse .pt, usually Mac.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-color-rows", type=int, default=500_000)
    args = parser.parse_args()

    left = load_sparse(resolve(args.left))
    right = load_sparse(resolve(args.right))
    left_feats, right_feats, overlap = align(left, right)
    if left_feats.shape[1] != right_feats.shape[1]:
        raise ValueError(f"feature channel mismatch: {left_feats.shape[1]} vs {right_feats.shape[1]}")

    left_np = left_feats.numpy()
    right_np = right_feats.numpy()
    diff = right_np - left_np
    report: dict[str, Any] = {
        "left": left["path"],
        "right": right["path"],
        "left_stage": left.get("stage"),
        "right_stage": right.get("stage"),
        "left_name": left.get("name"),
        "right_name": right.get("name"),
        "coordinate_overlap": overlap,
        "channels": int(left_np.shape[1]),
        "all_channels": {
            "left": stats(left_np),
            "right": stats(right_np),
            "delta": stats(diff),
            "mae": float(np.abs(diff).mean()) if diff.size else None,
            "rmse": float(np.sqrt(np.mean(diff**2))) if diff.size else None,
        },
    }
    if left_np.shape[1] >= 3:
        report["base_color"] = {
            "left": rgb_stats(left_np[:, :3]),
            "right": rgb_stats(right_np[:, :3]),
            "delta": color_delta(left_np[:, :3], right_np[:, :3], args.max_color_rows),
        }
    if left_np.shape[1] >= 4:
        report["metallic"] = {
            "left": stats(np.clip(left_np[:, 3:4], 0, 1)),
            "right": stats(np.clip(right_np[:, 3:4], 0, 1)),
            "delta": stats(np.clip(right_np[:, 3:4], 0, 1) - np.clip(left_np[:, 3:4], 0, 1)),
        }
    if left_np.shape[1] >= 5:
        report["roughness"] = {
            "left": stats(np.clip(left_np[:, 4:5], 0, 1)),
            "right": stats(np.clip(right_np[:, 4:5], 0, 1)),
            "delta": stats(np.clip(right_np[:, 4:5], 0, 1) - np.clip(left_np[:, 4:5], 0, 1)),
        }
    if left_np.shape[1] >= 6:
        report["alpha"] = {
            "left": stats(np.clip(left_np[:, 5:6], 0, 1)),
            "right": stats(np.clip(right_np[:, 5:6], 0, 1)),
            "delta": stats(np.clip(right_np[:, 5:6], 0, 1) - np.clip(left_np[:, 5:6], 0, 1)),
        }

    out = resolve(args.out)
    ensure_dir(out if out.suffix == "" else out.parent)
    write_json(out / "sparse_pbr_color_compare.json" if out.suffix == "" else out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
