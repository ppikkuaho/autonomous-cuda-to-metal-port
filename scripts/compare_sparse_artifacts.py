#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _common import ROOT, ensure_dir, write_json


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def load_artifact(path: Path) -> dict[str, Any]:
    if path.suffix == ".npz":
        data = np.load(path)
        coords = torch.from_numpy(data["coords"]).to(torch.long)
        return {"path": str(path), "coords": coords, "tensors": {}}

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "coords" not in payload:
        raise ValueError(f"unsupported sparse artifact: {path}")

    tensors = {}
    for key in ("feats", "raw_feats", "intersected_logits", "quad_lerp", "vertices"):
        value = payload.get(key)
        if isinstance(value, torch.Tensor) and value.shape[:1] == payload["coords"].shape[:1]:
            tensors[key] = value.detach().cpu()

    return {
        "path": str(path),
        "stage": payload.get("stage"),
        "name": payload.get("name"),
        "coords": payload["coords"].detach().cpu().to(torch.long),
        "tensors": tensors,
    }


def coords_report(coords: torch.Tensor) -> dict[str, Any]:
    coords_np = coords.cpu().numpy()
    report: dict[str, Any] = {
        "count": int(coords_np.shape[0]),
        "dims": int(coords_np.shape[1]) if coords_np.ndim == 2 else None,
    }
    if coords_np.size:
        report["min"] = coords_np.min(axis=0).astype(int).tolist()
        report["max"] = coords_np.max(axis=0).astype(int).tolist()
    return report


def row_keys(coords: torch.Tensor) -> np.ndarray:
    arr = np.ascontiguousarray(coords.cpu().numpy())
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))).reshape(-1)


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach().cpu().to(torch.float32)
    flat = t.reshape(-1)
    return {
        "shape": list(t.shape),
        "dtype": str(tensor.dtype),
        "min": float(flat.min().item()) if flat.numel() else None,
        "max": float(flat.max().item()) if flat.numel() else None,
        "mean": float(flat.mean().item()) if flat.numel() else None,
        "std": float(flat.std(unbiased=False).item()) if flat.numel() else None,
        "nan_count": int(torch.isnan(flat).sum().item()),
        "inf_count": int(torch.isinf(flat).sum().item()),
        "zero_fraction": float((flat == 0).to(torch.float32).mean().item()) if flat.numel() else 0.0,
    }


def aligned_tensor_compare(left: torch.Tensor, right: torch.Tensor, max_shared: int) -> dict[str, Any]:
    left_f = left.to(torch.float32)
    right_f = right.to(torch.float32)
    if left_f.shape[0] > max_shared:
        idx = torch.linspace(0, left_f.shape[0] - 1, max_shared, dtype=torch.long)
        left_f = left_f[idx]
        right_f = right_f[idx]
        sampled = True
    else:
        sampled = False

    diff = left_f - right_f
    report = {
        "shared_rows_compared": int(left_f.shape[0]),
        "sampled": sampled,
        "mae": float(diff.abs().mean().item()) if diff.numel() else None,
        "rmse": float(torch.sqrt((diff * diff).mean()).item()) if diff.numel() else None,
        "max_abs": float(diff.abs().max().item()) if diff.numel() else None,
    }
    if left_f.ndim == 2 and left_f.shape[1] == right_f.shape[1] and left_f.numel():
        denom = left_f.norm(dim=1) * right_f.norm(dim=1)
        valid = denom > 1e-12
        if bool(valid.any()):
            cosine = (left_f[valid] * right_f[valid]).sum(dim=1) / denom[valid]
            report["mean_cosine"] = float(cosine.mean().item())
            report["median_cosine"] = float(cosine.median().item())
            report["valid_cosine_rows"] = int(valid.sum().item())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare sparse coordinate/tensor artifacts saved by Pixal3D debug mode.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-shared", type=int, default=200_000)
    args = parser.parse_args()

    left = load_artifact(resolve(args.left))
    right = load_artifact(resolve(args.right))
    left_coords = left["coords"]
    right_coords = right["coords"]

    if left_coords.ndim != 2 or right_coords.ndim != 2 or left_coords.shape[1] != right_coords.shape[1]:
        raise ValueError(f"coordinate shape mismatch: {tuple(left_coords.shape)} vs {tuple(right_coords.shape)}")

    left_keys = row_keys(left_coords)
    right_keys = row_keys(right_coords)
    shared, left_idx, right_idx = np.intersect1d(left_keys, right_keys, return_indices=True)
    union_count = len(left_keys) + len(right_keys) - len(shared)

    report: dict[str, Any] = {
        "left_path": str(resolve(args.left)),
        "right_path": str(resolve(args.right)),
        "left": {
            "stage": left.get("stage"),
            "name": left.get("name"),
            "coords": coords_report(left_coords),
            "tensors": {name: tensor_stats(tensor) for name, tensor in left["tensors"].items()},
        },
        "right": {
            "stage": right.get("stage"),
            "name": right.get("name"),
            "coords": coords_report(right_coords),
            "tensors": {name: tensor_stats(tensor) for name, tensor in right["tensors"].items()},
        },
        "coordinate_overlap": {
            "shared_count": int(len(shared)),
            "left_only_count": int(len(left_keys) - len(shared)),
            "right_only_count": int(len(right_keys) - len(shared)),
            "union_count": int(union_count),
            "jaccard": float(len(shared) / union_count) if union_count else 1.0,
            "left_coverage": float(len(shared) / len(left_keys)) if len(left_keys) else 1.0,
            "right_coverage": float(len(shared) / len(right_keys)) if len(right_keys) else 1.0,
        },
        "aligned_tensors": {},
    }

    for key in sorted(set(left["tensors"]) & set(right["tensors"])):
        left_tensor = left["tensors"][key][torch.from_numpy(left_idx).long()]
        right_tensor = right["tensors"][key][torch.from_numpy(right_idx).long()]
        if left_tensor.shape[1:] != right_tensor.shape[1:]:
            report["aligned_tensors"][key] = {
                "status": "SKIP",
                "reason": f"shape mismatch after shared coord alignment: {list(left_tensor.shape)} vs {list(right_tensor.shape)}",
            }
            continue
        report["aligned_tensors"][key] = aligned_tensor_compare(left_tensor, right_tensor, args.max_shared)

    if args.out:
        out = resolve(args.out)
        ensure_dir(out)
        write_json(out / "sparse_compare_report.json", report)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
