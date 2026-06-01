#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from _common import ROOT, ensure_dir, write_json


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def first_texture_attr(report: dict[str, Any], attr: str) -> dict[str, Any] | None:
    for geom in report.get("textures", []):
        payload = geom.get("texture_attrs", {}).get(attr)
        if payload:
            return payload
    return None


def metric(stats: dict[str, Any] | None, name: str) -> float | None:
    if stats is None:
        return None
    value = stats.get(name)
    return float(value) if value is not None else None


def vector_metric(stats: dict[str, Any] | None, name: str) -> list[float] | None:
    if stats is None:
        return None
    value = stats.get(name)
    if value is None:
        return None
    return [float(item) for item in value]


def ratio(right: float | None, left: float | None) -> float | None:
    if right is None or left is None or abs(left) < 1e-12:
        return None
    return right / left


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def texture_fit(left_payload: dict[str, Any], right_payload: dict[str, Any], max_pixels: int = 500_000) -> dict[str, Any]:
    left_path = left_payload.get("path")
    right_path = right_payload.get("path")
    if not left_path or not right_path:
        return {"status": "SKIP", "reason": "missing extracted texture path"}
    left_image = Image.open(resolve(left_path)).convert("RGBA")
    right_image = Image.open(resolve(right_path)).convert("RGBA")
    if left_image.size != right_image.size:
        return {"status": "SKIP", "reason": f"size mismatch: {left_image.size} != {right_image.size}"}
    left_arr = np.asarray(left_image, dtype=np.float64) / 255.0
    right_arr = np.asarray(right_image, dtype=np.float64) / 255.0
    mask = (left_arr[..., 3] > 0.01) & (right_arr[..., 3] > 0.01)
    left_rgb = left_arr[..., :3][mask]
    right_rgb = right_arr[..., :3][mask]
    if left_rgb.shape[0] == 0:
        return {"status": "SKIP", "reason": "no shared nontransparent pixels"}
    if left_rgb.shape[0] > max_pixels:
        idx = np.linspace(0, left_rgb.shape[0] - 1, max_pixels).astype(np.int64)
        left_rgb = left_rgb[idx]
        right_rgb = right_rgb[idx]
        sampled = True
    else:
        sampled = False

    eps = 1e-6
    channel_reports = []
    for channel in range(3):
        x = left_rgb[:, channel]
        y = right_rgb[:, channel]
        A = np.stack([x, np.ones_like(x)], axis=1)
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        x_gamma = np.log(np.clip(x, eps, 1.0))
        y_gamma = np.log(np.clip(y, eps, 1.0))
        gamma, log_scale = np.linalg.lstsq(
            np.stack([x_gamma, np.ones_like(x_gamma)], axis=1),
            y_gamma,
            rcond=None,
        )[0]
        y_pred = slope * x + intercept
        y_gamma_pred = np.exp(log_scale) * np.power(np.clip(x, eps, 1.0), gamma)
        channel_reports.append(
            {
                "channel": channel,
                "affine_slope": float(slope),
                "affine_intercept": float(intercept),
                "affine_mae": float(np.mean(np.abs(y - y_pred))),
                "power_gamma": float(gamma),
                "power_scale": float(np.exp(log_scale)),
                "power_mae": float(np.mean(np.abs(y - y_gamma_pred))),
            }
        )

    left_lin = srgb_to_linear(left_rgb)
    right_lin = srgb_to_linear(right_rgb)
    return {
        "status": "PASS",
        "sampled": sampled,
        "pixels_compared": int(left_rgb.shape[0]),
        "stored_rgb_mae": float(np.mean(np.abs(left_rgb - right_rgb))),
        "linear_rgb_mae": float(np.mean(np.abs(left_lin - right_lin))),
        "stored_rgb_mean_left": left_rgb.mean(axis=0).tolist(),
        "stored_rgb_mean_right": right_rgb.mean(axis=0).tolist(),
        "linear_rgb_mean_left": left_lin.mean(axis=0).tolist(),
        "linear_rgb_mean_right": right_lin.mean(axis=0).tolist(),
        "channel_fit": channel_reports,
    }


def compare_attr(left: dict[str, Any], right: dict[str, Any], attr: str) -> dict[str, Any]:
    left_payload = first_texture_attr(left, attr)
    right_payload = first_texture_attr(right, attr)
    out: dict[str, Any] = {
        "attr": attr,
        "left_present": left_payload is not None,
        "right_present": right_payload is not None,
    }
    if not left_payload or not right_payload:
        return out

    left_stats = left_payload.get("stats", {})
    right_stats = right_payload.get("stats", {})
    out["left_size"] = left_stats.get("size")
    out["right_size"] = right_stats.get("size")
    out["metrics"] = {}
    for name in [
        "chroma_mean",
        "chroma_p95",
        "linear_chroma_mean",
        "linear_chroma_p95",
        "non_gray_fraction",
        "non_background_fraction",
        "alpha_mean",
        "alpha_nonzero_fraction",
    ]:
        lval = metric(left_stats, name)
        rval = metric(right_stats, name)
        out["metrics"][name] = {
            "left": lval,
            "right": rval,
            "right_over_left": ratio(rval, lval),
            "delta": None if lval is None or rval is None else rval - lval,
        }
    for name in ["rgb_mean", "rgb_std", "linear_rgb_mean", "linear_rgb_std"]:
        lvec = vector_metric(left_stats, name)
        rvec = vector_metric(right_stats, name)
        out["metrics"][name] = {
            "left": lvec,
            "right": rvec,
            "delta": None if lvec is None or rvec is None else [r - l for l, r in zip(lvec, rvec)],
        }
    if attr == "baseColorTexture":
        out["texture_fit"] = texture_fit(left_payload, right_payload)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare GLB texture inspection reports.")
    parser.add_argument("--left", required=True, help="Reference texture_report.json, usually CUDA.")
    parser.add_argument("--right", required=True, help="Candidate texture_report.json, usually Mac.")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--strict-visual",
        action="store_true",
        help="Add tighter warnings for visual parity diagnostics, not just non-collapsed texture validity.",
    )
    args = parser.parse_args()

    left = read_json(resolve(args.left))
    right = read_json(resolve(args.right))
    out_dir = resolve(args.out)
    ensure_dir(out_dir)

    attrs = ["baseColorTexture", "metallicRoughnessTexture", "normalTexture", "emissiveTexture", "occlusionTexture"]
    comparisons = [compare_attr(left, right, attr) for attr in attrs]
    base = next(item for item in comparisons if item["attr"] == "baseColorTexture")
    mr = next(item for item in comparisons if item["attr"] == "metallicRoughnessTexture")

    warnings: list[str] = []
    if not base.get("right_present"):
        warnings.append("candidate has no baseColorTexture")
    if not mr.get("right_present"):
        warnings.append("candidate has no metallicRoughnessTexture")
    if base.get("right_size") != base.get("left_size"):
        warnings.append(f"baseColorTexture size mismatch: {base.get('right_size')} vs {base.get('left_size')}")
    metrics = base.get("metrics", {})
    if metrics:
        if (metrics["non_gray_fraction"]["right"] or 0.0) < 0.8:
            warnings.append("candidate baseColorTexture non_gray_fraction below 0.8")
        if (metrics["chroma_mean"]["right"] or 0.0) < 0.12:
            warnings.append("candidate baseColorTexture chroma_mean below 0.12")
        if (metrics["non_background_fraction"]["right"] or 0.0) < 0.9:
            warnings.append("candidate baseColorTexture non_background_fraction below 0.9")
        rgb_delta = metrics.get("rgb_mean", {}).get("delta")
        if rgb_delta and max(abs(float(delta)) for delta in rgb_delta) > 0.15:
            warnings.append("candidate baseColorTexture rgb_mean differs from reference by >0.15 in at least one channel")
        if args.strict_visual:
            if rgb_delta and max(abs(float(delta)) for delta in rgb_delta) > 0.06:
                warnings.append("strict: candidate baseColorTexture rgb_mean differs from reference by >0.06 in at least one channel")
            chroma_ratio = metrics.get("chroma_mean", {}).get("right_over_left")
            if chroma_ratio is not None and not (0.95 <= float(chroma_ratio) <= 1.05):
                warnings.append(f"strict: candidate baseColorTexture chroma_mean ratio is {float(chroma_ratio):.3f}, outside [0.95, 1.05]")
            chroma_p95_ratio = metrics.get("chroma_p95", {}).get("right_over_left")
            if chroma_p95_ratio is not None and not (0.95 <= float(chroma_p95_ratio) <= 1.05):
                warnings.append(f"strict: candidate baseColorTexture chroma_p95 ratio is {float(chroma_p95_ratio):.3f}, outside [0.95, 1.05]")
            linear_chroma_ratio = metrics.get("linear_chroma_mean", {}).get("right_over_left")
            if linear_chroma_ratio is not None and not (0.95 <= float(linear_chroma_ratio) <= 1.05):
                warnings.append(f"strict: candidate baseColorTexture linear_chroma_mean ratio is {float(linear_chroma_ratio):.3f}, outside [0.95, 1.05]")
            linear_rgb_delta = metrics.get("linear_rgb_mean", {}).get("delta")
            if linear_rgb_delta and max(abs(float(delta)) for delta in linear_rgb_delta) > 0.04:
                warnings.append("strict: candidate baseColorTexture linear_rgb_mean differs from reference by >0.04 in at least one channel")

    mr_metrics = mr.get("metrics", {})
    if mr_metrics:
        chroma_delta = mr_metrics.get("chroma_mean", {}).get("delta")
        if chroma_delta is not None and abs(float(chroma_delta)) > 0.15:
            warnings.append("candidate metallicRoughnessTexture chroma_mean differs from reference by >0.15")
        if args.strict_visual:
            rgb_delta = mr_metrics.get("rgb_mean", {}).get("delta")
            if rgb_delta and max(abs(float(delta)) for delta in rgb_delta) > 0.04:
                warnings.append("strict: candidate metallicRoughnessTexture rgb_mean differs from reference by >0.04 in at least one channel")

    report = {
        "left": str(resolve(args.left)),
        "right": str(resolve(args.right)),
        "status": "PASS" if not warnings else "WARN",
        "warnings": warnings,
        "comparisons": comparisons,
    }
    write_json(out_dir / "texture_compare_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
