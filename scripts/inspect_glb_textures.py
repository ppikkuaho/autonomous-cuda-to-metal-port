#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from _common import ROOT, ensure_dir, write_json


TEXTURE_ATTRS = [
    "baseColorTexture",
    "metallicRoughnessTexture",
    "normalTexture",
    "emissiveTexture",
    "occlusionTexture",
]


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def image_stats(image: Image.Image) -> dict:
    arr = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    gray = rgb.mean(axis=2, keepdims=True)
    chroma = np.linalg.norm(rgb - gray, axis=2)
    linear_gray = linear_rgb.mean(axis=2, keepdims=True)
    linear_chroma = np.linalg.norm(linear_rgb - linear_gray, axis=2)
    non_bg = np.linalg.norm(rgb - np.array([184 / 255, 184 / 255, 184 / 255], dtype=np.float64), axis=2) > 0.04
    return {
        "mode": image.mode,
        "size": list(image.size),
        "rgb_min": rgb.reshape(-1, 3).min(axis=0).tolist(),
        "rgb_max": rgb.reshape(-1, 3).max(axis=0).tolist(),
        "rgb_mean": rgb.reshape(-1, 3).mean(axis=0).tolist(),
        "rgb_std": rgb.reshape(-1, 3).std(axis=0).tolist(),
        "alpha_mean": float(alpha.mean()),
        "alpha_nonzero_fraction": float((alpha > 0.01).mean()),
        "chroma_mean": float(chroma.mean()),
        "chroma_p95": float(np.percentile(chroma, 95)),
        "non_gray_fraction": float((chroma > 0.03).mean()),
        "non_background_fraction": float(non_bg.mean()),
        "linear_rgb_min": linear_rgb.reshape(-1, 3).min(axis=0).tolist(),
        "linear_rgb_max": linear_rgb.reshape(-1, 3).max(axis=0).tolist(),
        "linear_rgb_mean": linear_rgb.reshape(-1, 3).mean(axis=0).tolist(),
        "linear_rgb_std": linear_rgb.reshape(-1, 3).std(axis=0).tolist(),
        "linear_chroma_mean": float(linear_chroma.mean()),
        "linear_chroma_p95": float(np.percentile(linear_chroma, 95)),
    }


def save_preview(image: Image.Image, path: Path, max_size: int = 1024) -> None:
    preview = image.copy()
    preview.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    preview.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and optionally extract GLB texture images.")
    parser.add_argument("glb")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import trimesh

    glb = resolve(args.glb)
    out = resolve(args.out)
    ensure_dir(out)

    scene = trimesh.load(glb, force="scene")
    report = {
        "glb": str(glb),
        "geometry_count": len(scene.geometry),
        "textures": [],
    }

    for geom_name, geom in scene.geometry.items():
        visual = getattr(geom, "visual", None)
        material = getattr(visual, "material", None)
        uv = getattr(visual, "uv", None)
        geom_report = {
            "geometry": geom_name,
            "visual_type": type(visual).__name__ if visual is not None else None,
            "material_type": type(material).__name__ if material is not None else None,
            "uv_shape": list(uv.shape) if uv is not None else None,
            "texture_attrs": {},
        }
        if material is not None:
            for attr in TEXTURE_ATTRS:
                image = getattr(material, attr, None)
                if image is None:
                    geom_report["texture_attrs"][attr] = None
                    continue
                image = image.copy()
                image_path = out / f"{geom_name}_{attr}.png"
                preview_path = out / f"{geom_name}_{attr}_preview.png"
                image.save(image_path)
                save_preview(image, preview_path)
                geom_report["texture_attrs"][attr] = {
                    "path": str(image_path),
                    "preview_path": str(preview_path),
                    "stats": image_stats(image),
                }
        report["textures"].append(geom_report)

    write_json(out / "texture_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
