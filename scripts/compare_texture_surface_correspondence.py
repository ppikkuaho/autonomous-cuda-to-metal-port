#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from _common import ROOT, artifact_metadata, ensure_dir, write_json


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def chroma(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=1, keepdims=True)
    return np.linalg.norm(rgb - gray, axis=1)


def _first_textured_geom(scene: Any):
    for geom_name, geom in scene.geometry.items():
        if len(geom.vertices) and len(geom.faces) and getattr(geom.visual, "uv", None) is not None:
            return geom_name, geom
    raise ValueError("scene has no mesh geometry with UV coordinates")


def _material_image(material: Any, attr: str, mode: str) -> Image.Image | None:
    image = getattr(material, attr, None)
    if image is None and attr == "baseColorTexture":
        image = getattr(material, "image", None)
    if image is None:
        return None
    return image.convert(mode)


def _sample_texture(image: Image.Image, uv: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    u = np.mod(uv[:, 0], 1.0)
    v = 1.0 - np.mod(uv[:, 1], 1.0)
    x = np.clip(u * (w - 1), 0, w - 1)
    y = np.clip(v * (h - 1), 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    c00 = arr[y0, x0]
    c10 = arr[y0, x1]
    c01 = arr[y1, x0]
    c11 = arr[y1, x1]
    return (1 - wx) * (1 - wy) * c00 + wx * (1 - wy) * c10 + (1 - wx) * wy * c01 + wx * wy * c11


def _mesh_payload(path: Path) -> dict[str, Any]:
    import trimesh

    scene = trimesh.load(path, force="scene")
    geom_name, geom = _first_textured_geom(scene)
    material = geom.visual.material
    base = _material_image(material, "baseColorTexture", "RGBA")
    mr = _material_image(material, "metallicRoughnessTexture", "RGB")
    if base is None:
        raise ValueError(f"{path} has no baseColorTexture")
    vertices = np.asarray(geom.vertices, dtype=np.float32)
    faces = np.asarray(geom.faces, dtype=np.int64)
    uv = np.asarray(geom.visual.uv, dtype=np.float32)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return {
        "path": path,
        "scene": scene,
        "geom_name": geom_name,
        "geom": geom,
        "vertices": vertices,
        "faces": faces,
        "uv": uv,
        "base": base,
        "mr": mr,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_diag": float(np.linalg.norm(bbox_max - bbox_min)),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
    }


def _sample_surface(payload: dict[str, Any], count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = payload["vertices"]
    faces = payload["faces"]
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    positive = area > 0
    if not positive.any():
        raise ValueError(f"{payload['path']} has no positive-area triangles")
    area_weights = np.where(positive, area, 0.0).astype(np.float64)
    area_sum = float(area_weights.sum())
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(len(faces), size=count, replace=True, p=area_weights / area_sum)
    r1 = rng.random(count, dtype=np.float32)
    r2 = rng.random(count, dtype=np.float32)
    sr1 = np.sqrt(r1)
    bary = np.stack([1.0 - sr1, sr1 * (1.0 - r2), sr1 * r2], axis=1).astype(np.float32)
    points = np.einsum("ij,ijk->ik", bary, vertices[faces[face_ids]])
    return points.astype(np.float32), _uv_from_face_bary(payload, face_ids, bary)


def _uv_from_face_bary(payload: dict[str, Any], face_ids: np.ndarray, bary: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ijk->ik", bary, payload["uv"][payload["faces"][face_ids]]).astype(np.float32)


def _project_uv(payload: dict[str, Any], points: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    vertices = payload["vertices"]
    faces = payload["faces"]
    uv = payload["uv"]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    projected_uv = np.empty((len(points), 2), dtype=np.float32)
    distances = np.empty((len(points),), dtype=np.float32)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        closest, distance, triangle_id = trimesh.proximity.closest_point(mesh, points[start:end])
        triangles = vertices[faces[triangle_id]]
        bary = trimesh.triangles.points_to_barycentric(triangles, closest, method="cramer")
        projected_uv[start:end] = np.einsum("ij,ijk->ik", bary.astype(np.float32), uv[faces[triangle_id]])
        distances[start:end] = distance.astype(np.float32)
    return projected_uv, distances


def _summary_vector(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.shape[0]),
        "mean": values.mean(axis=0).tolist() if values.ndim == 2 else float(values.mean()),
        "p50": np.percentile(values, 50, axis=0).tolist() if values.ndim == 2 else float(np.percentile(values, 50)),
        "p95": np.percentile(values, 95, axis=0).tolist() if values.ndim == 2 else float(np.percentile(values, 95)),
        "p99": np.percentile(values, 99, axis=0).tolist() if values.ndim == 2 else float(np.percentile(values, 99)),
        "max": values.max(axis=0).tolist() if values.ndim == 2 else float(values.max()),
    }


def _delta_e(left_rgb: np.ndarray, right_rgb: np.ndarray) -> dict[str, Any] | None:
    try:
        from skimage.color import deltaE_ciede2000, rgb2lab
    except Exception:
        return None
    left_lab = rgb2lab(np.clip(left_rgb, 0.0, 1.0).reshape(1, -1, 3)).reshape(-1, 3)
    right_lab = rgb2lab(np.clip(right_rgb, 0.0, 1.0).reshape(1, -1, 3)).reshape(-1, 3)
    de = deltaE_ciede2000(left_lab, right_lab)
    return _summary_vector(de)


def _compare_samples(
    left_base: np.ndarray,
    right_base: np.ndarray,
    left_mr: np.ndarray | None,
    right_mr: np.ndarray | None,
    selected: np.ndarray,
) -> dict[str, Any]:
    if not selected.any():
        return {"count": 0}
    left_rgb = left_base[selected, :3].astype(np.float64)
    right_rgb = right_base[selected, :3].astype(np.float64)
    rgb_abs = np.abs(left_rgb - right_rgb)
    left_lin = srgb_to_linear(left_rgb)
    right_lin = srgb_to_linear(right_rgb)
    lin_abs = np.abs(left_lin - right_lin)
    left_chr = chroma(left_rgb)
    right_chr = chroma(right_rgb)
    out: dict[str, Any] = {
        "count": int(selected.sum()),
        "base_color": {
            "stored_rgb_mae": float(rgb_abs.mean()),
            "stored_rgb_rmse": float(np.sqrt(np.mean((left_rgb - right_rgb) ** 2))),
            "stored_rgb_p95_abs": float(np.percentile(rgb_abs, 95)),
            "stored_rgb_p99_abs": float(np.percentile(rgb_abs, 99)),
            "stored_rgb_mean_left": left_rgb.mean(axis=0).tolist(),
            "stored_rgb_mean_right": right_rgb.mean(axis=0).tolist(),
            "stored_rgb_mean_delta": (right_rgb.mean(axis=0) - left_rgb.mean(axis=0)).tolist(),
            "linear_rgb_mae": float(lin_abs.mean()),
            "linear_rgb_rmse": float(np.sqrt(np.mean((left_lin - right_lin) ** 2))),
            "linear_rgb_p95_abs": float(np.percentile(lin_abs, 95)),
            "linear_rgb_p99_abs": float(np.percentile(lin_abs, 99)),
            "linear_rgb_mean_left": left_lin.mean(axis=0).tolist(),
            "linear_rgb_mean_right": right_lin.mean(axis=0).tolist(),
            "linear_rgb_mean_delta": (right_lin.mean(axis=0) - left_lin.mean(axis=0)).tolist(),
            "chroma_mean_left": float(left_chr.mean()),
            "chroma_mean_right": float(right_chr.mean()),
            "chroma_mean_ratio": float(right_chr.mean() / left_chr.mean()) if left_chr.mean() > 1e-12 else None,
            "delta_e_ciede2000": _delta_e(left_rgb, right_rgb),
        },
    }
    if left_mr is not None and right_mr is not None:
        lmr = left_mr[selected, :3].astype(np.float64)
        rmr = right_mr[selected, :3].astype(np.float64)
        mr_abs = np.abs(lmr - rmr)
        out["metallic_roughness"] = {
            "stored_rgb_mae": float(mr_abs.mean()),
            "stored_rgb_rmse": float(np.sqrt(np.mean((lmr - rmr) ** 2))),
            "stored_rgb_p95_abs": float(np.percentile(mr_abs, 95)),
            "stored_rgb_p99_abs": float(np.percentile(mr_abs, 99)),
            "stored_rgb_mean_left": lmr.mean(axis=0).tolist(),
            "stored_rgb_mean_right": rmr.mean(axis=0).tolist(),
            "stored_rgb_mean_delta": (rmr.mean(axis=0) - lmr.mean(axis=0)).tolist(),
        }
    else:
        out["metallic_roughness"] = {
            "left_present": left_mr is not None,
            "right_present": right_mr is not None,
        }
    return out


def _thresholds(distances: np.ndarray, bbox_diag: float, rel_thresholds: list[float], abs_thresholds: list[float]) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = [("all_finite", math.inf)]
    for rel in rel_thresholds:
        values.append((f"rel_{rel:g}", float(rel) * bbox_diag))
    for abs_value in abs_thresholds:
        values.append((f"abs_{abs_value:g}", float(abs_value)))
    seen: set[str] = set()
    out: list[tuple[str, float]] = []
    for name, value in values:
        if name in seen:
            continue
        seen.add(name)
        out.append((name, value))
    return out


def _direction_report(
    name: str,
    source: dict[str, Any],
    target: dict[str, Any],
    sample_count: int,
    seed: int,
    chunk_size: int,
    rel_thresholds: list[float],
    abs_thresholds: list[float],
) -> dict[str, Any]:
    started = time.time()
    points, source_uv = _sample_surface(source, sample_count, seed)
    target_uv, distances = _project_uv(target, points, chunk_size)
    finite = np.isfinite(target_uv).all(axis=1) & np.isfinite(distances)

    source_base = _sample_texture(source["base"], source_uv)
    target_base = _sample_texture(target["base"], target_uv)
    source_mr = _sample_texture(source["mr"], source_uv) if source["mr"] is not None else None
    target_mr = _sample_texture(target["mr"], target_uv) if target["mr"] is not None else None

    thresholds = _thresholds(distances, max(source["bbox_diag"], target["bbox_diag"]), rel_thresholds, abs_thresholds)
    reports: dict[str, Any] = {}
    for label, value in thresholds:
        if math.isinf(value):
            selected = finite
        else:
            selected = finite & (distances <= value)
        report = _compare_samples(source_base, target_base, source_mr, target_mr, selected)
        report["projection_distance_limit"] = None if math.isinf(value) else float(value)
        report["selected_fraction"] = float(selected.mean())
        if selected.any():
            report["projection_distance"] = _summary_vector(distances[selected])
        reports[label] = report

    return {
        "name": name,
        "source": {
            "path": str(source["path"]),
            "geometry": source["geom_name"],
            "vertex_count": source["vertex_count"],
            "face_count": source["face_count"],
            "bbox_diag": source["bbox_diag"],
        },
        "target": {
            "path": str(target["path"]),
            "geometry": target["geom_name"],
            "vertex_count": target["vertex_count"],
            "face_count": target["face_count"],
            "bbox_diag": target["bbox_diag"],
        },
        "sample_count": int(sample_count),
        "seed": int(seed),
        "finite_projection_fraction": float(finite.mean()),
        "projection_distance_all_finite": _summary_vector(distances[finite]),
        "threshold_reports": reports,
        "duration_s": round(time.time() - started, 3),
    }


def compare(
    left_glb: Path,
    right_glb: Path,
    sample_count: int,
    seed: int,
    chunk_size: int,
    rel_thresholds: list[float],
    abs_thresholds: list[float],
) -> dict[str, Any]:
    left = _mesh_payload(left_glb)
    right = _mesh_payload(right_glb)
    return {
        "left_glb": artifact_metadata(left_glb),
        "right_glb": artifact_metadata(right_glb),
        "sample_count": int(sample_count),
        "seed": int(seed),
        "chunk_size": int(chunk_size),
        "relative_distance_thresholds": rel_thresholds,
        "absolute_distance_thresholds": abs_thresholds,
        "directions": [
            _direction_report("left_to_right", left, right, sample_count, seed, chunk_size, rel_thresholds, abs_thresholds),
            _direction_report("right_to_left", right, left, sample_count, seed + 1, chunk_size, rel_thresholds, abs_thresholds),
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Surface Texture Correspondence",
        "",
        f"Left: `{report['left_glb']['path']}`",
        f"Right: `{report['right_glb']['path']}`",
        f"Samples per direction: `{report['sample_count']}`",
        "",
        "| Direction | Threshold | Selected | Distance p95 | Base MAE | Linear MAE | DeltaE mean | Chroma ratio | MR MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for direction in report["directions"]:
        for threshold, item in direction["threshold_reports"].items():
            base = item.get("base_color", {})
            mr = item.get("metallic_roughness", {})
            de = (base.get("delta_e_ciede2000") or {}).get("mean")
            dist = (item.get("projection_distance") or {}).get("p95")
            lines.append(
                "| {direction} | `{threshold}` | {selected:.3f} | {dist} | {mae} | {lin_mae} | {de} | {chroma} | {mr_mae} |".format(
                    direction=direction["name"],
                    threshold=threshold,
                    selected=float(item.get("selected_fraction", 0.0)),
                    dist="n/a" if dist is None else f"{float(dist):.6g}",
                    mae="n/a" if base.get("stored_rgb_mae") is None else f"{float(base['stored_rgb_mae']):.6g}",
                    lin_mae="n/a" if base.get("linear_rgb_mae") is None else f"{float(base['linear_rgb_mae']):.6g}",
                    de="n/a" if de is None else f"{float(de):.6g}",
                    chroma="n/a" if base.get("chroma_mean_ratio") is None else f"{float(base['chroma_mean_ratio']):.6g}",
                    mr_mae="n/a" if mr.get("stored_rgb_mae") is None else f"{float(mr['stored_rgb_mae']):.6g}",
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two textured GLBs by sampled 3D surface correspondence instead of atlas pixel coordinate."
    )
    parser.add_argument("--left", required=True, help="Reference GLB, usually CUDA.")
    parser.add_argument("--right", required=True, help="Candidate GLB, usually Mac.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument(
        "--rel-threshold",
        type=float,
        action="append",
        default=[0.001, 0.0025, 0.005],
        help="Projection-distance threshold as a fraction of max bbox diagonal. Can repeat.",
    )
    parser.add_argument(
        "--abs-threshold",
        type=float,
        action="append",
        default=[],
        help="Projection-distance threshold in model units. Can repeat.",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(resolve(args.out))
    started = time.time()
    report = compare(
        resolve(args.left),
        resolve(args.right),
        int(args.samples),
        int(args.seed),
        int(args.chunk_size),
        [float(item) for item in args.rel_threshold],
        [float(item) for item in args.abs_threshold],
    )
    report["duration_s"] = round(time.time() - started, 3)
    write_json(out_dir / "surface_texture_correspondence.json", report)
    write_markdown(report, out_dir / "surface_texture_correspondence.md")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
