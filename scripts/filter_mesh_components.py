#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from _common import ROOT, ensure_dir


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep the largest connected mesh components for visual diagnostics.")
    parser.add_argument("glb")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--min-area-frac", type=float, default=0.0)
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()

    scene = trimesh.load(resolve(args.glb), force="scene")
    mesh = scene.dump(concatenate=True)
    components = mesh.split(only_watertight=False)
    if not components:
        raise SystemExit("no components found")

    areas = np.array([component.area for component in components], dtype=np.float64)
    total_area = float(areas.sum())
    order = np.argsort(-areas)
    keep = []
    for idx in order[: args.top]:
        if args.min_area_frac and areas[idx] / max(total_area, 1e-12) < args.min_area_frac:
            continue
        keep.append(components[idx])
    if not keep:
        keep = [components[int(order[0])]]

    filtered = trimesh.util.concatenate(keep)
    if args.repair:
        trimesh.repair.fill_holes(filtered)
        trimesh.repair.fix_normals(filtered)
        filtered.remove_unreferenced_vertices()

    out = resolve(args.out)
    ensure_dir(out.parent)
    filtered.export(out)
    print(
        {
            "input": str(resolve(args.glb)),
            "output": str(out),
            "component_count": len(components),
            "kept": len(keep),
            "total_area": total_area,
            "kept_area": float(sum(component.area for component in keep)),
            "vertices": int(len(filtered.vertices)),
            "faces": int(len(filtered.faces)),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
