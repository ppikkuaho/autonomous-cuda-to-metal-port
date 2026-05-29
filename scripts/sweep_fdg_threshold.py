#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

from _common import ROOT, ensure_dir


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def export_mesh(vertices: torch.Tensor, faces: torch.Tensor, output: Path) -> None:
    mesh = trimesh.Trimesh(
        vertices=vertices.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    )
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    mesh.apply_transform(rot)
    mesh.export(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay Pixal3D FDG mesh extraction with alternate edge thresholds.")
    parser.add_argument("fdg_decoder_pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid-size", type=int, default=1024)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[-20, -10, -5, -2, 0, 2, 5, 10])
    parser.add_argument("--work-pixal3d", default="work/Pixal3D")
    args = parser.parse_args()

    sys.path.insert(0, str(resolve(args.work_pixal3d)))
    from pixal3d.utils.mesh_extract import flexible_dual_grid_to_mesh

    data = torch.load(resolve(args.fdg_decoder_pt), map_location="cpu", weights_only=False)
    coords = data["coords"].to(torch.long)
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    dual_vertices = data["vertices"].float()
    logits = data["intersected_logits"].float()
    quad_lerp = data["quad_lerp"].float()

    out = resolve(args.out)
    ensure_dir(out)
    summaries = []
    for threshold in args.thresholds:
        flags = logits > threshold
        vertices, faces = flexible_dual_grid_to_mesh(
            coords,
            dual_vertices,
            flags,
            quad_lerp,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=args.grid_size,
            train=False,
        )
        label = f"thr_{threshold:g}".replace("-", "neg_").replace(".", "p")
        glb_path = out / f"{label}.glb"
        export_mesh(vertices, faces, glb_path)
        summaries.append({
            "threshold": threshold,
            "positive_fraction": float(flags.float().mean().item()),
            "vertices": int(vertices.shape[0]),
            "faces": int(faces.shape[0]),
            "glb": str(glb_path),
        })
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
