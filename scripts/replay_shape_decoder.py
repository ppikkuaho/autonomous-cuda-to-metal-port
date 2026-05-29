#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

from _common import ROOT, ensure_dir, sha256_file, write_json


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def export_geometry_glb(mesh, output_path: Path) -> None:
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    glb = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)
    ensure_dir(output_path.parent)
    glb.export(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay Pixal3D shape_slat decoder from a saved sparse latent.")
    parser.add_argument("--latent", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", default="TencentARC/Pixal3D")
    parser.add_argument("--work-pixal3d", default="work/Pixal3D")
    parser.add_argument("--return-subs", action="store_true", help="Keep decoder subdivision tensors; useful for debugging but memory-heavy.")
    parser.add_argument("--no-debug-artifacts", action="store_true", help="Do not dump full FDG decoder tensors during replay.")
    parser.add_argument("--load-all-models", action="store_true", help="Load the full Pixal3D pipeline instead of only the shape decoder.")
    args = parser.parse_args()

    work_pixal3d = resolve(args.work_pixal3d)
    sys.path.insert(0, str(work_pixal3d))
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
    os.environ.setdefault("PIXAL3D_MESH_CONVERT_BACKEND", "python")

    out = resolve(args.out)
    debug_dir = ensure_dir(out / "debug")
    outputs = ensure_dir(out / "outputs")
    if not args.no_debug_artifacts:
        os.environ["PIXAL3D_DEBUG_ARTIFACT_DIR"] = str(debug_dir)

    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline

    if not args.load_all_models:
        Pixal3DImageTo3DPipeline.model_names_to_load = ["shape_slat_decoder"]
        os.environ.setdefault("PIXAL3D_SKIP_REMBG_LOAD", "1")

    latent_path = resolve(args.latent)
    payload = torch.load(latent_path, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    slat = SparseTensor(
        coords=payload["coords"].to(device=device),
        feats=payload["feats"].to(device=device),
    )

    started = time.time()
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline._device = device
    pipeline.low_vram = False
    pipeline.models["shape_slat_decoder"].to(device)
    pipeline.models["shape_slat_decoder"].eval()

    decoded = pipeline.decode_shape_slat(slat, args.resolution, return_subs=args.return_subs)
    if args.return_subs:
        mesh_list, _subs = decoded
    else:
        mesh_list = decoded
    output_glb = outputs / "output.glb"
    export_geometry_glb(mesh_list[0], output_glb)

    report = {
        "status": "PASS",
        "latent": str(latent_path),
        "latent_sha256": sha256_file(latent_path),
        "resolution": args.resolution,
        "device": str(device),
        "duration_s": round(time.time() - started, 3),
        "output_glb": str(output_glb),
        "vertex_count": int(mesh_list[0].vertices.shape[0]),
        "face_count": int(mesh_list[0].faces.shape[0]),
    }
    write_json(out / "replay_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
