#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from _common import ROOT, ensure_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Best-effort GLB turntable renderer using trimesh if available.")
    parser.add_argument("glb")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()

    try:
        import trimesh
    except Exception as exc:
        print(f"trimesh unavailable: {exc!r}")
        return 2

    glb = Path(args.glb)
    if not glb.is_absolute():
        glb = ROOT / glb
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    ensure_dir(out)

    scene = trimesh.load(glb, force="scene")
    for i in range(args.frames):
        angle = 2 * math.pi * i / args.frames
        scene.camera_transform = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        try:
            png = scene.save_image(resolution=(1024, 1024), visible=True)
        except Exception as exc:
            print(f"trimesh preview unavailable ({exc!r}); using point-cloud fallback")
            mesh = scene.dump(concatenate=True)
            _write_pointcloud_turntable(mesh.vertices, out, args.frames)
            print(f"wrote {out}")
            return 0
        (out / f"frame_{i:03d}.png").write_bytes(png)
    print(f"wrote {out}")
    return 0


def _write_pointcloud_turntable(vertices, out: Path, frames: int) -> None:
    import numpy as np
    from PIL import Image

    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.shape[0] > 160_000:
        step = max(1, vertices.shape[0] // 160_000)
        vertices = vertices[::step]

    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    vertices = vertices - center
    extent = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-6).max()
    size = 1024
    scale = size * 0.82 / extent

    for i in range(frames):
        angle = 2 * math.pi * i / frames
        c, s = math.cos(angle), math.sin(angle)
        rot = vertices @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32).T
        xy = rot[:, :2] * scale + np.array([size / 2, size / 2], dtype=np.float32)
        depth = rot[:, 2]
        order = np.argsort(depth)
        xy = xy[order].astype(np.int32)
        depth = depth[order]
        depth_norm = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-6)

        image = Image.new("RGB", (size, size), (248, 248, 246))
        pixels = image.load()
        for (x, y), d in zip(xy, depth_norm):
            if 0 <= x < size and 0 <= y < size:
                shade = int(55 + 165 * d)
                pixels[x, size - 1 - y] = (shade, shade, shade)
        image.save(out / f"frame_{i:03d}.png")


if __name__ == "__main__":
    raise SystemExit(main())
