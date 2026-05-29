#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _common import ROOT, ensure_dir


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def render(vertices: np.ndarray, faces: np.ndarray, angle: float, size: int, max_faces: int) -> Image.Image:
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int64)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    vertices = (vertices - center) @ rotation_y(angle).T
    extent = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-6).max()
    scale = size * 0.82 / extent
    xy = vertices[:, :2] * scale + np.array([size / 2, size / 2], dtype=np.float32)
    xy[:, 1] = size - 1 - xy[:, 1]

    tri3 = vertices[faces]
    normals = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
    normal_len = np.linalg.norm(normals, axis=1)
    valid = normal_len > 1e-8
    faces = faces[valid]
    normals = normals[valid] / normal_len[valid, None]
    tri3 = tri3[valid]

    if faces.shape[0] > max_faces:
        step = max(1, faces.shape[0] // max_faces)
        faces = faces[::step]
        normals = normals[::step]
        tri3 = tri3[::step]

    depth = tri3[:, :, 2].mean(axis=1)
    order = np.argsort(depth)
    light = np.array([0.25, -0.35, 0.9], dtype=np.float32)
    light = light / np.linalg.norm(light)
    shade = np.clip(0.25 + 0.75 * np.maximum(normals @ light, 0), 0.18, 1.0)

    image = Image.new("RGB", (size, size), (248, 248, 246))
    draw = ImageDraw.Draw(image, "RGBA")
    for idx in order:
        pts = xy[faces[idx]]
        if (pts[:, 0].max() < 0 or pts[:, 0].min() >= size or pts[:, 1].max() < 0 or pts[:, 1].min() >= size):
            continue
        gray = int(215 * shade[idx])
        draw.polygon([tuple(p) for p in pts], fill=(gray, gray, gray, 245))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic CPU shaded GLB preview renderer.")
    parser.add_argument("glb")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--max-faces", type=int, default=250_000)
    args = parser.parse_args()

    import trimesh

    glb = resolve(args.glb)
    out = resolve(args.out)
    ensure_dir(out)
    scene = trimesh.load(glb, force="scene")
    mesh = scene.dump(concatenate=True)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    for i in range(args.frames):
        angle = 2 * math.pi * i / args.frames
        image = render(vertices, faces, angle, args.size, args.max_faces)
        image.save(out / f"frame_{i:03d}.png")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
