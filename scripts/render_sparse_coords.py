#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _common import ROOT, ensure_dir


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def render_projection(coords: np.ndarray, axes: tuple[int, int], size: int) -> Image.Image:
    pts = coords[:, list(axes)].astype(np.float32)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    xy = (pts - lo) / span
    pad = size * 0.08
    xy = xy * (size - 2 * pad) + pad
    xy[:, 1] = size - 1 - xy[:, 1]

    image = Image.new("RGB", (size, size), (248, 248, 246))
    draw = ImageDraw.Draw(image, "RGBA")
    for x, y in xy:
        draw.ellipse((x - 1.2, y - 1.2, x + 1.2, y + 1.2), fill=(25, 25, 25, 75))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Render sparse coordinate NPZ files as orthographic point-cloud views.")
    parser.add_argument("npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    npz = np.load(resolve(args.npz))
    coords = npz["coords"]
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    out = resolve(args.out)
    ensure_dir(out)

    views = {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
    }
    for name, axes in views.items():
        render_projection(coords, axes, args.size).save(out / f"{name}.png")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
