#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from _common import ROOT, ensure_dir


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (248, 248, 246))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a labeled horizontal image contact sheet.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--item", action="append", required=True, help="Label=path. Repeatable.")
    parser.add_argument("--tile-width", type=int, default=460)
    parser.add_argument("--tile-height", type=int, default=420)
    parser.add_argument("--label-height", type=int, default=32)
    parser.add_argument("--padding", type=int, default=12)
    args = parser.parse_args()

    items: list[tuple[str, Path]] = []
    for value in args.item:
        if "=" not in value:
            raise ValueError(f"--item must be Label=path, got {value!r}")
        label, path = value.split("=", 1)
        items.append((label, resolve(path)))

    tile_size = (args.tile_width, args.tile_height)
    sheet_width = args.padding + len(items) * (args.tile_width + args.padding)
    sheet_height = args.padding + args.tile_height + args.label_height + args.padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (label, path) in enumerate(items):
        x = args.padding + index * (args.tile_width + args.padding)
        y = args.padding
        image = fit_image(Image.open(path), tile_size)
        sheet.paste(image, (x, y))
        draw.text((x, y + args.tile_height + 8), label, fill=(25, 25, 25), font=font)

    out = resolve(args.out)
    ensure_dir(out.parent)
    sheet.save(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
