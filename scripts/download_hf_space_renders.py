#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote

import httpx

from _common import ROOT, ensure_dir, sha256_file, write_json


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def read_token(path: Path | None) -> str | None:
    if path is None:
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line.startswith("hf_"):
            return line
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Download render frames referenced by a Pixal3D HF Space generate_result.json.")
    parser.add_argument("generate_result")
    parser.add_argument("--out", required=True)
    parser.add_argument("--space-url", default="https://tencentarc-pixal3d.hf.space")
    parser.add_argument("--hf-token-file", default=None)
    parser.add_argument("--kinds", nargs="+", default=["clay", "base_color", "normal", "shaded_forest"])
    parser.add_argument("--frames", type=int, default=2)
    args = parser.parse_args()

    generate_result = json.loads(resolve(args.generate_result).read_text(encoding="utf-8"))
    out = resolve(args.out)
    ensure_dir(out)
    token = read_token(resolve(args.hf_token_file) if args.hf_token_file else None)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    downloaded = []
    errors = []
    with httpx.Client(headers=headers, follow_redirects=True, timeout=60) as client:
        for kind in args.kinds:
            frames = generate_result.get("render_paths", {}).get(kind, [])[: args.frames]
            kind_dir = ensure_dir(out / kind)
            for index, frame in enumerate(frames):
                remote_path = frame.get("path") if isinstance(frame, dict) else None
                if not remote_path:
                    continue
                url = f"{args.space_url}/gradio_api/file={quote(remote_path)}"
                target = kind_dir / f"frame_{index:03d}{Path(remote_path).suffix or '.jpg'}"
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    target.write_bytes(response.content)
                    downloaded.append({
                        "kind": kind,
                        "index": index,
                        "remote_path": remote_path,
                        "local_path": str(target),
                        "size_bytes": target.stat().st_size,
                        "sha256": sha256_file(target),
                    })
                except Exception as exc:
                    errors.append({
                        "kind": kind,
                        "index": index,
                        "remote_path": remote_path,
                        "error": repr(exc),
                    })

    report = {
        "status": "PASS" if downloaded else "FAIL",
        "downloaded": downloaded,
        "errors": errors,
        "auth": {"hf_token_supplied": token is not None},
    }
    write_json(out / "download_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if downloaded else 1


if __name__ == "__main__":
    raise SystemExit(main())
