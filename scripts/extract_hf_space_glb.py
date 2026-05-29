#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Extract a GLB from an existing Pixal3D HF Space state path.")
    parser.add_argument("--generate-result", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--space", default="TencentARC/Pixal3D")
    parser.add_argument("--hf-token-file", default=None)
    parser.add_argument("--decimation-target", type=int, default=50_000)
    parser.add_argument("--texture-size", type=int, default=256)
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()

    generate_result = json.loads(resolve(args.generate_result).read_text(encoding="utf-8"))
    state_path = generate_result["state_path"]
    out = resolve(args.out)
    outputs = ensure_dir(out / "outputs")
    token = read_token(resolve(args.hf_token_file) if args.hf_token_file else None)

    from gradio_client import Client

    client = Client(args.space, token=token, download_files=str(outputs))
    result = client.predict(
        state_path=state_path,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        session_id=args.session_id,
        api_name="/extract_glb_api",
    )
    copied = []
    if isinstance(result, str) and Path(result).exists():
        src = Path(result)
        dst = outputs / (src.name if src.suffix else "output.glb")
        if src != dst:
            shutil.copy2(src, dst)
        copied.append(dst)
    elif isinstance(result, dict):
        for value in result.values():
            if isinstance(value, str) and Path(value).exists():
                src = Path(value)
                dst = outputs / (src.name if src.suffix else "output.glb")
                shutil.copy2(src, dst)
                copied.append(dst)

    report = {
        "status": "PASS" if copied else "PARTIAL",
        "state_path": state_path,
        "decimation_target": args.decimation_target,
        "texture_size": args.texture_size,
        "result": repr(result),
        "copied_files": [str(path) for path in copied],
        "copied_sha256": {str(path): sha256_file(path) for path in copied},
        "auth": {"hf_token_supplied": token is not None},
    }
    write_json(out / "extract_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if copied else 2


if __name__ == "__main__":
    raise SystemExit(main())
