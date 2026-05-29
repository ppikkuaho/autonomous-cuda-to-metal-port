#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from _common import ROOT, collect_environment, ensure_dir, sha256_file, write_json


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def copy_result_file(value: Any, out_dir: Path, name_hint: str) -> list[Path]:
    copied: list[Path] = []
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            copied.extend(copy_result_file(item, out_dir, f"{name_hint}_{index}"))
        return copied
    if isinstance(value, dict):
        for key in ("path", "name", "url"):
            if key in value:
                copied.extend(copy_result_file(value[key], out_dir, f"{name_hint}_{key}"))
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple)):
                copied.extend(copy_result_file(item, out_dir, f"{name_hint}_{key}"))
        return copied
    if isinstance(value, str):
        path = Path(value)
        if path.exists() and path.is_file():
            suffix = path.suffix or ".bin"
            target = out_dir / f"{name_hint}{suffix}"
            if target.exists():
                target = out_dir / f"{name_hint}_{uuid.uuid4().hex[:8]}{suffix}"
            shutil.copy2(path, target)
            copied.append(target)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the public TencentARC/Pixal3D Hugging Face Space as an official visual reference.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--space", default="TencentARC/Pixal3D")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--tex-steps", type=int, default=12)
    parser.add_argument("--manual-fov", type=float, default=-1.0)
    parser.add_argument("--fov-unit", default="deg")
    parser.add_argument("--decimation-target", type=int, default=200_000)
    parser.add_argument("--texture-size", type=int, default=512)
    parser.add_argument("--hf-token-file", default=None)
    args = parser.parse_args()

    image = resolve(args.image)
    out = resolve(args.out)
    outputs = ensure_dir(out / "outputs")
    ensure_dir(out)

    started = time.time()
    session_id = f"codex-{int(started)}-{uuid.uuid4().hex[:8]}"
    hf_token = None
    if args.hf_token_file:
        token_file = resolve(args.hf_token_file)
        for line in reversed(token_file.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if line.startswith("hf_"):
                hf_token = line
                break
        if not hf_token:
            raise ValueError(f"No Hugging Face token found in {token_file}")

    manifest = {
        "type": "hf_space_reference",
        "space": args.space,
        "session_id": session_id,
        "input_image": str(image),
        "input_sha256": sha256_file(image),
        "args": {**vars(args), "hf_token_file": str(resolve(args.hf_token_file)) if args.hf_token_file else None},
        "auth": {"hf_token_supplied": hf_token is not None},
        "environment": collect_environment(),
    }
    write_json(out / "manifest.json", manifest)

    from gradio_client import Client, handle_file

    client = Client(args.space, token=hf_token, download_files=False)
    try:
        api_info = client.view_api(return_format="dict")
    except TypeError:
        api_info = client.view_api()
    write_json(out / "space_api.json", json_safe(api_info))

    generate_started = time.time()
    generate_result = client.predict(
        image=handle_file(str(image)),
        seed=args.seed,
        resolution=args.resolution,
        ss_guidance_strength=7.5,
        ss_guidance_rescale=0.7,
        ss_sampling_steps=args.ss_steps,
        ss_rescale_t=5.0,
        shape_slat_guidance_strength=7.5,
        shape_slat_guidance_rescale=0.5,
        shape_slat_sampling_steps=args.shape_steps,
        shape_slat_rescale_t=3.0,
        tex_slat_guidance_strength=1.0,
        tex_slat_guidance_rescale=0.0,
        tex_slat_sampling_steps=args.tex_steps,
        tex_slat_rescale_t=3.0,
        manual_fov=args.manual_fov,
        fov_unit=args.fov_unit,
        session_id=session_id,
        api_name="/generate_3d",
    )
    generate_duration = time.time() - generate_started
    write_json(out / "generate_result.json", json_safe(generate_result))

    extract_input = generate_result
    if isinstance(generate_result, dict):
        for key in ("state_path", "path", "name"):
            if key in generate_result:
                extract_input = generate_result[key]
                break

    extract_client = Client(args.space, token=hf_token, download_files=str(outputs))
    extract_started = time.time()
    extract_result = extract_client.predict(
        state_path=extract_input,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        session_id=session_id,
        api_name="/extract_glb_api",
    )
    extract_duration = time.time() - extract_started
    write_json(out / "extract_result.json", json_safe(extract_result))

    copied = copy_result_file(extract_result, outputs, "hf_space_output")
    copied.extend(copy_result_file(generate_result, outputs, "hf_space_generate"))

    result = {
        "status": "PASS" if copied else "PARTIAL",
        "duration_s": round(time.time() - started, 3),
        "generate_duration_s": round(generate_duration, 3),
        "extract_duration_s": round(extract_duration, 3),
        "copied_files": [str(path) for path in copied],
        "copied_sha256": {str(path): sha256_file(path) for path in copied if path.exists()},
    }
    write_json(out / "hf_reference_report.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if copied else 2


if __name__ == "__main__":
    raise SystemExit(main())
