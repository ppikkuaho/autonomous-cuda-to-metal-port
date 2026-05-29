#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT, ensure_dir, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Hugging Face config/model access without downloading full weights.")
    parser.add_argument("--repo", default="TencentARC/Pixal3D")
    parser.add_argument("--out", default="artifacts/runs/m8a_hf_config_probe")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    ensure_dir(out)

    report: dict = {
        "repo": args.repo,
        "downloads": {},
        "model_access": {},
        "pipeline": {},
    }

    pipeline_path = hf_hub_download(args.repo, "pipeline.json")
    report["downloads"]["pipeline.json"] = pipeline_path
    pipeline = json.loads(Path(pipeline_path).read_text())
    (out / "pipeline.json").write_text(json.dumps(pipeline, indent=2) + "\n")

    args_obj = pipeline.get("args", {})
    report["pipeline"] = {
        "name": pipeline.get("name"),
        "default_pipeline_type": args_obj.get("default_pipeline_type"),
        "models": args_obj.get("models", {}),
        "image_cond_model": args_obj.get("image_cond_model", {}),
        "rembg_model": args_obj.get("rembg_model", {}),
        "sampler_steps": {
            "sparse_structure": args_obj.get("sparse_structure_sampler", {}).get("params", {}).get("steps"),
            "shape_slat": args_obj.get("shape_slat_sampler", {}).get("params", {}).get("steps"),
            "tex_slat": args_obj.get("tex_slat_sampler", {}).get("params", {}).get("steps"),
        },
    }

    candidates = []
    image_model = report["pipeline"]["image_cond_model"].get("args", {}).get("model_name")
    rembg_model = report["pipeline"]["rembg_model"].get("args", {}).get("model_name")
    for model_name in [image_model, rembg_model]:
        if model_name:
            candidates.append(model_name)

    for model_name in candidates:
        try:
            config_path = hf_hub_download(model_name, "config.json")
            report["model_access"][model_name] = {
                "status": "accessible",
                "config_path": config_path,
            }
        except Exception as exc:
            report["model_access"][model_name] = {
                "status": "blocked_or_missing",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    write_json(out / "manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

