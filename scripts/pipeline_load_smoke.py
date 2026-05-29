#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from _common import ROOT, collect_environment, ensure_dir, write_json


WORK_PIXAL3D = ROOT / "work" / "Pixal3D"


def classify_error(exc: BaseException) -> str:
    text = "\n".join([repr(exc), traceback.format_exc()])
    access_needles = [
        "401 Client Error",
        "GatedRepoError",
        "gated repo",
        "Repo model",
        "Unauthorized",
        "Cannot access gated repo",
    ]
    if any(needle.lower() in text.lower() for needle in access_needles):
        return "external_hf_access_gate"
    cuda_needles = ["cuda", "flash_attn", "flex_gemm", "o_voxel._C", "nvdiffrast", "cumesh"]
    if any(needle.lower() in text.lower() for needle in cuda_needles):
        return "cuda_backend_dependency"
    return "runtime_error"


def module_device(module) -> str | None:
    try:
        param = next(module.parameters())
        return str(param.device)
    except StopIteration:
        return "no_parameters"
    except Exception:
        return None


def module_summary(module) -> dict:
    result = {
        "type": type(module).__name__,
        "device": module_device(module),
    }
    try:
        result["parameter_count"] = int(sum(p.numel() for p in module.parameters()))
    except Exception as exc:
        result["parameter_count_error"] = repr(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Load Pixal3D pipeline on Mac without running sampling.")
    parser.add_argument("--model", default="TencentARC/Pixal3D")
    parser.add_argument("--device", default=None)
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--skip-image-cond", action="store_true")
    parser.add_argument("--allow-external-gate", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    ensure_dir(out)

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    sys.path.insert(0, str(WORK_PIXAL3D))

    started = time.time()
    report = {
        "status": "FAIL",
        "stage": "start",
        "model": args.model,
        "device_requested": args.device,
        "low_vram": args.low_vram,
        "skip_image_cond": args.skip_image_cond,
        "environment": collect_environment(),
        "backend_env": {
            "ATTN_BACKEND": os.environ.get("ATTN_BACKEND"),
            "SPARSE_ATTN_BACKEND": os.environ.get("SPARSE_ATTN_BACKEND"),
            "SPARSE_CONV_BACKEND": os.environ.get("SPARSE_CONV_BACKEND"),
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        },
        "timings": [],
    }

    try:
        report["stage"] = "import"
        import torch
        from pixal3d.pipelines import Pixal3DImageTo3DPipeline
        from pixal3d.utils.device import resolve_device

        device = resolve_device(args.device)
        report["device_resolved"] = str(device)
        report["torch"] = {
            "version": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
            "mps_built": bool(torch.backends.mps.is_built()),
            "cuda_available": bool(torch.cuda.is_available()),
        }

        report["stage"] = "pipeline_from_pretrained"
        t0 = time.time()
        pipeline = Pixal3DImageTo3DPipeline.from_pretrained(args.model)
        report["timings"].append({"stage": "pipeline_from_pretrained", "duration_s": round(time.time() - t0, 6)})

        pipeline.low_vram = args.low_vram
        pipeline._device = device

        report["models"] = {name: module_summary(module) for name, module in pipeline.models.items()}
        report["samplers"] = {
            "sparse_structure_sampler": type(pipeline.sparse_structure_sampler).__name__,
            "shape_slat_sampler": type(pipeline.shape_slat_sampler).__name__,
            "tex_slat_sampler": type(pipeline.tex_slat_sampler).__name__,
        }
        report["default_pipeline_type"] = getattr(pipeline, "default_pipeline_type", None)
        report["low_vram_after_load"] = bool(getattr(pipeline, "low_vram", False))

        if not args.skip_image_cond:
            report["stage"] = "image_conditioning_models"
            from inference import IMAGE_COND_CONFIGS, build_image_cond_model

            cond = {}
            for name, config in IMAGE_COND_CONFIGS.items():
                t0 = time.time()
                module = build_image_cond_model(config)
                cond[name] = {
                    **module_summary(module),
                    "config": config,
                    "duration_s": round(time.time() - t0, 6),
                }
            report["image_conditioning_models"] = cond

        report["stage"] = "complete"
        report["status"] = "PASS"
        return_code = 0
    except Exception as exc:
        category = classify_error(exc)
        report["status"] = "PARTIAL" if category == "external_hf_access_gate" and args.allow_external_gate else "FAIL"
        report["error_category"] = category
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        return_code = 0 if report["status"] == "PARTIAL" else 1
    finally:
        report["duration_s"] = round(time.time() - started, 6)
        write_json(out / "pipeline_load_report.json", report)
        print(json.dumps({
            "status": report["status"],
            "stage": report["stage"],
            "error_category": report.get("error_category"),
            "out": str(out),
        }, indent=2, sort_keys=True))

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
