from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_command(args: Iterable[str], cwd: Path | str | None = None, timeout: int = 30) -> dict[str, Any]:
    started = time.time()
    args = list(args)
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "duration_s": round(time.time() - started, 6),
        }
    except FileNotFoundError as exc:
        return {
            "cmd": args,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "duration_s": round(time.time() - started, 6),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": args,
            "returncode": 124,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else f"timed out after {timeout}s",
            "duration_s": round(time.time() - started, 6),
        }


def command_stdout(args: Iterable[str], cwd: Path | str | None = None, timeout: int = 30) -> str | None:
    result = run_command(args, cwd=cwd, timeout=timeout)
    if result["returncode"] == 0:
        return result["stdout"]
    return None


def git_commit(path: Path | str) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    return command_stdout(["git", "rev-parse", "HEAD"], cwd=path)


def git_status(path: Path | str) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    return command_stdout(["git", "status", "--short"], cwd=path)


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_metadata(path: Path | str) -> dict[str, Any]:
    """Stable file identity used to prove reports describe the same GLB bytes."""
    p = Path(path)
    stat = p.stat()
    return {
        "path": str(p.expanduser().resolve(strict=False)),
        "exists": p.exists(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(p),
    }


def write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def collect_environment() -> dict[str, Any]:
    torch_info: dict[str, Any]
    try:
        import torch

        torch_info = {
            "available": True,
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "mps_built": bool(torch.backends.mps.is_built()),
        }
    except Exception as exc:
        torch_info = {"available": False, "error": repr(exc)}

    python_arch = platform.machine()
    macos_version = command_stdout(["sw_vers", "-productVersion"], timeout=5)
    chip = command_stdout(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5)

    return {
        "timestamp_unix": time.time(),
        "project_root": str(ROOT),
        "machine": {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": python_arch,
            "processor": platform.processor(),
            "chip": chip,
            "macos": macos_version,
        },
        "software": {
            "python": sys.version,
            "python_executable": sys.executable,
            "python_arch": python_arch,
            "nvidia_smi": run_command(["nvidia-smi"], timeout=10),
            "torch": torch_info,
            "xcodebuild": run_command(["xcodebuild", "-version"], timeout=10),
            "clang": run_command(["clang", "--version"], timeout=10),
            "metal_path": shutil.which("metal") or command_stdout(["xcrun", "--find", "metal"], timeout=10),
        },
        "device": {
            "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "mps_high_watermark_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
            "mps_low_watermark_ratio": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
        },
        "repos": {
            "upstream_pixal3d_commit": git_commit(ROOT / "upstream" / "Pixal3D"),
            "upstream_trellis_mac_commit": git_commit(ROOT / "upstream" / "trellis-mac"),
            "work_pixal3d_commit": git_commit(ROOT / "work" / "Pixal3D"),
            "work_pixal3d_status": git_status(ROOT / "work" / "Pixal3D"),
        },
    }


def tensor_summary(name: str, tensor: Any, sample_size: int = 4096) -> dict[str, Any]:
    try:
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"expected torch.Tensor, got {type(tensor)!r}")
        detached = tensor.detach()
        flat = detached.reshape(-1)
        finite = torch.isfinite(flat)
        sample = flat[: min(sample_size, flat.numel())].to("cpu", dtype=torch.float32)
        sample_hash = hashlib.sha256(sample.numpy().tobytes()).hexdigest() if sample.numel() else None
        numeric = detached.to("cpu", dtype=torch.float32)
        return {
            "name": name,
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "min": float(numeric.min().item()) if numeric.numel() else None,
            "max": float(numeric.max().item()) if numeric.numel() else None,
            "mean": float(numeric.mean().item()) if numeric.numel() else None,
            "std": float(numeric.std(unbiased=False).item()) if numeric.numel() else None,
            "nan_count": int(torch.isnan(flat).sum().item()),
            "inf_count": int(torch.isinf(flat).sum().item()),
            "finite_fraction": float(finite.to(torch.float32).mean().item()) if flat.numel() else 1.0,
            "zero_fraction": float((flat == 0).to(torch.float32).mean().item()) if flat.numel() else 0.0,
            "sha256_sample": sample_hash,
        }
    except Exception as exc:
        return {"name": name, "error": repr(exc)}
