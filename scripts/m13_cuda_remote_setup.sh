#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY310="${PY310:-}"
VENV="${VENV:-$ROOT/.venv-cuda}"
RUN_ID="${RUN_ID:-m13_cuda_reference_robot_9_4step}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.cache/pip}"
HF_HOME="${HF_HOME:-$ROOT/artifacts/hf_cache}"
BUNDLE_SHA_FILE="${BUNDLE_SHA_FILE:-$ROOT/../pixal3d-mac-port-m13.sha256}"
RUN_TARGET="${RUN_TARGET:-github_hfdemo}"
MODEL_REPO="${MODEL_REPO:-TencentARC/Pixal3D}"

cd "$ROOT"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" artifacts/runs
export PIP_CACHE_DIR HF_HOME PIP_DISABLE_PIP_VERSION_CHECK=1 MODEL_REPO
export MODEL_PATH="${MODEL_PATH:-$MODEL_REPO}"

exec > >(tee -a "$ROOT/m13_remote_setup.log") 2>&1
echo "=== M13 remote setup log started ==="
date -u
echo "RUN_TARGET=$RUN_TARGET"
echo "MODEL_REPO=$MODEL_REPO"

if [[ "$RUN_TARGET" != "github_hfdemo" ]]; then
  echo "ERROR: unsupported RUN_TARGET=$RUN_TARGET. Current bundle is prepared for github_hfdemo." >&2
  exit 2
fi

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1"
  else
    shasum -a 256 "$1"
  fi
}

write_source_snapshot() {
  {
    date -u
    echo "root=$ROOT"
    for repo in . work/Pixal3D upstream/Pixal3D; do
      echo "=== ${repo} ==="
      if git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "$repo" rev-parse HEAD || true
        git -C "$repo" status --short || true
      else
        echo "not a git worktree"
      fi
    done
  } > "$ROOT/m13_source_snapshot.txt" 2>&1
  git -C "$ROOT/work/Pixal3D" diff -- > "$ROOT/m13_work_pixal3d.diff" 2>/dev/null || true
  if [[ -f "$BUNDLE_SHA_FILE" ]]; then
    cp -f "$BUNDLE_SHA_FILE" "$ROOT/m13_bundle.sha256"
  fi
}

package_failure() {
  local rc=$?
  if [[ "$rc" -eq 0 ]]; then
    return 0
  fi
  set +e
  local fail_id="m13_setup_failure_$(date -u +%Y%m%dT%H%M%SZ)"
  local fail_dir="artifacts/runs/${fail_id}"
  mkdir -p "$fail_dir"
  for file in \
    m13_remote_setup.log \
    m13_cuda_preflight.txt \
    m13_cuda_import_check.txt \
    m13_optional_import_failures.txt \
    m13_pip_check.txt \
    m13_pip_freeze.txt \
    m13_hf_access_check.txt \
    m13_hf_snapshots.json \
    m13_cuda_runtime_smoke.txt \
    m13_source_snapshot.txt \
    m13_work_pixal3d.diff \
    m13_bundle.sha256; do
    if [[ -f "$file" ]]; then
      cp -f "$file" "$fail_dir/"
    fi
  done
  tar -czf "${fail_dir}.tar.gz" "$fail_dir"
  hash_file "${fail_dir}.tar.gz" > "${fail_dir}.sha256"
  echo "M13 setup failed with rc=${rc}; packaged failure evidence: ${fail_dir}.tar.gz" >&2
  return "$rc"
}
trap package_failure EXIT

find_python310() {
  local candidates=()
  if [[ -n "$PY310" ]]; then
    candidates+=("$PY310")
  fi
  candidates+=(python3.10 python3 python)

  local candidate
  for candidate in "${candidates[@]}"; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY
    then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN must be set so gated DINO/RMBG/Pixal3D weights can download." >&2
  exit 2
fi 
RUN_TEXTURED_WAS_SET="${RUN_TEXTURED+x}"

bash scripts/m13_runpod_preflight.sh
write_source_snapshot
PY310_RESOLVED="$(find_python310 || true)"
if [[ -z "$PY310_RESOLVED" ]]; then
  echo "ERROR: Python 3.10 is required because Pixal3D's CUDA wheels are cp310." >&2
  echo "Use a CUDA 12.4 / Python 3.10 image, or install python3.10 and python3.10-venv first." >&2
  exit 2
fi
PY310="$PY310_RESOLVED"

{
  date -u
  uname -a || true
  nvidia-smi || true
  df -h . || true
  echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
  echo "HF_HOME=$HF_HOME"
  "$PY310" --version || true
} > "m13_cuda_preflight.txt" 2>&1

echo "=== Creating venv: $VENV ==="
time "$PY310" -m venv "$VENV"
echo "=== Installing pip bootstrap packages ==="
time "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
echo "=== Installing Pixal3D CUDA requirements ==="
time "$VENV/bin/python" -m pip install -r work/Pixal3D/requirements-hfdemo.txt
"$VENV/bin/python" -m pip check | tee "m13_pip_check.txt"
"$VENV/bin/python" -m pip freeze > "m13_pip_freeze.txt"

"$VENV/bin/python" - <<'PY' | tee "m13_cuda_import_check.txt"
import importlib
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
print("cuda_device", torch.cuda.get_device_name(0))
print("cuda_capability", torch.cuda.get_device_capability(0))
critical_modules = [
    "natten",
    "flash_attn_interface",
    "flex_gemm",
    "o_voxel",
]
optional_modules = [
    "cumesh",
    "nvdiffrast.torch",
    "nvdiffrec_render",
]
critical_failures = []
optional_failures = []
for module in critical_modules:
    try:
        importlib.import_module(module)
        print("import_ok", module)
    except Exception as exc:
        print("import_fail", module, repr(exc))
        critical_failures.append((module, repr(exc)))
for module in optional_modules:
    try:
        importlib.import_module(module)
        print("optional_import_ok", module)
    except Exception as exc:
        print("optional_import_fail", module, repr(exc))
        optional_failures.append((module, repr(exc)))
if critical_failures:
    raise SystemExit(f"Critical CUDA import check failed: {critical_failures}")
if optional_failures:
    print("WARNING optional CUDA import failures:", optional_failures)
    with open("m13_optional_import_failures.txt", "w", encoding="utf-8") as f:
        for module, error in optional_failures:
            f.write(f"{module}\t{error}\n")
PY

if [[ -s "m13_optional_import_failures.txt" && -z "$RUN_TEXTURED_WAS_SET" ]]; then
  echo "WARNING: optional texture/render imports failed; defaulting RUN_TEXTURED=0 for RUN_SEQUENCE to avoid wasting paid time." >&2
  echo "Set RUN_TEXTURED=1 explicitly to attempt the textured run anyway." >&2
  export RUN_TEXTURED=0
fi

"$VENV/bin/python" - <<'PY' | tee "m13_cuda_runtime_smoke.txt"
import torch

torch.manual_seed(42)
a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
b = torch.randn((512, 512), device="cuda", dtype=torch.float16)
c = a @ b
torch.cuda.synchronize()
print("cuda_runtime_smoke_ok", float(c.float().mean().item()), torch.cuda.max_memory_allocated())
PY

"$VENV/bin/python" - <<'PY' | tee "m13_hf_access_check.txt"
import json
import os
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

checks = [
    (os.environ.get("MODEL_REPO", "TencentARC/Pixal3D"), "pipeline.json"),
    ("camenduru/dinov3-vitl16-pretrain-lvd1689m", "config.json"),
]
snapshot_report = []
for repo_id, filename in checks:
    path = hf_hub_download(repo_id, filename)
    print("hf_ok", repo_id, filename, path)
    if os.environ.get("MATERIALIZE_HF_SNAPSHOTS", "1") not in {"0", "false", "FALSE"}:
        snapshot_path = snapshot_download(repo_id)
        snapshot_report.append({"repo_id": repo_id, "snapshot_path": snapshot_path})
        print("snapshot_ok", repo_id, snapshot_path)
with open("m13_hf_snapshots.json", "w", encoding="utf-8") as f:
    json.dump(snapshot_report, f, indent=2, sort_keys=True)
PY

if [[ "${RUN_SEQUENCE:-0}" == "1" || "${RUN_SEQUENCE:-0}" == "true" || "${RUN_SEQUENCE:-0}" == "TRUE" ]]; then
  PYTHON_BIN="$VENV/bin/python" bash scripts/m13_cuda_hour_sequence.sh
  echo "CUDA audit sequence artifacts: artifacts/runs/m13_cuda_reference_*.tar.gz"
else
  PYTHON_BIN="$VENV/bin/python" bash scripts/m13_cuda_reference.sh
  tar -czf "artifacts/runs/${RUN_ID}.tar.gz" "artifacts/runs/${RUN_ID}"
  hash_file "artifacts/runs/${RUN_ID}.tar.gz" > "artifacts/runs/${RUN_ID}.sha256"
  echo "CUDA audit artifacts: artifacts/runs/${RUN_ID}.tar.gz"
fi
