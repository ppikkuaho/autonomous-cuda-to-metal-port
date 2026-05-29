#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY310="${PY310:-}"
MIN_DISK_GB="${MIN_DISK_GB:-100}"
REQUIRE_HOPPER="${REQUIRE_HOPPER:-1}"
INPUT_IMAGE="${INPUT_IMAGE:-work/Pixal3D/assets/images/9_img.png}"

cd "$ROOT"

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

echo "=== M13 RunPod preflight ==="
date -u
uname -a || true

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Set it before installing/downloading models." >&2
  exit 2
fi

if [[ ! -s "$INPUT_IMAGE" ]]; then
  echo "ERROR: input image missing or empty: $INPUT_IMAGE" >&2
  exit 2
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$INPUT_IMAGE"
else
  shasum -a 256 "$INPUT_IMAGE"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This is not a CUDA-ready environment." >&2
  exit 2
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found. requirements-hfdemo.txt installs MoGe from git." >&2
  exit 2
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "WARNING: tmux not found. Install it or use nohup so long setup is not tied to an SSH session." >&2
fi

nvidia-smi
GPU_QUERY="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true)"
echo "GPU_QUERY=${GPU_QUERY}"

if [[ "$REQUIRE_HOPPER" == "1" || "$REQUIRE_HOPPER" == "true" || "$REQUIRE_HOPPER" == "TRUE" ]]; then
  if ! echo "$GPU_QUERY" | grep -Eiq 'H100|H200|H800|B100|B200'; then
    echo "ERROR: REQUIRE_HOPPER=1 but GPU does not look H-series/Blackwell: ${GPU_QUERY}" >&2
    echo "Use REQUIRE_HOPPER=0 only for an explicitly labeled fallback diagnostic." >&2
    exit 3
  fi
fi

PY310_RESOLVED="$(find_python310 || true)"
if [[ -z "$PY310_RESOLVED" ]]; then
  echo "ERROR: Python 3.10 not found. Choose a Python 3.10 image or install python3.10/python3.10-venv before proceeding." >&2
  exit 2
fi
PY310="$PY310_RESOLVED"

"$PY310" --version
"$PY310" -m venv --help >/dev/null

AVAIL_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
AVAIL_GB="$((AVAIL_KB / 1024 / 1024))"
echo "available_disk_gb=${AVAIL_GB}"
if (( AVAIL_GB < MIN_DISK_GB )); then
  echo "ERROR: available disk ${AVAIL_GB} GB is below MIN_DISK_GB=${MIN_DISK_GB}." >&2
  exit 2
fi

if command -v curl >/dev/null 2>&1; then
  curl -L -I --max-time 10 https://huggingface.co >/dev/null
  curl -L -I --max-time 10 https://github.com >/dev/null
fi

echo "python310=$PY310"
echo "Preflight PASS."
