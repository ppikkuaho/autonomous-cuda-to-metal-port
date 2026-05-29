#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-cuda/bin/python}"
RUN_TEXTURED="${RUN_TEXTURED:-1}"

cd "$ROOT"

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1"
  else
    shasum -a 256 "$1"
  fi
}

run_case() {
  local run_id="$1"
  local ss_steps="$2"
  local shape_steps="$3"
  local tex_steps="$4"
  local texture="$5"
  local image_path="${6:-assets/images/9_img.png}"
  local use_manual_fov="${7:-1}"

  echo "=== M13 case: ${run_id} image=${image_path} ss=${ss_steps} shape=${shape_steps} tex=${tex_steps} texture=${texture} use_manual_fov=${use_manual_fov} ==="
  set +e
  RUN_ID="$run_id" \
  SS_STEPS="$ss_steps" \
  SHAPE_STEPS="$shape_steps" \
  TEX_STEPS="$tex_steps" \
  TEXTURE="$texture" \
  IMAGE_PATH="$image_path" \
  USE_MANUAL_FOV="$use_manual_fov" \
  PYTHON_BIN="$PYTHON_BIN" \
    bash scripts/m13_cuda_reference.sh
  local rc=$?
  set -e

  if [[ -d "artifacts/runs/${run_id}" ]]; then
    tar -czf "artifacts/runs/${run_id}.tar.gz" "artifacts/runs/${run_id}"
    hash_file "artifacts/runs/${run_id}.tar.gz" > "artifacts/runs/${run_id}.sha256"
  fi
  return "$rc"
}

# Exact Mac comparison target: the cleanest diagnostic for backend parity.
run_case "m13_cuda_reference_robot_9_4step" 4 4 4 0

# Higher-step geometry sanity check: run in the same warmed pod so model cache
# and environment setup are not paid twice.
run_case "m13_cuda_reference_robot_9_12step" 12 12 4 0

# Known repo sample: separates a broken environment from a hard project input.
run_case "m13_cuda_reference_sample_0_12step_auto_camera" 12 12 4 0 "assets/images/0_img.png" 0

# Texture is a portfolio artifact and export-path check, not the primary
# sampling diagnostic. Run it only after geometry variants in the same session.
if [[ "$RUN_TEXTURED" == "1" || "$RUN_TEXTURED" == "true" || "$RUN_TEXTURED" == "TRUE" ]]; then
  run_case "m13_cuda_reference_robot_9_12step_textured" 12 12 12 1
fi

echo "M13 sequence complete. Packaged archives:"
ls -lh artifacts/runs/m13_cuda_reference_*.tar.gz
