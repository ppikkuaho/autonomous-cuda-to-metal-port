#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-m13_cuda_reference_robot_9_4step}"
OUT_DIR="${OUT_DIR:-artifacts/runs/${RUN_ID}}"
DEVICE="${DEVICE:-cuda}"
NAF_BACKEND="${PIXAL3D_NAF_BACKEND:-native}"
IMAGE_PATH="${IMAGE_PATH:-assets/images/9_img.png}"
MODEL_PATH="${MODEL_PATH:-TencentARC/Pixal3D}"
RUN_TARGET="${RUN_TARGET:-github_hfdemo}"
SEED="${SEED:-42}"
RESOLUTION="${RESOLUTION:-1024}"
FOV="${FOV:-0.3119156565218918}"
USE_MANUAL_FOV="${USE_MANUAL_FOV:-1}"
SS_STEPS="${SS_STEPS:-4}"
SHAPE_STEPS="${SHAPE_STEPS:-4}"
TEX_STEPS="${TEX_STEPS:-4}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-49152}"
MESH_CONVERT_BACKEND="${PIXAL3D_MESH_CONVERT_BACKEND:-o_voxel}"
TEXTURE="${TEXTURE:-0}"
LOCAL_NATTEN_DIR="$ROOT/work/Pixal3D/natten"
LOCAL_NATTEN_DISABLED_DIR="$ROOT/work/Pixal3D/.m13_cuda_disabled_natten"
RESTORE_LOCAL_NATTEN=0

cd "$ROOT"
mkdir -p "$OUT_DIR/outputs" "$OUT_DIR/debug"

if [[ "$RUN_TARGET" != "github_hfdemo" ]]; then
  echo "ERROR: unsupported RUN_TARGET=$RUN_TARGET. This script currently audits github_hfdemo only." >&2
  exit 2
fi

resolve_python_bin() {
  if [[ "$PYTHON_BIN" == /* ]]; then
    echo "$PYTHON_BIN"
    return 0
  fi
  if [[ -x "$PYTHON_BIN" ]]; then
    echo "$ROOT/$PYTHON_BIN"
    return 0
  fi
  command -v "$PYTHON_BIN"
}

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1"
  else
    shasum -a 256 "$1"
  fi
}

copy_root_evidence() {
  local file
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
      cp -f "$file" "$OUT_DIR/"
    fi
  done
}

prepare_cuda_import_path() {
  if [[ "$DEVICE" != cuda* ]]; then
    return 0
  fi
  if [[ ! -d "$LOCAL_NATTEN_DIR" ]]; then
    return 0
  fi
  if [[ ! -f "$LOCAL_NATTEN_DIR/__init__.py" ]] || ! grep -q "natten_mps" "$LOCAL_NATTEN_DIR/__init__.py"; then
    return 0
  fi
  if [[ -e "$LOCAL_NATTEN_DISABLED_DIR" ]]; then
    echo "ERROR: $LOCAL_NATTEN_DISABLED_DIR already exists; refusing to clobber local natten shim state." >&2
    exit 2
  fi
  mv "$LOCAL_NATTEN_DIR" "$LOCAL_NATTEN_DISABLED_DIR"
  RESTORE_LOCAL_NATTEN=1
  echo "Disabled local MPS natten shim for CUDA reference: $LOCAL_NATTEN_DIR -> $LOCAL_NATTEN_DISABLED_DIR"
}

restore_cuda_import_path() {
  if [[ "$RESTORE_LOCAL_NATTEN" == "1" && -d "$LOCAL_NATTEN_DISABLED_DIR" && ! -e "$LOCAL_NATTEN_DIR" ]]; then
    mv "$LOCAL_NATTEN_DISABLED_DIR" "$LOCAL_NATTEN_DIR"
    echo "Restored local MPS natten shim: $LOCAL_NATTEN_DIR"
  fi
}

on_exit() {
  local rc=$?
  restore_cuda_import_path || true
  copy_root_evidence || true
  return "$rc"
}
trap on_exit EXIT

prepare_cuda_import_path

PYTHON_BIN="$(resolve_python_bin)"
IMAGE_ABS="$ROOT/work/Pixal3D/$IMAGE_PATH"
if [[ ! -s "$IMAGE_ABS" ]]; then
  echo "ERROR: input image missing or empty: $IMAGE_ABS" >&2
  exit 2
fi
hash_file "$IMAGE_ABS" > "$OUT_DIR/input_image.sha256"

M13_RUN_ID="$RUN_ID" \
M13_OUT_DIR="$OUT_DIR" \
M13_DEVICE="$DEVICE" \
M13_NAF_BACKEND="$NAF_BACKEND" \
M13_IMAGE_PATH="$IMAGE_PATH" \
M13_MODEL_PATH="$MODEL_PATH" \
M13_RUN_TARGET="$RUN_TARGET" \
M13_SEED="$SEED" \
M13_RESOLUTION="$RESOLUTION" \
M13_FOV="$FOV" \
M13_USE_MANUAL_FOV="$USE_MANUAL_FOV" \
M13_SS_STEPS="$SS_STEPS" \
M13_SHAPE_STEPS="$SHAPE_STEPS" \
M13_TEX_STEPS="$TEX_STEPS" \
M13_MAX_NUM_TOKENS="$MAX_NUM_TOKENS" \
M13_MESH_CONVERT_BACKEND="$MESH_CONVERT_BACKEND" \
M13_TEXTURE="$TEXTURE" \
M13_ATTN_BACKEND="${ATTN_BACKEND:-flash_attn_3}" \
M13_SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-flash_attn_3}" \
M13_SPARSE_CONV_BACKEND="${SPARSE_CONV_BACKEND:-flex_gemm}" \
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

keys = [
    "M13_RUN_ID",
    "M13_DEVICE",
    "M13_NAF_BACKEND",
    "M13_IMAGE_PATH",
    "M13_MODEL_PATH",
    "M13_RUN_TARGET",
    "M13_SEED",
    "M13_RESOLUTION",
    "M13_FOV",
    "M13_USE_MANUAL_FOV",
    "M13_SS_STEPS",
    "M13_SHAPE_STEPS",
    "M13_TEX_STEPS",
    "M13_MAX_NUM_TOKENS",
    "M13_MESH_CONVERT_BACKEND",
    "M13_TEXTURE",
    "M13_ATTN_BACKEND",
    "M13_SPARSE_ATTN_BACKEND",
    "M13_SPARSE_CONV_BACKEND",
]
out = Path(os.environ["M13_OUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
(out / "run_config.json").write_text(
    json.dumps({key.removeprefix("M13_").lower(): os.environ.get(key) for key in keys}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

{
  date -u
  uname -a || true
  nvidia-smi || true
  "$PYTHON_BIN" --version || true
  "$PYTHON_BIN" -m pip freeze || true
} > "$OUT_DIR/env.txt" 2>&1

INFERENCE_ARGS=(
  inference.py
  --image "$IMAGE_PATH"
  --output "../../${OUT_DIR}/outputs/output.glb"
  --seed "$SEED"
  --low_vram
  --resolution "$RESOLUTION"
  --device "$DEVICE"
  --model_path "$MODEL_PATH"
  --ss_steps "$SS_STEPS"
  --shape_steps "$SHAPE_STEPS"
  --tex_steps "$TEX_STEPS"
  --max_num_tokens "$MAX_NUM_TOKENS"
)

if [[ "$USE_MANUAL_FOV" == "1" || "$USE_MANUAL_FOV" == "true" || "$USE_MANUAL_FOV" == "TRUE" ]]; then
  INFERENCE_ARGS+=(--fov "$FOV")
fi

if [[ "$TEXTURE" != "1" && "$TEXTURE" != "true" && "$TEXTURE" != "TRUE" ]]; then
  INFERENCE_ARGS+=(--no_texture)
fi

"$PYTHON_BIN" scripts/run_with_manifest.py \
  --out "$OUT_DIR" \
  --cwd work/Pixal3D \
  --env ATTN_BACKEND="${ATTN_BACKEND:-flash_attn_3}" \
  --env SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-flash_attn_3}" \
  --env SPARSE_CONV_BACKEND="${SPARSE_CONV_BACKEND:-flex_gemm}" \
  --env PIXAL3D_NAF_BACKEND="$NAF_BACKEND" \
  --env PIXAL3D_MESH_CONVERT_BACKEND="$MESH_CONVERT_BACKEND" \
  --env PIXAL3D_DEBUG_ARTIFACT_DIR="../../${OUT_DIR}/debug" \
  --env PIXAL3D_DEBUG_SAVE_SPARSE=1 \
  --env HF_HOME="${HF_HOME:-../../artifacts/hf_cache}" \
  -- "$PYTHON_BIN" "${INFERENCE_ARGS[@]}"

"$PYTHON_BIN" scripts/validate_glb.py "$OUT_DIR/outputs/output.glb" --out "$OUT_DIR"
"$PYTHON_BIN" scripts/fdg_graph_metrics.py "$OUT_DIR/debug/fdg_decoder.pt" --out "$OUT_DIR/fdg_graph_metrics" || true
"$PYTHON_BIN" scripts/render_turntable.py "$OUT_DIR/outputs/output.glb" --out "$OUT_DIR/render_turntable" --frames 4 || true
copy_root_evidence

echo "M13 CUDA reference wrote: $OUT_DIR"
