#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID_OVERRIDE="${RUN_ID:-}"
MAC_TARGET="${MAC_TARGET:-artifacts/runs/m12_9_robot_9_official_fov_4step}"
COMPARE_OUT="${COMPARE_OUT:-artifacts/runs/m13_cuda_reference_comparison}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT"

if [[ "$#" -lt 1 ]]; then
  echo "usage: scripts/m13_ingest_cuda_result.sh /path/to/m13_cuda_reference_*.tar.gz [...]" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: local ingest Python is missing or not executable: $PYTHON_BIN" >&2
  exit 2
fi

"$PYTHON_BIN" - <<'PY'
import trimesh  # noqa: F401
print("local_ingest_deps_ok")
PY

if [[ ! -d "$MAC_TARGET" ]]; then
  echo "ERROR: Mac comparison target not found: $MAC_TARGET" >&2
  exit 2
fi

for ARCHIVE in "$@"; do
  if [[ ! -f "$ARCHIVE" ]]; then
    echo "ERROR: archive not found: $ARCHIVE" >&2
    exit 2
  fi

  inferred_run_id="$(basename "$ARCHIVE")"
  inferred_run_id="${inferred_run_id%.tar.gz}"
  RUN_ID="${RUN_ID_OVERRIDE:-$inferred_run_id}"
  THIS_COMPARE_OUT="$COMPARE_OUT"
  if [[ "$#" -gt 1 ]]; then
    THIS_COMPARE_OUT="${COMPARE_OUT}/${RUN_ID}"
  fi

  echo "=== Ingesting ${ARCHIVE} as ${RUN_ID} ==="
  tar -xzf "$ARCHIVE"

  if [[ ! -f "artifacts/runs/${RUN_ID}/outputs/output.glb" ]]; then
    echo "ERROR: expected GLB missing: artifacts/runs/${RUN_ID}/outputs/output.glb" >&2
    exit 2
  fi

  "$PYTHON_BIN" scripts/validate_glb.py "artifacts/runs/${RUN_ID}/outputs/output.glb" --out "artifacts/runs/${RUN_ID}"

  if [[ -f "artifacts/runs/${RUN_ID}/debug/fdg_decoder.pt" ]]; then
    "$PYTHON_BIN" scripts/fdg_graph_metrics.py \
      "artifacts/runs/${RUN_ID}/debug/fdg_decoder.pt" \
      --out "artifacts/runs/${RUN_ID}/fdg_graph_metrics"
  fi

  "$PYTHON_BIN" scripts/compare_glb_artifacts.py \
    --left "$MAC_TARGET" \
    --right "artifacts/runs/${RUN_ID}" \
    --out "$THIS_COMPARE_OUT"

  echo "CUDA audit comparison wrote: $THIS_COMPARE_OUT"
done
