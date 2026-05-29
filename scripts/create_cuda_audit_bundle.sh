#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/cuda_audit_bundle}"
NAME="${NAME:-pixal3d-mac-port-m13}"

mkdir -p "$OUT_DIR"
cd "$ROOT"

tar \
  --exclude './artifacts' \
  --exclude './work/trellis-mac-baseline' \
  --exclude './**/__pycache__' \
  -czf "$OUT_DIR/${NAME}.tar.gz" \
  README.md pyproject.toml docs scripts tests work/Pixal3D upstream/Pixal3D

(cd "$OUT_DIR" && shasum -a 256 "${NAME}.tar.gz" > "${NAME}.sha256")
echo "$OUT_DIR/${NAME}.tar.gz"
