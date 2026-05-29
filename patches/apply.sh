#!/usr/bin/env bash
#
# Apply the Apple Silicon Mac-port patch onto a clean upstream Pixal3D checkout.
#
# The patch is generated against the pinned upstream commit recorded below. It
# touches 14 files (8 modified, 6 new) and contains only the Mac adaptation
# work; it does not vendor upstream code or model weights.
#
# Usage:
#   ./patches/apply.sh /path/to/Pixal3D
#
# where /path/to/Pixal3D is a fresh `git clone` of upstream Pixal3D checked out
# at the base commit (see PIXAL3D_BASE_COMMIT below).

set -euo pipefail

PIXAL3D_BASE_COMMIT="28efad66fdcbd8174a8538d9baf71fe34fe4b6d2"
PIXAL3D_UPSTREAM="https://github.com/TencentARC/Pixal3D"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/pixal3d-mac-port.patch"

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
  echo "usage: $0 /path/to/Pixal3D" >&2
  echo "" >&2
  echo "Clone and pin upstream first:" >&2
  echo "  git clone ${PIXAL3D_UPSTREAM}" >&2
  echo "  git -C Pixal3D checkout ${PIXAL3D_BASE_COMMIT}" >&2
  exit 2
fi

if [[ ! -d "${TARGET}/.git" ]]; then
  echo "error: ${TARGET} is not a git checkout" >&2
  exit 1
fi

HEAD_SHA="$(git -C "${TARGET}" rev-parse HEAD)"
if [[ "${HEAD_SHA}" != "${PIXAL3D_BASE_COMMIT}" ]]; then
  echo "warning: target is at ${HEAD_SHA}, patch base is ${PIXAL3D_BASE_COMMIT}." >&2
  echo "         The patch may not apply cleanly. Continuing in 3 seconds..." >&2
  sleep 3
fi

echo "Checking patch against ${TARGET} ..."
git -C "${TARGET}" apply --check "${PATCH_FILE}"
echo "Applying patch ..."
git -C "${TARGET}" apply "${PATCH_FILE}"
echo "Done. The Mac-port changes are now applied to ${TARGET}."
