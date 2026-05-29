# Reproducibility

This document records the minimum commands and artifacts needed to reproduce the geometry claim.

## Target Hardware

Mac:

- Apple Silicon Mac, tested on M4 MacBook Pro 16 inch with 48 GB unified memory.
- Python 3.11 virtual environment.
- PyTorch with MPS support.

CUDA reference:

- A rented H100 cloud GPU pod.
- Python 3.10.
- PyTorch `2.6.0+cu124`.
- Critical CUDA imports: `natten`, `flash_attn_interface`, `flex_gemm`, `o_voxel`.

## Frozen Target

- Input: `Pixal3D/assets/images/9_img.png`
- Seed: `42`
- Resolution: `1024`
- Manual FOV: `0.3119156565218918`
- Low VRAM: enabled
- Texture: disabled
- Sparse/shape steps: `12`
- Max tokens: `49152`

## Mac 12-Step Geometry Command

Representative command (writes a manifest, logs, and the GLB under `runs/mac-geometry`):

```bash
python3 scripts/run_with_manifest.py \
  --out runs/mac-geometry \
  --cwd Pixal3D \
  --env HF_HOME=../../hf-cache \
  --env SPARSE_CONV_BACKEND=none \
  --env ATTN_BACKEND=sdpa \
  --env SPARSE_ATTN_BACKEND=sdpa \
  --env PYTORCH_ENABLE_MPS_FALLBACK=0 \
  -- python inference.py \
  --image assets/images/9_img.png \
  --output ../../runs/mac-geometry/outputs/output.glb \
  --low_vram \
  --resolution 1024 \
  --fov 0.3119156565218918 \
  --device mps \
  --no_texture \
  --ss_steps 12 \
  --shape_steps 12 \
  --tex_steps 4 \
  --max_num_tokens 49152
```

Expected result:

- Valid GLB.
- About `1307672` vertices.
- About `2332768` faces.
- Finite bounds.

## CUDA Reference Command

The CUDA reference was produced on a rented H100 pod using the same ported
tree with native CUDA dependencies. The remote setup scripts are in `scripts/`
(`m13_runpod_preflight.sh`, `m13_cuda_remote_setup.sh`, `m13_cuda_reference.sh`).
The high-level command is:

```bash
export HF_TOKEN='hf_...'    # your own Hugging Face token
export RUN_SEQUENCE=1
bash scripts/m13_cuda_remote_setup.sh
```

This produces the geometry reference runs (4-step and 12-step robot,
12-step textured, and a repo-sample auto-camera run).

## Strict Geometry Comparison

```bash
python scripts/compare_glb_artifacts.py \
  --left runs/cuda-reference-12step \
  --right runs/mac-geometry \
  --out runs/mac-geometry/compare-strict
```

Expected result:

- `status: PASS`
- `count_rtol: 0.15`
- `extent_rtol: 0.05`
- vertex relative delta about `0.0992`
- face relative delta about `0.0978`
- max bbox extent delta about `0.0102`

## Minimal Public Artifact Set

For a reviewer who does not need to rerun:

- `docs/CASE_STUDY.md`
- `docs/CLAIMS.md`
- `docs/PUBLICATION_EVIDENCE.md`
- `evidence/geometry-comparison-contact-sheet.png`
- `evidence/reports/glb_compare_cuda12_strict.json`

## Reproducibility Caveats

- Model downloads require Hugging Face access to gated or mirrored model repos.
- Exact runtime depends heavily on local cache state, thermal behavior, PyTorch version, and MPS behavior.
- Raw artifacts contain absolute local paths and should be sanitized before public release.
- CUDA reference artifacts were produced using the ported tree with native CUDA dependencies, not a pristine upstream app checkout.
