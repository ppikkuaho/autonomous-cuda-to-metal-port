# Patch Summary

This summarizes the selected Mac-port changes relative to upstream Pixal3D. It is
not a full diff.

The published patch (`patches/pixal3d-mac-port.patch`, applied with
`patches/apply.sh` onto a pinned upstream commit) touches 14 files (8 modified,
6 new). It covers the **geometry inference path** and the **texture-export Python
path** in `inference.py`. It does **not** vendor upstream Pixal3D code, model
weights, CUDA backends, or the native Metal narrow-band remesh toolchain used by
the accepted textured candidate — those are documented as dependencies, not
shipped (see "What the patch does not cover" below).

## Inference Entry Point

Primary file: `Pixal3D/inference.py`

Changes:

- Default attention backends to SDPA on Mac.
- Add explicit `--device` handling.
- Add low-VRAM movement policy for selected pipeline stages.
- Support geometry-only output with `--no_texture`.
- Add the texture-export Python path: `export_textured_glb` routes to native
  `o_voxel.postprocess.to_glb` (with `remesh=True, remesh_band=1`) when its
  dependencies are present, and otherwise falls back to a pure-Python `trimesh`
  UV texture export (`export_textured_glb_fallback`).
- Avoid unguarded CUDA cache/synchronization calls in the selected path.
- Add manual FOV/debug-step controls used by the frozen canary.

## Device Policy

Primary file: `Pixal3D/pixal3d/utils/device.py`

Changes:

- Add `resolve_device`, `empty_device_cache`, and `synchronize_device`.
- Route CUDA/MPS/CPU cache and sync behavior through a small helper instead of hardcoded `torch.cuda.*`.

## Projection Sampling

Primary files:

- `Pixal3D/pixal3d/utils/projection.py`
- `Pixal3D/pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`

Changes:

- Add an MPS-compatible `grid_sample` wrapper for border-padding behavior.
- Patch Pixal3D projection conditioning to use the wrapper.
- Support current DINOv3 layout and accessible DINOv3 mirror used by the Mac path.

## Sparse Convolution

Primary file: `Pixal3D/pixal3d/modules/sparse/conv/conv_none.py`

Changes:

- Add a pure-PyTorch submanifold sparse convolution fallback.
- Add coordinate duplicate detection.
- Add cached neighbor lookup for small/medium tensors.
- Add streaming/vectorized neighbor lookup for larger tensors.
- Add edge chunking to reduce memory pressure.

Limitations:

- Correctness-oriented, not performance-oriented.
- Submanifold stride-1 path only.
- Not a replacement for `flex_gemm` performance.
- Dense CUDA-12 HR latent neural replay remains too large for this fallback.

## Sparse Tensor Utilities

Primary file: `Pixal3D/pixal3d/modules/sparse/basic.py`

Changes:

- Add compatibility behavior needed by the fallback sparse convolution path and debug replay scripts.

## Mesh Extraction

Primary files:

- `Pixal3D/pixal3d/models/sc_vaes/fdg_vae.py`
- `Pixal3D/pixal3d/utils/mesh_extract.py`

Changes:

- Guard CUDA `o_voxel` dependency.
- Add Python/PyTorch flexible dual-grid mesh extraction fallback.
- Preserve synthetic mesh tests and CUDA FDG replay evidence.

## Pipeline Loading

Primary files:

- `Pixal3D/pixal3d/pipelines/pixal3d_image_to_3d.py`
- `Pixal3D/pixal3d/pipelines/rembg/BiRefNet.py`

Changes:

- Lazy-load or skip background removal when not needed.
- Support reduced model loading for decode replay.
- Add decoder control flags used for decode-replay diagnostics.

## Verification Harness

Primary files:

- `scripts/run_with_manifest.py`
- `scripts/validate_glb.py`
- `scripts/compare_glb_artifacts.py`
- `scripts/compare_sparse_artifacts.py`
- `scripts/replay_shape_decoder.py`
- `scripts/fdg_graph_metrics.py`
- `scripts/m13_cuda_reference.sh`
- `scripts/m13_cuda_remote_setup.sh`

Changes:

- Standardize manifests/logs.
- Validate GLB structure programmatically.
- Compare GLB mesh stats with explicit thresholds.
- Compare sparse coordinates/features by coordinate overlap and aligned tensor stats.
- Replay saved shape latents or FDG tensors through selected Mac boundaries.
- Package and run CUDA reference audits.

## What The Patch Does Not Cover

The patch is the Mac adaptation only. The following are deliberately treated as
documented dependencies rather than vendored, consistent with not shipping the
model or CUDA backends:

- **Upstream Pixal3D source and model weights.** The patch applies onto a clean
  upstream checkout at a pinned commit; obtain weights from the upstream model
  cards.
- **CUDA backends.** `flash_attn`, `flex_gemm`, `o_voxel`, `natten`,
  `nvdiffrast`, and related CUDA wheels are imported or guarded, not vendored.
- **Native Metal narrow-band remesh toolchain.** The accepted textured candidate
  is produced through the native `o_voxel.postprocess.to_glb` remesh boundary
  invoked by the patched `inference.py`; the native Metal narrow-band remesh
  kernels themselves are a documented dependency, not part of this patch. The
  pure-Python `trimesh` fallback in the patch produces a loadable textured GLB
  but not the accepted-candidate parity result. See
  `docs/REPRODUCIBILITY.md` and `docs/CUDA_REFERENCE_COMPARISON.md`.
- **The final-candidate parity-suite runner.** The texture/viewer parity gates
  are described in `docs/CUDA_REFERENCE_COMPARISON.md`; the suite runner that
  produced `evidence/reports/final_candidate_parity_summary.json` is not part of
  the published harness.
