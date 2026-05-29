# Patch Summary

This summarizes the selected Mac-port changes relative to upstream Pixal3D. It is not a full diff.

## Inference Entry Point

Primary file: `Pixal3D/inference.py`

Changes:

- Default attention backends to SDPA on Mac.
- Add explicit `--device` handling.
- Add low-VRAM movement policy for selected pipeline stages.
- Support geometry-only output with `--no_texture`.
- Add fallback textured GLB export path.
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
