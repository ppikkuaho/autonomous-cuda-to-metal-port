# Publication Claims

This file defines what the project can and cannot claim publicly.

## Supported Claim

Recommended wording:

> This project adapts Pixal3D's geometry inference path to run locally on Apple Silicon. It replaces CUDA-oriented pieces with MPS/CPU/Python fallbacks and validates one frozen 12-step geometry target against a CUDA/H100 reference using native Pixal3D CUDA dependencies.

Shorter wording:

> Pixal3D geometry inference on Apple Silicon, verified against CUDA for one frozen 12-step target.

Most precise wording:

> For one frozen robot/manual-FOV 12-step target, the Mac path matches the CUDA/H100 reference at `0.988` stage-1 sparse-coordinate Jaccard and produces a valid geometry-only GLB within about `9.9%` vertex/face-count delta and `1.02%` max bbox-extent delta. High-resolution latent features diverge, and texture/style parity is not claimed.

## Evidence Behind The Claim

- GLB mesh comparison report: `evidence/reports/glb_compare_cuda12_strict.json`
- Stage-1 sparse comparison report: `evidence/reports/stage1_sparse_compare.json`
- Stage-3 high-resolution latent comparison report: `evidence/reports/stage3b_hr_latent_compare.json`
- Geometry comparison image: `evidence/geometry-comparison-contact-sheet.png`
- Compatibility matrix: `docs/COMPATIBILITY_MATRIX.md`
- CUDA comparison narrative: `docs/CUDA_REFERENCE_COMPARISON.md`
- Metric definitions: `docs/METRICS.md`

## Allowed Qualifiers

Use:

- "geometry path"
- "structurally comparable"
- "one frozen robot/manual-FOV 12-step target"
- "CUDA-anchored validation"
- "fallback texture export is mechanical only"
- "texture/style parity deferred"

Avoid:

- "full Pixal3D port"
- "CUDA-equivalent"
- "production-ready"
- "native Apple Silicon backend"
- "texture parity"
- "works for arbitrary inputs"
- "official quality on Mac"

## Unsupported Claims

These are not supported by the current evidence:

- Mac textured output matches CUDA textured output.
- Mac performance is competitive with CUDA.
- The fallback `conv_none` backend is a general replacement for `flex_gemm`.
- The Mac path supports all Pixal3D modes, resolutions, and releases.
- The project is ready for public user support.

## Public Abstract

Pixal3D's released inference path is CUDA-oriented, relying on specialized attention, sparse convolution, mesh extraction, and texture/postprocess components. This project treats the port as a verification problem rather than a simple dependency swap. It builds a local Apple Silicon path with MPS/CPU/Python fallbacks, records artifacts at each milestone, and uses a RunPod H100 CUDA audit as the reference point. The final result validates geometry generation for one frozen 12-step target, while explicitly deferring texture/style equivalence.
