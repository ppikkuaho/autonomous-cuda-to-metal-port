# Publication Claims

This file defines what the project can and cannot claim publicly. The boundary
is deliberately sharp: this is a frozen-target study, and the value is in being
exact about what the evidence does and does not support.

## Supported Claim

Recommended wording:

> This project adapts Pixal3D's CUDA-oriented inference/export path to run locally on Apple Silicon, replacing CUDA-oriented pieces with Metal/MPS/CPU/Python fallbacks. For one frozen robot/manual-FOV 12-step target it validates geometry, and real-viewer texture/material parity for one accepted candidate, against a CUDA/H100 reference produced from the same tree with native CUDA dependencies.

Shorter wording:

> Pixal3D inference on Apple Silicon, with geometry and frozen-target texture/viewer parity verified against CUDA.

Most precise wording:

> For one frozen robot/manual-FOV 12-step target, the Mac path matches the CUDA/H100 reference at `0.988` stage-1 sparse-coordinate Jaccard and produces a valid GLB within about `9.9%` vertex/face-count delta and `1.02%` max bbox-extent delta. High-resolution latent features diverge (feature cosine ~`0.52`), so full latent parity is not claimed. One integrated native textured candidate additionally clears all 13 final-candidate parity gates against the CUDA reference under a 4-million-sample gate stable across three seeds, plus human visual acceptance — for that exact candidate/reference pair only.

## Evidence Behind The Claim

- GLB mesh comparison report: `evidence/reports/glb_compare_cuda12_strict.json`
- Stage-1 sparse comparison report: `evidence/reports/stage1_sparse_compare.json`
- Stage-3 high-resolution latent comparison report: `evidence/reports/stage3b_hr_latent_compare.json`
- Final-candidate texture/viewer parity summary: `evidence/reports/final_candidate_parity_summary.json`
- Texture comparison report: `evidence/reports/texture_compare_m117_vs_cuda.json`
- Geometry comparison image: `evidence/geometry-comparison-contact-sheet.png`
- Texture comparison image: `evidence/texture-comparison-contact-sheet.png`
- Compatibility matrix: `docs/COMPATIBILITY_MATRIX.md`
- CUDA comparison narrative: `docs/CUDA_REFERENCE_COMPARISON.md`
- Metric definitions: `docs/METRICS.md`

## Allowed Qualifiers

Use:

- "geometry path" / "inference/export path"
- "structurally comparable"
- "one frozen robot/manual-FOV 12-step target"
- "CUDA-anchored validation"
- "frozen-target texture/viewer parity for one accepted candidate against one CUDA reference"
- "structural geometry validation, not full latent equivalence"

Avoid:

- "full Pixal3D port"
- "CUDA-equivalent"
- "production-ready"
- "native Apple Silicon backend"
- "texture parity" *without the frozen-target scope*
- "works for arbitrary inputs"
- "official quality on Mac"

## Unsupported Claims

These are not supported by the current evidence:

- General or arbitrary-input texture/style parity with CUDA (parity is accepted for one frozen candidate/reference pair only; earlier candidates explicitly failed).
- Full latent or bitwise CUDA equivalence (HR latent features diverge).
- Mac performance competitive with CUDA.
- The fallback `conv_none` backend as a general replacement for `flex_gemm`.
- Support for all Pixal3D modes, resolutions, and releases.
- Readiness for public user support, or a consumer app.

## Public Abstract

Pixal3D's released inference path is CUDA-oriented, relying on specialized attention, sparse convolution, mesh extraction, and texture/postprocess components. This project treats the port as a verification problem rather than a dependency swap: it builds a local Apple Silicon path with Metal/MPS/CPU/Python fallbacks, records artifacts at each milestone (including failures), and uses a RunPod H100 CUDA audit as the reference point. For one frozen 12-step target it validates geometry generation at the coarse structural level, and accepts real-viewer texture/material parity for one integrated native candidate against the CUDA reference under adversarially-controlled metric gates plus human review — while explicitly declining to claim general texture/style equivalence, full latent parity, or arbitrary-input support. The verification loop that produced the result — steering docs, hypothesis/experiment registries, advisor checkpoints, and the gates themselves — is documented as a first-class part of the work.
