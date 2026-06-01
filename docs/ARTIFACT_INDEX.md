# Artifact index

This repository ships a small, curated set of published artifacts rather than
the full multi-gigabyte run tree. The raw run tree (model weights, Hugging Face
caches, virtual environments, GLBs, tensor dumps, and per-run logs) is
**intentionally not published**. What remains is enough to read and check the
geometry and frozen-target texture claims.

## Evidence (`evidence/`)

| Purpose | Path | Notes |
|---|---|---|
| Geometry comparison image | `evidence/geometry-comparison-contact-sheet.png` | Input image, CUDA 12-step geometry, Mac 12-step geometry, and a Mac mesh replay from CUDA tensors. Geometry only. |
| Texture comparison image | `evidence/texture-comparison-contact-sheet.png` | Paired default/close-up/specular render sheets for the accepted candidate vs the CUDA reference. |
| Mesh comparison report | `evidence/reports/glb_compare_cuda12_strict.json` | Mac-vs-CUDA mesh count and bounding-box comparison at strict thresholds (`count_rtol=0.15`, `extent_rtol=0.05`). |
| Stage-1 sparse comparison report | `evidence/reports/stage1_sparse_compare.json` | The headline coarse-structure metric: coordinate Jaccard `0.9882`. |
| Stage-3 latent comparison report | `evidence/reports/stage3b_hr_latent_compare.json` | The disclosed caveat: high-resolution latent feature cosine ~`0.52` mean. |
| Final-candidate parity summary | `evidence/reports/final_candidate_parity_summary.json` | The texture/viewer verifier pass: all 13 final-candidate gates under a 4-million-sample gate stable across seeds `42`/`43`/`44`, for one accepted candidate/reference pair. |
| Texture comparison report | `evidence/reports/texture_compare_m117_vs_cuda.json` | Decoded texture/material comparison vs the CUDA reference (base-color and metallic-roughness textures). |

## Documentation (`docs/`)

The case-study, claims, comparison, metric, compatibility, limitation, patch,
and reproducibility docs in `docs/`. Start with `docs/CASE_STUDY.md` and
`docs/CLAIMS.md`; `docs/PUBLICATION_EVIDENCE.md` maps each public claim to a file
above.

## Verification harness (`scripts/`)

The manifest runner, GLB validator, and the comparators that regenerate the
reports above (for example `compare_glb_artifacts.py`,
`compare_sparse_artifacts.py`, `compare_tensor_artifacts.py`,
`replay_shape_decoder.py`, `make_contact_sheet.py`, and the CUDA-reference
handoff scripts). See `docs/REPRODUCIBILITY.md` for representative commands.

## Patch (`patches/`)

`patches/pixal3d-mac-port.patch` plus `patches/apply.sh`: the Mac-port changes
as a patch against a pinned upstream Pixal3D commit. See `docs/PATCH_SUMMARY.md`.

## Tests (`tests/`)

Environment, device-hygiene, attention, projection, sparse-conv, mesh, pipeline,
geometry-smoke, and GLB-export tests for the replacement subsystems.

## What is excluded and why

- Model weights and Hugging Face caches — large and governed by their own
  licenses; obtain them from the upstream model cards instead.
- Virtual environments and package caches — environment-specific.
- Raw run outputs (GLBs, `.pt` / `.npz` tensor dumps, logs) and the full
  per-milestone run tree — large and not needed to read the result. The
  committed reports and contact sheets above were sanitized of absolute local
  paths.
- The native Metal narrow-band remesh toolchain — documented as a dependency of
  the texture path rather than vendored, consistent with not vendoring the model
  or CUDA dependencies. See `docs/PATCH_SUMMARY.md`.

The numbers in the reports above can be regenerated from a full run using the
comparators in `scripts/`; see `docs/REPRODUCIBILITY.md`.
