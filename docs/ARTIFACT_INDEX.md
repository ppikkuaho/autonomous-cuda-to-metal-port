# Evidence index

This repository ships a small, curated evidence set rather than the full
multi-gigabyte run tree. Model weights, Hugging Face caches, virtual
environments, and raw run outputs are intentionally excluded; what remains is
enough to read and check the geometry claim.

## Evidence in this repository

| Purpose | Path | Notes |
|---|---|---|
| Geometry comparison image | `evidence/geometry-comparison-contact-sheet.png` | Input image, CUDA 12-step geometry, Mac 12-step geometry, and a Mac mesh replay from CUDA tensors. Geometry only — not a texture or style comparison. |
| Mesh comparison report | `evidence/reports/glb_compare_cuda12_strict.json` | Mac-vs-CUDA mesh count and bounding-box comparison at strict thresholds (`count_rtol=0.15`, `extent_rtol=0.05`). |
| Stage-1 sparse comparison report | `evidence/reports/stage1_sparse_compare.json` | The headline coarse-structure metric: coordinate Jaccard `0.9882`. |
| Stage-3 latent comparison report | `evidence/reports/stage3b_hr_latent_compare.json` | The disclosed caveat: high-resolution latent feature cosine ~`0.52` mean. |

## What is excluded and why

- Model weights and Hugging Face caches — large and governed by their own
  licenses; obtain them from the upstream model cards instead.
- Virtual environments and package caches — environment-specific.
- Raw run outputs (GLBs, `.pt` / `.npz` tensor dumps, logs) — large and not
  needed to read the result. The committed reports above were sanitized of
  absolute local paths.

The numbers in the reports above can be regenerated from a full run using the
comparators in `scripts/`; see `docs/REPRODUCIBILITY.md`.
