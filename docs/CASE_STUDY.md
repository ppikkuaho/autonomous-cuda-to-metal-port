# Case study: porting Pixal3D geometry to Apple Silicon

## The question

Pixal3D is a single-image-to-3D model. Its released inference path inherits a TRELLIS.2-style stack with CUDA-oriented components for attention, sparse 3D convolution, mesh extraction, texture baking, and postprocessing. Apple Silicon has a very different execution stack: PyTorch MPS, Metal, unified memory, no CUDA extensions, and different operator support and performance behavior.

The technical question was whether the Pixal3D geometry path could be made to run locally on an M4 MacBook Pro, and — more importantly — whether that result could be verified against official CUDA output well enough to stand behind a concrete claim. The answer is yes for geometry, with texture and style parity explicitly deferred.

The framing was deliberately not "build a polished Mac app" and not "rewrite Pixal3D from scratch." It was to demonstrate a rigorous porting workflow for a modern CUDA-first 3D generation system: build the verification harness before changing the model path, port one subsystem at a time, record artifacts for every step including the failures, and use a CUDA reference at the end as an audit rather than as a crutch.

## Scope

In scope: the Apple Silicon local inference path; geometry generation and GLB export; device-hygiene and CUDA-call removal for that path; an SDPA attention fallback; MPS-compatible projection sampling; a pure-PyTorch sparse-convolution fallback; a Python/PyTorch mesh-extraction fallback; and a CUDA H100 reference audit with a documentation and artifact trail.

Out of scope: training, broad distribution, consumer packaging, arbitrary Pixal3D release compatibility, bitwise CUDA equivalence, and full texture/style parity.

## Architecture

The upstream tree was kept clean and all Apple Silicon adaptations were isolated, so the contribution can be read as a patch rather than a fork. The verification harness and tests live outside the model tree.

```text
patches/    the Mac-port changes as a patch (applies on a pinned upstream commit)
scripts/    manifest runner, GLB / sparse / tensor comparators, replay tools
tests/      environment, device, math, sparse-conv, mesh, and smoke tests
docs/        plan, claims, compatibility matrix, limitations, metrics
evidence/    sanitized geometry comparison image and metric reports
```

## The porting work

| Area | CUDA / Linux assumption | Mac strategy |
|---|---|---|
| Device movement | `.cuda()`, `device="cuda"`, `torch.cuda.*` | a device helper and selected-path cleanup |
| Attention | `flash_attn` / CUDA backends | PyTorch SDPA |
| Projection sampling | `grid_sample(..., padding_mode="border")` | an MPS-compatible clamped-grid shim |
| Sparse convolution | `flex_gemm` | a pure-PyTorch `conv_none` fallback with streaming and chunking |
| Mesh extraction | `o_voxel` C/CUDA path | a Python/PyTorch flexible dual-grid fallback |
| Texture export | CUDA / native postprocess | a fallback GLB texture writer (mechanical validation only) |
| Verification | ad-hoc run success | manifests, stage artifacts, GLB validation, CUDA comparison |

## The verification ladder

The point of the ladder is that "a GLB exists" is weak evidence for a port. The strongest evidence comes from boundary tests and same-target comparison against a reference.

1. Environment and static CUDA scan.
2. A native Apple Silicon TRELLIS.2 baseline, to prove the underlying stack runs at all.
3. Pixal3D import and device-hygiene tests.
4. CPU/MPS math tests for attention and projection.
5. Sparse-convolution unit tests.
6. Synthetic mesh-extraction tests.
7. Real Pixal3D image-conditioning and pipeline loading.
8. First local geometry export and a fallback texture export.
9. Automatic-camera (MoGe) proof.
10. A CUDA H100 reference audit.
11. Post-audit recalibration and a 12-step Mac-vs-CUDA geometry comparison.

## The CUDA reference audit

The audit was run on a rented H100 using the official-style CUDA stack, with the same ported tree and native CUDA dependencies (`natten`, `flash_attn`, `flex_gemm`, `o_voxel` all imported and ran). Its role was not to prove bitwise equality but to provide a concrete target for what the official geometry path produces under the frozen input and configuration.

| Reference run | Vertices | Faces | Materials | Textures |
|---|---:|---:|---:|---:|
| CUDA robot, 4-step geometry | 257,544 | 419,244 | 0 | 0 |
| CUDA robot, 12-step geometry | 1,451,711 | 2,585,722 | 0 | 0 |
| CUDA robot, 12-step textured | 899,351 | 969,613 | 1 | 1 |
| CUDA repo sample, auto-camera | 167,215 | 228,494 | 0 | 0 |

## The geometry result

The comparison anchors on the frozen robot, manual-FOV target at 12 steps. The strongest agreement is at the coarse sparse-structure stage, not the final bounding box.

| Stage metric | Result |
|---|---:|
| Stage-1 sparse coordinate Jaccard | 0.9882 |
| Stage-1 shared coordinates | 3,686 |
| Stage-1 CUDA-only / Mac-only coordinates | 24 / 20 |
| Stage-3 high-resolution latent Jaccard | 0.6467 |
| Stage-3 high-resolution latent feature mean / median cosine | 0.5161 / 0.5891 |

The coarse geometry layout is nearly identical, while the high-resolution latent features diverge. The final mesh comparison stays structurally close:

| Metric | CUDA 12-step | Mac 12-step | Relative delta |
|---|---:|---:|---:|
| Vertices | 1,451,711 | 1,307,672 | 9.92% |
| Faces | 2,585,722 | 2,332,768 | 9.78% |
| Bounding-box extents | | | 0.30%, 1.02%, 0.33% |

The reading is straightforward: the Mac geometry path is structurally close to official CUDA for this frozen target, anchored on the stage-1 sparse coordinate Jaccard, with mesh counts and bounding box as supporting sanity checks. The high-resolution latent divergence is an open limitation and is why this is not claimed as full latent parity. The output should be presented as geometry-only — not as matching the input image's color, materials, or styled appearance.

A geometry comparison image is provided in `evidence/`, showing the input image alongside the CUDA 12-step geometry, the Mac 12-step geometry, and a Mac mesh replay from CUDA tensors. It is a geometry comparison, not a texture or style comparison, and a human sign-off that the two render as the same object remains a pending check; until then the project rests on the measured agreement above rather than on visual plausibility.

## A debugging detour worth keeping

An early working theory treated mesh-graph fragmentation as evidence of semantic failure. The CUDA audit corrected it: the official CUDA path is also highly multi-component under the same threshold-0 graph metric, with an even lower largest-component fraction than the Mac canary.

| Run | Coordinates | Active components | Largest-component node fraction |
|---|---:|---:|---:|
| Mac 4-step robot | 187,131 | 1,524 | 19.4% |
| CUDA 4-step robot | 257,544 | 2,571 | 7.8% |
| CUDA 12-step robot | 1,451,711 | 5,536 | 15.7% |

The lesson: a diagnostic metric is not a fidelity gate until it has been calibrated against a reference.

## What is and is not claimed

The supported claim: the Pixal3D geometry inference path was adapted to run locally on Apple Silicon, replacing CUDA-oriented dependencies with MPS/CPU/Python fallbacks, and a frozen 12-step geometry target was validated against a CUDA H100 reference produced from the same ported tree with native CUDA dependencies.

Not claimed: CUDA-equivalent texture quality, CUDA-equivalent speed, general Pixal3D Mac distribution, arbitrary-input parity, bitwise equality, or a polished consumer app.

## Future work

The natural follow-up is texture and style equivalence: revisiting `o_voxel` texture baking and cleanup, investigating Metal or CPU replacements for CUDA rasterization and UV baking, comparing Mac and CUDA textured GLBs under fixed-camera renders, and adding a visual gate for material, color, and style transfer. That is a reasonable next project, but it is not required for the geometry-port result documented here.
