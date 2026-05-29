# pixal3d-apple-silicon-port

> A verification-first adaptation of Pixal3D's CUDA-oriented image-to-3D geometry path to Apple Silicon, validated stage-by-stage against a CUDA H100 reference.

**Why this is hard.** Pixal3D's geometry path depends on CUDA-only kernels — flash attention, `flex_gemm` sparse 3D convolution, `o_voxel` dual-grid mesh extraction, and a border-padded `grid_sample` projection — none of which has an Apple Silicon equivalent. Each had to be reimplemented in pure PyTorch / MPS / CPU and then *proven correct*, not merely made to run. The sparse-conv fallback alone is a from-scratch submanifold 3D-convolution kernel checked against an independently written reference and an axis-orientation sentinel; the whole port is ~1,800 lines across 14 files, behind 41 tests. To audit it, the same patched tree was rented onto an H100 to produce a native-CUDA reference of the exact same run, and the Mac output was compared against it at the tensor, sparse-coordinate, mesh, and GLB levels.

## What it is

Pixal3D is a single-image-to-3D model whose released inference path is built for Linux and NVIDIA CUDA. This project adapts the geometry portion of that pipeline to run locally on an M4 MacBook Pro — no NVIDIA GPU — replacing each CUDA-oriented component with a PyTorch MPS, CPU, or pure-Python fallback, and then treats the result as a measurement problem rather than a "does it run" problem. A run is only credible if its output can be checked against the platform the model was designed for, so the same ported tree was also run on a rented H100 to produce a CUDA reference, and the Mac output is compared against it at the tensor, sparse-coordinate, mesh, and GLB levels.

The adaptation ships as a patch against a pinned upstream commit, not as a vendored copy of the model. The interesting content is the porting decisions and the validation discipline: what was replaced, how each replacement was checked, and exactly where the Mac path agrees with CUDA and where it does not.

**The most senior moment is a debugging detour that got kept.** An early diagnostic treated mesh-graph fragmentation as evidence of a broken port — until the H100 CUDA reference turned out to be *even more fragmented* under the same metric. The fragmentation was the model's behavior, not a Mac bug. The lesson, recorded in [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md): a diagnostic is not a fidelity gate until it has been calibrated against a reference. Building a same-tree CUDA reference specifically to audit a Mac port — and then trusting the data over the initial theory — is the part of this repo that is hardest to fake.

## Headline result

The Mac geometry matches the H100 reference to within ~10% mesh density and ~1% scale, with near-identical coarse structure (**0.9882** coordinate Jaccard). High-resolution latent features still diverge, so this is a structural-geometry result, not full latent equivalence — and not a texture, color, or style result.

For one frozen input (a sample robot image, fixed seed, manual field-of-view, 12 geometry sampling steps, no texture), the Mac geometry path was compared against a CUDA H100 reference produced by the same ported tree with native CUDA dependencies:

| Metric | Mac vs CUDA (12-step) |
|---|---:|
| Stage-1 coarse sparse-structure coordinate Jaccard | **0.9882** |
| Final vertex-count relative delta | 9.92% |
| Final face-count relative delta | 9.78% |
| Max bounding-box extent delta | 1.02% |
| Stage-3 high-resolution latent feature cosine (mean / median) | 0.5161 / 0.5891 |

The Stage-3 high-resolution latent feature cosine (mean ~0.52) is the honest limit of the claim: coarse geometry transfers, fine latent detail does not, and texture parity is explicitly out of scope. These numbers are the quantitative claim. Whether the Mac geometry also *reads* as the same object to a human eye is a separate visual-plausibility check that has not yet been signed off; see [Status](#status).

## Technical approach

The pipeline was ported one subsystem at a time, each behind its own test, with the upstream tree kept clean and all Mac-specific changes isolated into a patch.

| CUDA / Linux component | Mac strategy |
|---|---|
| `flash_attn` / xFormers attention | PyTorch scaled-dot-product attention (SDPA) |
| `flex_gemm` sparse 3D convolution | pure-PyTorch submanifold sparse-conv fallback (`conv_none`) with streaming neighbor lookup |
| `o_voxel` dual-grid mesh extraction | Python / PyTorch flexible dual-grid mesh extraction |
| `o_voxel` / `cumesh` texture baking and cleanup | a fallback `trimesh` UV exporter (mechanical GLB validation only) |
| `nvdiffrast` rasterization | not used by the geometry path |
| `.cuda()` / `torch.cuda.*` / `device="cuda"` | a small device helper routing MPS / CPU cache and sync behavior |
| Pixal3D projection `grid_sample(..., padding_mode="border")` | an MPS-compatible clamped-grid sampling shim |

The validation strategy is layered: environment and static CUDA scans, isolated math tests for attention and projection, unit tests for the sparse-conv and mesh-extraction fallbacks, real image-conditioning and pipeline-load checks, a local end-to-end geometry run, and finally a CUDA H100 reference audit. In total, 41 tests across 11 files and 28 validation scripts back the comparison; ported numerics are checked against independently written references with explicit tolerances, not just exercised. The calibration detour described above ([`docs/CASE_STUDY.md`](docs/CASE_STUDY.md)) came out of this harness.

See [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md) for the full account, [`docs/COMPATIBILITY_MATRIX.md`](docs/COMPATIBILITY_MATRIX.md) for the per-component status, [`docs/METRICS.md`](docs/METRICS.md) for what each metric does and does not prove, and [`docs/CLAIMS.md`](docs/CLAIMS.md) for the precise supported and unsupported claims.

## Status

Validated against a CUDA H100 reference for one frozen 12-step geometry target, May 2026; texture/style parity is future work, and may need adaptation for other inputs, resolutions, or Pixal3D releases.

The quantitative geometry comparison above is complete. A human visual confirmation that the Mac and CUDA geometry render as the same object under a fixed camera is a remaining check; the geometry comparison image is in [`evidence/`](evidence/) for that purpose. Until that confirmation, the project asserts the measured agreement, not visual plausibility.

## Repository layout

```text
patches/      the Mac-port changes as a patch + an apply script
scripts/      validation harness: manifest runner, GLB / sparse / tensor comparators
tests/        focused unit tests for each ported subsystem
docs/         case study, claims, metrics, compatibility matrix, limitations
evidence/     sanitized comparison image + the load-bearing metric reports
```

## Run it

The port is distributed as a patch against a pinned upstream Pixal3D commit. No upstream source or model weights are included.

```bash
git clone https://github.com/TencentARC/Pixal3D
git -C Pixal3D checkout 28efad66fdcbd8174a8538d9baf71fe34fe4b6d2
./patches/apply.sh Pixal3D
```

The standalone validators and tests run without the full model:

```bash
pip install -r requirements.txt
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest -q \
  tests/test_000_env.py tests/test_030_sdpa.py tests/test_040_projection_math.py
```

Reproducing the full geometry run and the CUDA comparison requires the Pixal3D dependencies and Hugging Face access to the model weights; see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Notes

- The geometry comparison image in `evidence/` was rendered from a sample input shipped with upstream Pixal3D. No model weights are redistributed here.
- License: MIT (see `LICENSE`). This repository adapts Pixal3D (Tencent, MIT) and draws its Apple Silicon strategy from trellis-mac (MIT); both upstream licenses and the Pixal3D `NOTICE` are preserved. See `ATTRIBUTION.md` and `NOTICE-pixal3d.txt`.
