# autonomous-cuda-to-metal-port

> A long-horizon agentic engineering run: a coding agent kept on a hard, verifiable goal for days by a steering harness, and the artifact the run produced. The artifact is a verification-first port of Pixal3D's CUDA-oriented image-to-3D path to Apple Silicon, validated against a CUDA H100 reference.

This repository contains two things. The **artifact** is a working Apple Silicon adaptation of Pixal3D's geometry-and-texture inference/export path, checked stage-by-stage against native CUDA. The **process** is how it was produced: the port was built by a coding agent (Codex) running against a single long-horizon goal, kept on course by a steering harness operating in place of continuous human supervision. The harness consists of durable planning files, a multi-model advisor panel, falsification-first experiment registries, and adversarial verification gates that grade the work computationally. The run took roughly **72 hours of agent working time across about a week**, paused by hand only for local-compute contention or usage limits, and was **human-steered about five times**. The division of labor:

> **Codex performed the technical execution; a multi-model advisor panel and a set of durable artifacts held the work pointed at the right problem and tied each claim to recorded evidence.**

## The setup, so you can copy it

The agent ran as a long-horizon **Codex goal**: an objective plus a definition-of-done that could only be marked complete once the work actually met it, re-entered whenever the agent went idle so the run kept making progress past the first plausible result. That engine is generic. What made it productive on a hard problem is the per-project harness it ran on, which is the reusable part:

- Durable steering files carry strategy, the live diagnosis, and the next move. A first-tier document set (`GOAL.md` → `STEERING.md` → `RESEARCH_PROTOCOL.md` → `RESEARCH_STATE.md`) holds them, and decisions are made through this layer. Any fresh agent instance resumes by reading state off disk.
- A falsification-first research loop attaches every experiment to a numbered hypothesis, names the cheapest diagnostic that could disconfirm it, and commits its pass/fail rule before looking at the output. Hypotheses carry live statuses (`open / supported / weakened / rejected`), and a failed run records what it ruled out. A machine check has to pass before a result counts as evidence.
- A multi-model advisor panel keeps the focus correct. At defined checkpoints the agent consults a panel (GPT‑5.5, GPT‑5.5 Pro, Claude Opus 4.8) acting as supervising researchers. Their role is to prevent local hill-climbing: to challenge whether the run is working on the right problem, call a branch exhausted, and gate expensive or irreversible moves. Advice is recorded against a fixed schema, only when it traces to a saved artifact.
- A verifier validated by controls. The harness produced a parity gate and validated the gate with a control suite: a CUDA-vs-CUDA self-check and semantic-preserving round-trips must pass, while planted adversarial controls (scaled geometry, flipped winding, gamma-shifted color, a known-bad candidate) must fail. Human review was positioned as a final sanity check after the computational gate, with the gate carrying the measurement.
- Manifest-backed run folders. Every milestone wrote a self-documenting capsule: manifest, logs, timings, tensor summaries, outputs. Failed runs are preserved as artifacts, since a failed run with a precise cause is useful evidence. The run produced hundreds of them.

The full account, with representative slices of the registries and the advisor-checkpoint table, is in [`HARNESS.md`](HARNESS.md).

## What the harness produced

Pixal3D's released inference path is built for Linux and NVIDIA CUDA. The run adapted it to run locally on an **M4 MacBook Pro with no NVIDIA GPU**, replacing each CUDA-oriented component with a PyTorch MPS, CPU, or pure-Python equivalent, and treated the result as a measurement problem: a run is credible only if it can be checked against the platform the model was designed for. The same ported tree was also run on a rented **H100** to produce a native-CUDA reference of the exact same input, and the Mac output was compared at the tensor, sparse-coordinate, mesh, and GLB levels.

For one frozen input (a sample robot image, fixed seed, manual field-of-view, 12 sampling steps):

| Metric | Mac vs CUDA H100 |
|---|---:|
| Stage-1 coarse sparse-structure coordinate Jaccard | **0.9882** |
| Final vertex / face count delta | 9.92% / 9.78% |
| Max bounding-box extent delta | 1.02% |
| Mesh-replay of CUDA 12-step FDG tensors through the Mac path | exact vertex/face parity |
| Decode of CUDA 4-step HR shape latents through the Mac path | sub-0.1% mesh delta |
| Stage-3 high-resolution latent feature cosine (mean / median) | 0.5161 / 0.5891 |

**Geometry** transfers at the coarse structural level (0.9882 coordinate Jaccard; ~10% mesh density; ~1% scale), and the Mac mesh/export path is exact when fed CUDA's own neural tensors, so the divergence is localized upstream in the neural decode. The HR latent cosine of ~0.52 marks the limit: coarse layout matches; fine latent detail does not. This is structural-geometry agreement; high-resolution latent features diverge.

**Texture/viewer parity** was the long final stretch. After dozens of rejected candidates and a 6,912-variant convention sweep that ruled out cheap explanations, a fresh H100 boundary capture showed the neural texture path is numerically very close to CUDA at the captured boundaries (texture-SLat cosine 0.999971; decoded-PBR color MAE ~0.0009). The integrated native candidate then cleared all 13 final-candidate gates against the CUDA reference, under a 4-million-sample parity gate stable across three seeds, with viewer-accurate normal semantics, followed by human visual acceptance of the paired render sheets. This parity is **accepted for that exact frozen candidate/reference pair only**; it is not general Pixal3D texture parity.

## Why it's hard

Pixal3D depends on CUDA-only kernels with no Apple Silicon equivalent. Each had to be reimplemented and then validated against a reference:

| CUDA / Linux component | Mac strategy |
|---|---|
| `flash_attn` / xFormers attention | PyTorch scaled-dot-product attention (SDPA) |
| `flex_gemm` sparse 3D convolution | pure-PyTorch submanifold sparse-conv fallback (`conv_none`) with streaming neighbor lookup |
| `o_voxel` dual-grid mesh extraction | Python / PyTorch flexible dual-grid mesh extraction |
| `o_voxel` / `cumesh` texture baking | sparse-trilinear / xatlas PBR exporter + a native Metal narrow-band remesh path |
| Pixal3D projection `grid_sample(..., padding_mode="border")` | an MPS-compatible clamped-grid sampling shim |
| `.cuda()` / `torch.cuda.*` / `device="cuda"` | a device helper routing MPS / CPU cache and sync behavior |

The sparse-conv fallback alone is a from-scratch submanifold 3D-convolution kernel checked against an independently written reference and an axis-orientation sentinel. Ported numerics are validated against independent references with explicit tolerances, behind a focused test suite. See [`docs/COMPATIBILITY_MATRIX.md`](docs/COMPATIBILITY_MATRIX.md) and [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md).

## Calibrating the fidelity metric

An early diagnostic treated mesh-graph fragmentation as evidence of a broken port, until the H100 reference turned out to be more fragmented under the same metric (CUDA: 2,571 components at 4-step vs the Mac's 1,524). The fragmentation was the model's behavior, and the diagnostic was uncalibrated. The lesson, recorded in [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md): a diagnostic is not a fidelity gate until it has been calibrated against a reference. The harness routed the run to measure ground truth by spending a paid H100 capture before promoting the diagnostic to a gate.

Along the way the run found and fixed real bugs: a **silent node-dropping traversal bug in a Metal BVH kernel** (lifting shell-area parity vs CUDA from 0.33 to 0.99 once it switched to stackless traversal), and a **texture-color sampling bug** in the MPS `grid_sample` fallback (collapsing base-color error from 0.169 to 0.012). Both were caught by validating against the real CUDA path.

## Status

Frozen-target port study complete (May–June 2026). Geometry is CUDA-anchored for one frozen 12-step target; integrated texture/viewer parity is accepted for one exact candidate against one CUDA reference. May need adaptation for other inputs, resolutions, or Pixal3D releases. Not claimed: general/arbitrary-input parity, bitwise equivalence, CUDA-equivalent speed, training, or a consumer app. See [`docs/CLAIMS.md`](docs/CLAIMS.md) and [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) for the precise supported/unsupported boundary.

## Repository layout

```text
HARNESS.md    the copyable setup: steering files, advisor loop, verification gates, registry slices
patches/      the Mac-port changes as a patch + an apply script
scripts/      validation harness: manifest runner, GLB / sparse / tensor comparators, parity gates
tests/        focused unit tests for each ported subsystem
docs/         case study, claims, metrics, compatibility matrix, limitations, reproducibility
evidence/     sanitized comparison images + the metric reports (geometry and texture)
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

- The comparison images in `evidence/` were rendered from a sample input shipped with upstream Pixal3D. No model weights are redistributed here.
- License: MIT (see `LICENSE`). This repository adapts Pixal3D (Tencent, MIT) and draws its Apple Silicon strategy from trellis-mac (MIT); both upstream licenses and the Pixal3D `NOTICE` are preserved. See `ATTRIBUTION.md` and `NOTICE-pixal3d.txt`.
