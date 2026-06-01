# CUDA reference comparison

The Mac geometry path was checked against a CUDA H100 reference produced from
the same ported tree with native CUDA dependencies. The reference is not a
bitwise oracle; it provides a concrete target for what the official geometry
path produces under a frozen input and configuration.

## How the reference was run

The CUDA reference ran on a rented H100 SXM cloud GPU pod, using a CUDA 12.1
PyTorch container image (Python 3.10, PyTorch `2.6.0+cu124`) with the official
CUDA backends where available (`ATTN_BACKEND=flash_attn_3`,
`SPARSE_ATTN_BACKEND=flash_attn_3`, `SPARSE_CONV_BACKEND=flex_gemm`,
`PIXAL3D_MESH_CONVERT_BACKEND=o_voxel`).

- Critical CUDA imports passed: `natten`, `flash_attn_interface`, `flex_gemm`, `o_voxel`.
- Optional texture imports passed: `cumesh`, `nvdiffrast.torch`, `nvdiffrec_render`.
- Model snapshots were materialized for `TencentARC/Pixal3D` and `camenduru/dinov3-vitl16-pretrain-lvd1689m`.
- One issue surfaced and was fixed: the local `Pixal3D/natten` MPS shim shadowed
  the CUDA `natten` wheel for NAF, so the CUDA reference path temporarily
  disables that shim when running on CUDA.

The audit captured per-run manifests, environment and import checks, the source
diff applied to upstream, input-image checksums, stage-level debug tensors
(low-res and high-res shape latents, the decoder tensors), mesh-graph metrics,
turntable renders, and the final GLB with a programmatic validation report. The
intent was to capture low-res and high-res latent metrics, not only final-GLB
existence.

## Reference framing

- The CUDA audit is a check on the replacement decisions, not a proof of
  general textured parity.
- Native CUDA NAF versus the Mac interpolation fallback is documented as a
  quality fallback, not compared as equivalent.
- The Mac fallback UV texture export is not scored as parity. A separate H100
  texture boundary capture and one accepted integrated native candidate are
  scored against the CUDA reference; see "Texture boundary capture" below.

## Reference runs

Four CUDA reference runs were produced and checksum-verified:

| Run | Mode | Validation | Vertices | Faces | Materials | Textures |
|---|---|---:|---:|---:|---:|---:|
| Robot, 4-step, geometry | matched canary | PASS | 257,544 | 419,244 | 0 | 0 |
| Robot, 12-step, geometry | primary reference | PASS | 1,451,711 | 2,585,722 | 0 | 0 |
| Repo sample, 12-step, auto camera | environment check | PASS | 167,215 | 228,494 | 0 | 0 |
| Robot, 12-step, textured | textured target | PASS | 899,351 | 969,613 | 1 | 1 |

The 4-step run is a low-step canary and debug parity check; the 12-step robot
run is the primary geometry reference. The repo-sample auto-camera run
distinguishes a broken environment from a hard project input.

## First comparison pass (low-step canary)

The first pass compared the Mac low-step canary against the CUDA runs:

| Comparison | Result | Interpretation |
|---|---|---|
| Mac robot 4-step vs CUDA robot 4-step | PASS | Matched debug configuration. Vertex/face counts differ by ~27%/~25%, but bbox extents differ only ~0.9–2.9%, so Mac geometry is structurally in-family with CUDA for this low-step canary. |
| Mac robot 4-step vs CUDA robot 12-step | PASS with count warnings | Not a strict parity comparison because CUDA uses more steps; bounds remain close while CUDA is much denser. |
| Mac robot 4-step vs CUDA robot 12-step textured | PASS with vertex-count warning | Validates the official textured target, not Mac texture parity. CUDA uses its native texture/cleanup path and produces one material/texture visual. |
| Mac robot 4-step vs repo sample auto-camera | PASS structurally, not semantically comparable | Mainly validates the official CUDA environment and the auto-camera path. |

Conclusion from this pass: the CUDA audit validates the Mac geometry path
structurally for the matched low-step robot/manual-FOV case. It does not
validate Mac texture parity, and it does not prove visual-fidelity parity at the
full 12-step textured target, where the CUDA output is more complete and
coherent than the Mac debug artifact.

## Recalibration

A diagnostic was corrected at this point. An earlier theory treated FDG
mesh-graph active-component count (at threshold 0) as a fidelity gate. The CUDA
reference showed the official path is also highly multi-component under the same
metric — with a lower largest-component fraction than the Mac canary — so that
count is no longer treated as a defect signal by itself; it remains useful only
as a descriptive metric.

The geometry-remediation stage then focused on three things: replaying CUDA
high-res shape latents through the Mac decoder and mesh path, generating a full
Mac 12-step geometry for the frozen robot/manual-FOV target, and running
coordinate-aware sparse comparisons between the Mac and CUDA stage artifacts.

## Geometry-remediation comparison

| Check | Result | Interpretation |
|---|---|---|
| CUDA 4-step high-res shape latent decoded on Mac | PASS | Mac decoder/export replay produced `257445` vertices / `418978` faces versus CUDA native `257544` / `419244`; bbox deltas were effectively zero. |
| CUDA 12-step FDG tensors meshed on Mac | PASS | Mac Python mesh/export replay matched CUDA native `1451711` vertices / `2585722` faces exactly. |
| CUDA 12-step high-res latent decoded on the Mac neural FDG path | PARTIAL | Full dense replay hits memory/performance limits in the `conv_none` fallback decoder path; a scale limitation for replay, not an end-to-end Mac generation failure. |
| Mac 12-step robot/manual-FOV geometry vs CUDA 12-step | PASS | Mac generated `1307672` vertices / `2332768` faces versus CUDA `1451711` / `2585722`; count deltas about `9.9%`, bbox extent deltas `0.30%`, `1.02%`, `0.33%`. |

## Texture boundary capture

A separate H100 capture session was run later to localize the texture path. It
captured the texture-side intermediates from the native CUDA stack and compared
them against the Mac path at the same boundaries:

| Boundary check | Result | Interpretation |
|---|---|---|
| Texture SLat coordinate Jaccard / mean cosine | `1.0` / `0.999971` | The texture sparse latent support and features are effectively identical at the captured boundary. |
| Decoded texture-voxel Jaccard / feature MAE | `0.99653` / `0.000931` | The decoded texture voxels agree to a small MAE. |
| Same-shell decoded-PBR base-color MAE | ~`0.0009` | Baking the same shell yields near-identical base color. |

These captures localize the remaining texture gap to native shell/remesh,
sampler, and topology handling rather than the neural texture field. The texture
comparison report is published at
`evidence/reports/texture_compare_m117_vs_cuda.json`.

## Texture acceptance

One integrated native textured candidate was then evaluated against the CUDA
reference. It clears all 13 final-candidate parity gates under a 4-million-sample
parity gate that is stable across seeds `42`/`43`/`44`, with viewer-accurate
stored-vertex normal semantics, followed by human visual acceptance of paired
default/close-up/specular render sheets. The verifier summary is published at
`evidence/reports/final_candidate_parity_summary.json` and the paired renders at
`evidence/texture-comparison-contact-sheet.png`.

This acceptance holds **only** for that exact candidate/reference pair. Earlier
candidates explicitly failed visual parity. It is not general CUDA texture
parity, and it is not arbitrary-input texture parity.

## Conclusion

- Apple Silicon geometry generation is validated against an official CUDA
  12-step geometry reference for the frozen robot/manual-FOV target.
- The comparison uses strict thresholds: `count_rtol=0.15`, `extent_rtol=0.05`.
- The geometry portion of the Mac output should be described as a structural
  geometry result, not a styled Pixal3D-quality result on its own.
- Texture/material parity is established only at the captured boundaries and for
  one accepted integrated native candidate against one CUDA reference, under
  metric gates plus human review. General CUDA texture parity is not claimed,
  because the broader Mac texture/export surface still uses fallbacks while CUDA
  uses its native texture and cleanup components.
