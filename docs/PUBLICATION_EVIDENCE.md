# Publication Evidence

This table maps public claims to evidence, metrics, and caveats.

The raw run artifacts referenced below are not redistributed (model weights and
multi-gigabyte outputs are excluded); the load-bearing comparison reports are
included under `evidence/reports/`, and the numbers can be regenerated with the
harness in `scripts/`.

| Public claim | Evidence | Metric / result | Caveat |
|---|---|---|---|
| Selected Pixal3D geometry path runs locally on Apple Silicon | local geometry run | valid GLB, `1307672` vertices / `2332768` faces | one frozen robot/manual-FOV target |
| Coarse sparse geometry agrees with CUDA | `evidence/reports/stage1_sparse_compare.json` | coordinate Jaccard `0.9882`, shared `3686`, CUDA-only `24`, Mac-only `20` | does not prove feature or texture parity |
| High-resolution latent features diverge | `evidence/reports/stage3b_hr_latent_compare.json` | HR latent Jaccard `0.6467`, mean/median feature cosine `0.5161`/`0.5891` | disclose as an active limitation |
| Mac geometry is structurally close to CUDA geometry for the frozen target | `evidence/reports/glb_compare_cuda12_strict.json` | `9.92%` vertex delta, `9.78%` face delta, max bbox extent delta `1.02%`; strict thresholds `count_rtol=0.15`, `extent_rtol=0.05` | structural mesh stats, not texture/style fidelity |
| Mac decoder/export can reproduce CUDA 4-step HR-latent geometry | 4-step decoder replay run | `0.038%` vertex delta, `0.063%` face delta | 4-step replay, not full CUDA 12-step neural replay |
| Mac Python mesh/export path can reproduce CUDA 12-step mesh | 12-step mesh replay run | exact vertex/face/bbox match | starts after CUDA neural decode |
| CUDA reference used native CUDA components | `docs/CUDA_REFERENCE_COMPARISON.md` | critical imports passed: `natten`, `flash_attn_interface`, `flex_gemm`, `o_voxel` | reference used the ported tree with native CUDA dependencies, not a pristine upstream app |
| Texture output is mechanically possible on Mac | fallback texture export run | valid textured GLB with one material/texture visual | fallback exporter only; not CUDA texture parity |
| Key replacement subsystems have tests | the test suite in `tests/` | sparse conv normal `14 passed`, streaming `14 passed`, import/device `5 passed, 1 skipped` | not exhaustive coverage for all Pixal3D modes |

## Reviewer Shortcut

For a quick review, open:

1. `docs/CASE_STUDY.md`
2. `docs/CLAIMS.md`
3. `evidence/geometry-comparison-contact-sheet.png`
4. `evidence/reports/glb_compare_cuda12_strict.json`

## Evidence Not Used As A Public Claim

- Raw FDG component counts are descriptive only; they were recalibrated after CUDA showed similar multi-component structure.
- Old low-step visual failures are retained as debugging history, not as the final visual packet.
- Texture fallback validation proves GLB serialization, not visual quality.
