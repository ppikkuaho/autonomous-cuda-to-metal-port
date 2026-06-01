# Publication Evidence

This table maps public claims to evidence, metrics, and caveats.

The raw run artifacts referenced below are not redistributed (model weights and
multi-gigabyte outputs are excluded); the load-bearing comparison reports and
contact sheets are included under `evidence/`, and the numbers can be
regenerated with the harness in `scripts/`. Every public claim in the table
below points to a file shipped in this repository or to a narrative doc under
`docs/`.

| Public claim | Evidence | Metric / result | Caveat |
|---|---|---|---|
| Selected Pixal3D geometry path runs locally on Apple Silicon | `evidence/reports/glb_compare_cuda12_strict.json`, `evidence/geometry-comparison-contact-sheet.png` | valid GLB, `1307672` vertices / `2332768` faces | one frozen robot/manual-FOV target |
| Coarse sparse geometry agrees with CUDA | `evidence/reports/stage1_sparse_compare.json` | coordinate Jaccard `0.9882`, shared `3686`, CUDA-only `24`, Mac-only `20` | does not prove feature or texture parity |
| High-resolution latent features diverge | `evidence/reports/stage3b_hr_latent_compare.json` | HR latent Jaccard `0.6467`, mean/median feature cosine `0.5161`/`0.5891` | disclose as an active limitation |
| Mac geometry is structurally close to CUDA geometry for the frozen target | `evidence/reports/glb_compare_cuda12_strict.json` | `9.92%` vertex delta, `9.78%` face delta, max bbox extent delta `1.02%`; strict thresholds `count_rtol=0.15`, `extent_rtol=0.05` | structural mesh stats, not texture/style fidelity |
| Mac mesh/export path reproduces CUDA tensors exactly | `docs/CUDA_REFERENCE_COMPARISON.md` (geometry-remediation comparison) | meshing CUDA 12-step FDG tensors on Mac = exact vertex/face match; decoding CUDA 4-step HR latents on Mac = sub-0.1% mesh delta | replay starts after CUDA neural decode, not a full CUDA 12-step neural replay |
| CUDA reference used native CUDA components | `docs/CUDA_REFERENCE_COMPARISON.md` | critical imports passed: `natten`, `flash_attn_interface`, `flex_gemm`, `o_voxel` | reference used the ported tree with native CUDA dependencies, not a pristine upstream app |
| Texture neural path is numerically close to CUDA at captured boundaries | `evidence/reports/texture_compare_m117_vs_cuda.json`, `docs/CUDA_REFERENCE_COMPARISON.md` | texture SLat coord Jaccard `1.0` / mean cosine `0.999971`; decoded texture-voxel Jaccard `0.99653` / feature MAE `0.000931`; same-shell base-color MAE ~`0.0009` | boundary capture, not end-to-end Mac texture generation |
| One integrated native textured candidate reaches viewer parity with CUDA | `evidence/reports/final_candidate_parity_summary.json`, `evidence/texture-comparison-contact-sheet.png` | clears all 13 final-candidate gates under a 4-million-sample gate stable across seeds `42`/`43`/`44`, plus human visual acceptance | accepted for that exact candidate/reference pair only; earlier candidates failed visual parity |
| Key replacement subsystems have tests | the test suite in `tests/` | sparse conv normal `14 passed`, streaming `14 passed`, import/device `5 passed, 1 skipped` | not exhaustive coverage for all Pixal3D modes |

## Reviewer Shortcut

For a quick review, open:

1. `docs/CASE_STUDY.md`
2. `docs/CLAIMS.md`
3. `evidence/geometry-comparison-contact-sheet.png`
4. `evidence/texture-comparison-contact-sheet.png`
5. `evidence/reports/glb_compare_cuda12_strict.json`
6. `evidence/reports/final_candidate_parity_summary.json`

## Evidence Not Used As A Public Claim

- Raw FDG component counts are descriptive only; they were recalibrated after CUDA showed similar multi-component structure.
- Old low-step visual failures are retained as debugging history, not as the final visual packet.
- The earlier Mac fallback-exporter run proves GLB serialization, not visual quality, and is not part of the accepted texture-parity claim. The accepted texture result is the integrated native candidate evaluated against the CUDA reference, recorded in `evidence/reports/final_candidate_parity_summary.json`.
