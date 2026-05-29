# License Notes

This is not legal advice. It is a publication checklist for avoiding obvious redistribution and attribution mistakes.

## Current Publication Plan

Safe default:

- publish source patches, scripts, docs, and small derived evidence images;
- do not publish third-party model weights;
- do not publish Hugging Face caches;
- do not publish raw run outputs without sanitization;
- link to upstream projects and model cards instead of redistributing their assets.

## Items To Review Before Public Release

| Component | Why it matters | Current stance |
|---|---|---|
| Pixal3D code | Upstream project code/license | Pixal3D was reported MIT during the project, but verify current repository license before release. |
| Pixal3D weights | Model redistribution terms may differ from code | Do not redistribute weights unless model license explicitly allows it. |
| TRELLIS.2 | Inherited architecture and dependencies | Attribute any TRELLIS-derived concepts and code paths. |
| trellis-mac | Apple Silicon strategy reference | Attribute clearly; do not imply those pieces were original. |
| DINOv3 / DINOv3 mirror | Gated/model-license constraints | Link to model card; do not redistribute weights/cache. |
| MoGe | Camera estimation model | Link to source/model card; verify license before shipping artifacts that require it. |
| NAF / natten-mps | Feature upsampling / attention substitute | Attribute code and model usage if included. |
| RMBG / BiRefNet | Background removal | Do not redistribute gated weights; document lazy/bypass behavior. |
| RunPod CUDA artifacts | Generated evidence from third-party model | Publish only if allowed; otherwise provide metrics/contact sheets and reproducibility commands. |
| Generated GLBs/images | Generated from model weights and input assets | Treat as portfolio evidence; verify input image rights before public posting. |

## Attribution Notes

The case study should explicitly say:

- Pixal3D is the upstream model/project being adapted.
- trellis-mac influenced the Apple Silicon fallback strategy.
- This project contributes the verification harness, scoped Mac-port patches, fallback integration, CUDA audit runbook, and evidence trail.

## Do Not Publish Without Review

- Hugging Face caches
- `.venv` directories
- downloaded model snapshots
- raw downloaded CUDA-reference archives
- credentials files or token-file paths
- uncurated raw run outputs
