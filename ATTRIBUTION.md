# Attribution

This repository contains an Apple Silicon adaptation study of a third-party
model. It does not redistribute any upstream source code or model weights. The
Mac-port changes ship as a patch (`patches/pixal3d-mac-port.patch`) that applies
on top of a clean upstream checkout.

## Original contribution

The original work in this repository is:

- the Mac-port patch — device-aware execution, an SDPA attention path, an
  MPS-compatible projection-sampling shim, a pure-PyTorch sparse-convolution
  fallback, and a Python/PyTorch mesh-extraction fallback;
- the verification harness (`scripts/`, `tests/`) that compares the Mac path
  against a CUDA reference at the tensor, sparse-coordinate, mesh, and GLB
  levels;
- the CUDA-reference audit runbook and the comparison/metrics methodology;
- the documentation and evidence trail.

It is licensed under the MIT License (see `LICENSE`).

## Upstream projects

- **Pixal3D** — Tencent ARC. The image-to-3D model being adapted.
  Licensed MIT. Upstream: <https://github.com/TencentARC/Pixal3D>.
  The patch is generated against upstream commit
  `28efad66fdcbd8174a8538d9baf71fe34fe4b6d2`.
  Pixal3D's `NOTICE` (responsible-use clause and third-party component
  attributions) is reproduced in `NOTICE-pixal3d.txt`.

- **TRELLIS.2** — Microsoft. The base architecture Pixal3D builds on.
  Upstream: <https://github.com/microsoft/TRELLIS.2>.

- **trellis-mac** — the Apple Silicon compatibility strategy that informed the
  fallback approach (SDPA attention, pure-PyTorch sparse conv, Python mesh
  extraction). Licensed MIT. Upstream:
  <https://github.com/shivampkumar/trellis-mac>.

## Model weights and external models

No model weights are included. The pipeline depends on third-party models that
must be obtained from their own sources under their own licenses, including:

- Pixal3D (`TencentARC/Pixal3D`)
- DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`, gated; an accessible
  mirror was used during the study)
- MoGe (`Ruicheng/moge-2-vitl`)
- RMBG / BiRefNet (`briaai/RMBG-2.0`, gated)

Consult each model card for its license and usage terms. Do not assume the MIT
license on this repository extends to any model weights.

## Generated outputs and input image

The single committed evidence image (`evidence/`) is a rendered geometry
comparison produced from a sample input shipped with the upstream Pixal3D
repository. No personal photographs or third-party copyrighted imagery are
included.
