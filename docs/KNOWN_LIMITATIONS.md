# Known Limitations

- No claim of bitwise CUDA equivalence. The CUDA comparison validates one frozen robot/manual-FOV geometry target structurally, not arbitrary Pixal3D parity.
- 1536 mode is not an initial target.
- Texture baking is verified through a fallback exporter, not through CUDA `o_voxel.postprocess.to_glb`.
- Some CPU fallbacks may remain and must be logged.
- Mesh hole filling may be skipped early.
- Visual quality has two different bars: geometry-only output should look like a coherent clay/structural 3D object with the same major forms, while textured output should also transfer color/material/style cues from the input image. This work only supports the geometry-only bar.
- Performance may be much slower than CUDA.
- The sparse convolution fallback is correctness-oriented only. `SPARSE_CONV_BACKEND=none` supports submanifold sparse conv with stride 1 and no padding, does not implement `SparseInverseConv3d`, and builds neighbor maps with Python/CPU hashing. It is suitable for verifying portable Pixal3D execution paths, but not a performance replacement for `flex_gemm` or a future Metal backend. Streaming/chunked paths reduce peak memory, but full dense 12-step high-res latent neural replay still hits scale limits through this fallback.
- The mesh extraction fallback covers `flexible_dual_grid_to_mesh` for inference decode, not `mesh_to_flexible_dual_grid` for training/encoding.
- Native `torch.nn.functional.grid_sample` with `padding_mode="border"` is version-dependent on MPS. The Pixal3D path now uses a tested compatibility shim in `pixal3d/utils/projection.py`.
- `PIXAL3D_NAF_BACKEND=interpolate` preserves the shape/channel contract and unblocks execution, but it ignores the learned NAF guide-image refinement. Native NAF via `natten-mps` now imports and runs on MPS for the tested image-conditioning path. The geometry validation suggests NAF alone is not the main remaining geometry blocker, but texture/style parity is still unvalidated.
- The fallback textured GLB uses a simple `trimesh` UV texture fallback: decoded voxel base-color attributes are matched to vertices and splatted into a texture image. This validates texture-stage execution and GLB texture serialization, but it is not equivalent to `o_voxel` remeshing, atlas generation, or high-quality texture baking.
- RMBG/BiRefNet is lazy-loaded and bypassed for the current RGBA input. Non-alpha inputs still require access to gated `briaai/RMBG-2.0` or an alternate background-removal path.
- The selected DINOv3 weights use the accessible `camenduru/dinov3-vitl16-pretrain-lvd1689m` mirror. The official `facebook/dinov3-vitl16-pretrain-lvd1689m` repo remains a gating/documentation issue unless local Hugging Face auth is configured and access is approved.
- MoGe installation updated `utils3d` in the venv from the trellis-mac-pinned local wheel to upstream `utils3d==1.3`; later CUDA or trellis-mac baseline comparisons should record that environment difference.
- The proof is geometry-only. It should not be presented as matching the input image's color, material, or visual style.
