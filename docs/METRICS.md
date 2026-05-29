# Metrics

This project uses several structural metrics. They do not all prove the same thing.

## Coordinate Jaccard

Definition:

```text
shared sparse coordinates / union sparse coordinates
```

Use:

- Strong signal for whether two sparse supports occupy the same voxel locations.
- Most useful at early/coarse sparse-structure stages.

Current headline result:

- Mac 12-step vs CUDA 12-step stage-1 sparse structure: Jaccard `0.9882`.

Limits:

- Does not compare feature values at those coordinates.
- High Jaccard at a coarse stage does not guarantee final texture/style parity.

## Coordinate Coverage

Definition:

```text
shared coordinates / coordinates in one side
```

Use:

- Helps distinguish symmetric divergence from one side having many extra/missing coordinates.

Current result:

- Stage-1 CUDA coverage `0.9935`, Mac coverage `0.9946`.

## Feature Cosine

Definition:

Cosine similarity between aligned feature vectors at shared sparse coordinates.

Use:

- Measures whether two paths assign similar latent features where their sparse supports overlap.

Current important caveat:

- Stage-3 HR SLat support has Jaccard `0.6467`.
- Stage-3 HR SLat features have mean cosine `0.5161`, median cosine `0.5891`.

Interpretation:

- The coarse sparse layout is nearly identical, but high-resolution latent features diverge meaningfully.
- This must be disclosed; it is why the project claims structural geometry validation, not full latent equivalence.

## Vertex / Face Count Delta

Definition:

```text
abs(left - right) / max(abs(left), abs(right), 1)
```

Use:

- Coarse sanity check for mesh density.
- Useful when paired with support overlap, bbox, and visual review.

Current result:

- Mac 12-step vs CUDA 12-step: about `9.9%` vertex/face-count delta.

Limits:

- Similar counts do not prove similar shape.
- Different remeshing paths can change counts without changing perceived shape.

## Bbox Extent Delta

Definition:

Relative difference in x/y/z bounding-box extents.

Use:

- Sanity check for camera/scale mismatch.

Current result:

- Mac 12-step vs CUDA 12-step: `0.30%`, `1.02%`, `0.33%`.

Limits:

- Bounding boxes are weak evidence by themselves. Two unrelated meshes can have similar extents.
- This metric should never be the headline proof.

## Human Geometry Review

Use:

- Determines whether the geometry reads as the same structural object class in fixed-camera renders.

Limits:

- Current review is geometry-only.
- It does not approve texture, color, material, or style transfer.
