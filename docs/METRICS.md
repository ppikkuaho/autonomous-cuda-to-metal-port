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

- Geometry review approves structure only.
- It does not, by itself, approve texture, color, material, or style transfer; the texture/viewer gates below cover that.

## Texture / Viewer Parity Metrics

These metrics were added for the texture stage. They are evaluated against the
CUDA textured GLB treated as a frozen oracle for the same frozen
robot/manual-FOV 12-step input, seed `42`. They support a frozen-target
texture/viewer parity result for one accepted candidate, not a general texture
or style parity claim.

### Texture Boundary-Capture Metrics

Definition:

Tensor-level comparison of the neural texture path between CUDA and Mac at
captured internal boundaries, with shape and noise controlled (CUDA texture
noise replayed through the Mac path).

Use:

- Localizes where, if anywhere, the neural texture path diverges from CUDA.
- Separates a neural-field problem from a downstream shell/UV/export problem.

Current results (H100 boundary capture):

- Texture-SLat coordinate Jaccard `1.0`, feature mean cosine `0.999971`.
- Decoded texture-voxel coordinate Jaccard `0.99653`, feature MAE `0.000931`.
- Decoded-PBR base-color MAE `~0.0009` on the same shell.

Interpretation:

- The neural texture field is numerically very close to CUDA at the captured
  boundaries for this frozen target.
- This proves the texture content is not the dominant gap. It does not by itself
  prove final viewer parity, because mesh extraction, UV unwrap/atlas, surface
  sampling, and viewer/material interpretation are downstream of these
  boundaries.

Limit:

- Boundary capture is available only for this frozen input. It is not a general
  neural-texture parity statement.

### Sampled Multi-View Parity Gate

Definition:

Fixed, normalized orthographic comparison of the candidate GLB against the CUDA
GLB across eight azimuths and high/low elevations, plus a deterministic
sampled-surface backend run under a locked gate signature. The accepted
candidate was summarized under a `4`-million-sample surface-proxy gate, and the
gate held across seeds `42`, `43`, and `44`.

Use:

- Rejects or ranks local geometry/export/texture candidates on silhouette,
  boundary, support, area, depth, ray-thickness, hit topology, surface
  distance, normal, albedo, and shaded-color rows before any human review.

What it proves / does not prove:

- A pass means the candidate is viewer-equivalent to CUDA under the locked
  software-render and `model-viewer` conditions for this frozen target.
- It does not prove bitwise tensor identity, identical triangle order, identical
  UV packing, or parity for any other input.

### Final-Candidate Gate Set

Definition:

The integrated native candidate is accepted only when all thirteen
final-candidate output gates pass: GLB validation, GLB material semantics,
deterministic software multi-view geometry/solidity/color, sampled high/low
support, full-mesh exact-horizon orientation/winding, browser `model-viewer`
parity, browser high/low parity, browser close-up texture/detail parity,
browser specular/material-response parity, and the locked one-command summary
that recomputes each verdict and validates artifact provenance.

Use:

- A single accept/reject decision for a final texture candidate, anchored to a
  fixed reference and candidate GLB with locked thresholds.

Limit:

- Tensor-boundary and texture-attribution diagnostics explain failures but
  cannot override a failing output gate.

### Adversarial And Positive Controls

Use:

- Calibrate the gate so a pass is meaningful. The gate must accept benign
  positives and reject planted defects under the same locked thresholds.

Current control behavior:

- Positives that must pass: CUDA versus itself, a CUDA `trimesh` round-trip, a
  benign uniform-scale GLB, and a topology-changing face-split GLB that
  preserves the surface and UVs.
- Negatives that must fail: scaled-geometry, flipped-winding, gamma/color-shifted,
  metallic/roughness-texture, and a known-bad earlier candidate.

Interpretation:

- A diagnostic is not a fidelity gate until controls calibrate it. The same
  thresholds that accept the integrated native candidate reject these planted
  defects, which is what gives the final pass its evidential weight for the
  frozen target.

### Human Texture Review

Use:

- Final confirmation after the computational gate passes. Reviewers inspect
  paired default, close-up, and specular render sheets.

Limits:

- Human review is positioned after the computational gate, never as the
  measurement. It accepts texture/material parity only for the exact accepted
  candidate against the CUDA reference. Earlier candidates explicitly failed
  this review.
