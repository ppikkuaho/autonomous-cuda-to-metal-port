# The Harness

How a coding agent was kept working a hard, verifiable goal for days. This is the reusable part of the project; the artifact (the Pixal3D port) is what the run produced, and this is the setup that produced it. Nothing here is Pixal3D-specific; it transfers to any long-horizon technical goal where measurement discipline matters.

The division of labor is the whole idea:

> **Codex performed the technical execution. A multi-model advisor panel and a set of durable artifacts held the work pointed at the right problem and tied each claim to recorded evidence.**

## The engine: a long-horizon goal

The agent ran against a single **Codex goal**: an objective plus a definition-of-done, with a status that could be set to `active`, `paused`, `blocked`, `usage_limited`, `budget_limited`, or `complete`. Two properties made it long-horizon:

- It re-entered the work whenever the agent went idle, so the run kept making progress past the first plausible-looking result.
- It could only be marked `complete` once the objective was actually met. In this run the goal was held at "not complete, pending a human decision" even after the computational gates passed. Completion bias is the default failure mode of an autonomous agent; the goal contract is the structural defense against it.

(The human paused the goal a handful of times, for local-compute contention or usage limits, which is why ~72 hours of agent working time spans about a week of calendar.)

The engine is generic. Everything below is the **track** it ran on, committed as part of the project so the work stays legible and resumable.

## The control loop

Four durable layers, each a file the agent reads and updates before changing direction. State is stored on disk:

```
GOAL.md            immutable objective + 13-item definition-of-done + authority grant
   │
STEERING.md        the directing layer — names the first-tier doc set and the rules
   │
RESEARCH_PROTOCOL  the required before / during / after loop for every experiment
   │
RESEARCH_STATE.md  the live strategic brief: current diagnosis, active decision, next move
   │
   └── registries (hypotheses · experiments · decisions) feed back up
```

> *"This is the directing layer for the project. Future technical decisions should be made through this layer, not from transient chat context or the last command that happened to run."* — `STEERING.md`

The governing rule: *"No new experiment family, advisor consult, or claim change should happen unless it is reflected in the first-tier artifacts."* And an anti-confabulation rule: *"Do not backfill from memory. If a historical claim cannot be traced to saved docs/artifacts, mark it as not backfilled instead of inventing a cleaner history."* The steering docs are treated as goal-level deliverables: if they are stale, the goal is not done even when the code runs.

## The research loop (falsification-first)

Every experiment, before running, must:

1. attach to a numbered **hypothesis**;
2. state the **decision** it could change;
3. choose **the cheapest diagnostic that can falsify the hypothesis**;
4. write the **pass/fail interpretation before looking at the output**.

After running, it updates the experiment row, the hypothesis status, the debugging log, and the live state, and a machine check (`check_steering_docs.py`) has to pass before the result is allowed to count as evidence. A failed experiment is required to record what it ruled out, along with the fact that it failed.

**Hypothesis register (representative slice).** Fourteen hypotheses (H1–H14) carried live statuses. The statuses moved as the loop disconfirmed its own ideas:

| ID | Hypothesis | Status | What moved it |
|---|---|---|---|
| H4 | The transparent look is caused by material alpha | **rejected** | stated falsifier failed |
| H5 | Geometry/export solidity closes the parity gap | **plateaued** | a ~16-variant pitch/origin/quantizer sweep hit a stable failure floor; branch declared exhausted and abandoned |
| H7 | The Mac neural texture *field* differs from CUDA | **strongly weakened** | the H100 boundary capture showed texture-SLat cosine 0.999971 — the field was nearly identical, so the gap was downstream |
| H8 | A global color correction fixes texture | **weakened** | affine/gamma fits moved error only 0.157 → 0.126; not the mechanism |
| H10 | A new paid CUDA run is required to proceed | **rejected** | local evidence still reduced uncertainty; spend deferred |

**Decision log (representative entries).** Strategy decisions, each with a reason:

- *"Build local harness and Mac evidence first. Use RunPod/H100 CUDA as a reference audit after the Mac path is alive. Reason: prevents the project from becoming cloud-driven trial-and-error."*
- *"Stop continuing the reactive experiment loop until the project has a durable research ledger. Reason: evidence existed but was not organized by hypothesis or strategic decision."*
- *"Treat advisors as research-direction control. They should challenge whether the current problem selection is right, whether a branch is exhausted, and what the highest expected-information next experiment is."*

## The advisor panel

At defined checkpoints the agent consulted a panel: **GPT‑5.5, GPT‑5.5 Pro, and Claude Opus 4.8**, instructed to respond *"as a supervising researcher, not as an implementation reviewer."* Their role was to prevent local hill-climbing: repeating easy variants in place of the highest-information next move. They are a direction control for research direction.

Check-ins were triggered by conditions; the agent decided when to escalate. Representative triggers:

- a hypothesis status changes, or a branch plateaus;
- before running another variant in a branch that looks exhausted;
- **before any paid CUDA capture**, and again after ingesting it;
- before strengthening any public claim;
- before declaring a final-review candidate.

Every consult recorded a fixed schema (*source · recommendation · reasoning · evidence used · guardrail it protects against · what would change it · resulting decision*), only when it traced to a saved artifact. The panel had measurable effect: it forced a redesign of the grader (replacing an unbounded raycast with a deterministic sampled-surface proxy), it pointed at the distance-truth test that surfaced the Metal BVH bug, and at one checkpoint it advised against running an experiment that was already planned, which the agent then superseded before implementing. The panel is the mechanism that lets the loop change direction at the points where completion bias would otherwise stop it.

## Verification validated by controls

The harness produced a parity gate and validated the gate with a control suite, so a pass carries information:

| Control | Required verdict | Why |
|---|---|---|
| CUDA reference vs itself | **pass** | a real candidate must clear the bar the ground truth clears |
| `trimesh` round-trip · face-split re-triangulation | **pass** | semantic-preserving changes must not be false-rejected |
| scaled-X / scaled-Z geometry | **fail** | the gate must catch scale errors |
| flipped winding · double-sided · alpha-blend | **fail** | must catch orientation/material cheats |
| gamma-shifted color · metallic/roughness swap | **fail** | must catch texture cheats |
| a known-bad earlier candidate | **fail** | must reject work already judged wrong |

The accepted candidate cleared **all 13 final-candidate gates** under a 4-million-sample parity gate that was stable across three seeds (it was raised from smaller sample counts because those gave seed-unstable verdicts), with viewer-accurate stored-vertex normal semantics. A human was then shown the paired render sheets for a final sanity check: *"human visual approval is a final sanity check, not the primary measurement,"* and *"no human approval until the computational gate passes."*

## The artifact trail

Every milestone wrote a self-documenting run folder, and failed runs are preserved as artifacts: *"a failed run with a precise cause is useful evidence."* The contract per run:

```
manifest.json   stdout.log   stderr.log   env.txt   git_commits.txt
timings.jsonl   memory.jsonl   tensor_summaries.jsonl   errors.jsonl   outputs/
```

The run produced hundreds of these capsules across the milestone chain, each scored against a pre-committed gate and naming the next experiment it implied, which is the mechanism that let the loop choose "what next" without a human at each step. The curated, leak-clean subset of that trail is in [`evidence/`](evidence/); the raw tree (model caches, weights, RunPod archives, absolute local paths) is deliberately not published.

## What transfers

Strip out Pixal3D and the copyable recipe is:

1. **A goal contract** that re-fires on idle and holds the goal open until a human decision is recorded.
2. **Durable steering files** as the continuity layer, so any fresh agent instance resumes from disk.
3. **A falsification-first loop**: hypotheses with falsifiers, pass/fail committed before looking, failures recorded as ruled-out knowledge.
4. **A multi-model advisor panel** as a direction control against hill-climbing, with a recorded decision schema.
5. **A verifier validated by adversarial controls**, with human review positioned as a post-gate sanity check.
6. **Manifest-backed, failure-preserving artifacts** so the final claim traces to evidence.
