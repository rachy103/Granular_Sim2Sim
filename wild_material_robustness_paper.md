# Online Multimodal Belief Estimation for Wild Granular-Like Material Families

**Draft status:** paper-style internal draft. The current evidence supports a
hostile synthetic Sim2Sim robustness claim, not a real-world generalization
claim.

## Abstract

Robotic excavation and tool-material interaction policies depend strongly on
the material being manipulated, but granular-like materials are heterogeneous:
gravel, sand, cohesive soil, and brittle "crunching" debris can produce similar
surface motion under one action and sharply different force responses under
another. We introduce a hostile synthetic benchmark for online material
estimation across four material families: gravel, sand, soil, and crunching
stuff. The benchmark includes overlapping physical-property ranges, held-out
tool actions, held-out material instances, joint held-out material-action
evaluation, sensor gain shifts, visual occlusion, lighting/camera jitter, force
impulses, and feature dropout.

We train `WildMaterialBeliefNet`, an online multimodal belief model that fuses
robot wrench/proprioception, visual deformation summaries, and action context.
The model maintains a cumulative Gaussian posterior over four material
parameters `(rho, phi_deg, delta_deg, cohesion_kpa)` and simultaneously predicts
the material family. A validation-calibrated posterior temperature corrects
overconfident uncertainty under wild corruptions.

On the stress benchmark, the model reaches `93.1%` family accuracy,
`87.7%` worst-family accuracy, `0.082` mean normalized property MAE, and
`0.046` mean posterior coverage error. The weakest case is crunching material,
which is most often confused with sand under joint held-out material-action
conditions.

![Figure 1: Stress benchmark robustness summary](../outputs/wild_material_robustness_stress/wild_robustness_summary.png)

## 1. Introduction

Excavation policies fail when the material model is wrong. A fixed action that
works on dry sand can stall on cohesive soil, scatter gravel, or crush brittle
debris. The earlier four-property estimator showed strong Sim2Sim behavior on a
friendly synthetic material distribution, but that setting was too easy: the
same generator produced the training and evaluation trajectories, and material
properties were often directly visible in the force/vision features.

This draft moves to a harder question:

```text
Can an online model infer both physical properties and material family
under unseen actions, unseen material instances, and corrupted observations?
```

The answer is currently yes within the synthetic stress benchmark, with the
important caveat that this is not yet a real-world Real2Sim result.

## 2. Related Work

This benchmark follows several threads in robot material inference and
simulation parameter estimation. BayesSim-style methods frame simulator
parameter identification as posterior inference rather than point regression
([BayesSim](https://arxiv.org/abs/1906.01728)). Visual likelihood-free Bayesian
inference has also been used for granular property estimation from observed
particle behavior ([arXiv:2003.08032](https://arxiv.org/abs/2003.08032)).
Differentiable filtering work motivates recursive multimodal belief updates
instead of frame-independent predictions
([arXiv:2010.13021](https://arxiv.org/abs/2010.13021)). Differentiable physics
for elastoplastic manipulation further suggests that simple tool interactions
can identify physically meaningful material parameters, while also exposing the
importance of occlusion and simulator mismatch
([arXiv:2411.00554](https://arxiv.org/abs/2411.00554)). Force-based interactive
granular recognition work motivates using rolling force dynamics and
high-frequency descriptors rather than instantaneous wrench alone
([arXiv:2403.17606](https://arxiv.org/abs/2403.17606)).

## 3. Benchmark

The benchmark contains four material families:

```text
gravel, sand, soil, crunching
```

Each sample carries four continuous targets:

```text
rho, phi_deg, delta_deg, cohesion_kpa
```

The stress configuration intentionally overlaps property ranges across
families. For example, crunching materials can overlap sand and soil in
`phi_deg`, `delta_deg`, and cohesion; gravel and sand overlap in density and
friction. This prevents the classifier from solving the task only by reading a
non-overlapping property interval.

The evaluation split is hostile:

| Split | Count | Meaning |
|---|---:|---|
| held-out material | 144 | unseen material instances under known action families |
| held-out action | 264 | unseen action templates on train-family material instances |
| held-out material + action | 144 | unseen material instances under unseen actions |

Corruptions include sensor gain shift, force impulses, visual occlusion, camera
scale jitter, lighting shift, feature dropout, and family-specific chatter.

## 4. Method

`WildMaterialBeliefNet` uses three encoders:

```text
sensor encoder:  wrench, proprioception, work, rolling force descriptors
vision encoder:  surface deformation, flow, pile, contact-edge summaries
context encoder: blade geometry, depth, speed, rake angle, drag distance
```

At each frame, a learned modality gate weights sensor, vision, and context
tokens. A recurrent update produces frame evidence. The continuous property
posterior is updated with a precision-weighted Gaussian filter:

```text
precision_t = precision_{t-1} + evidence_precision_t
mu_t = (precision_{t-1} * mu_{t-1} + evidence_precision_t * evidence_mu_t)
       / precision_t
```

The family classifier uses both the recurrent hidden state and the current
property posterior. This matters because stress failures often occur when
dynamics alone makes gravel look brittle like crunching, or crunching look
flowing like sand. The posterior-property branch gives the classifier a second,
physics-shaped vote.

Posterior uncertainty is calibrated after training using the validation split.
The scale is estimated from validation z-scores and applied per target:

```text
rho:          1.662
phi_deg:      1.501
delta_deg:    1.750
cohesion_kpa: 1.421
```

## 5. Results

### 5.1 Stress Benchmark

The stress run is:

```bash
python scripts/run_wild_material_robustness.py \
  --config configs/learning/wild_material_robustness_stress.json \
  --output-dir outputs/wild_material_robustness_stress
```

Overall metrics:

| Metric | Value | Gate |
|---|---:|---:|
| family accuracy | 0.931 | >= 0.800 |
| worst-family accuracy | 0.877 | >= 0.650 |
| mean property nMAE | 0.082 | <= 0.200 |
| posterior coverage error | 0.046 | <= 0.250 |

Per-family metrics:

| Family | Accuracy | Property nMAE |
|---|---:|---:|
| gravel | 0.913 | 0.082 |
| sand | 0.986 | 0.070 |
| soil | 0.949 | 0.091 |
| crunching | 0.877 | 0.085 |

Per-split metrics:

| Evaluation tag | Count | Accuracy | Property nMAE |
|---|---:|---:|---:|
| held-out material | 144 | 0.951 | 0.074 |
| held-out action | 264 | 0.973 | 0.082 |
| held-out material + action | 144 | 0.833 | 0.091 |

Confusion matrix, rows are GT and columns are prediction:

| GT \\ Pred | gravel | sand | soil | crunching |
|---|---:|---:|---:|---:|
| gravel | 126 | 9 | 2 | 1 |
| sand | 1 | 136 | 1 | 0 |
| soil | 0 | 7 | 131 | 0 |
| crunching | 0 | 14 | 3 | 121 |

### 5.3 Reviewer-Style Ablations

A skeptical reviewer can reasonably ask whether the benchmark leaks labels
through action context, whether one modality is carrying the result, and whether
posterior calibration is doing too much work. We therefore run a stress-quick
audit:

```bash
python scripts/run_wild_review_audit.py \
  --config configs/learning/wild_material_robustness_stress.json \
  --quick \
  --output-dir outputs/wild_review_audit
```

| Variant | Family Acc | Worst Family Acc | nMAE | Coverage Err | Paper Gate |
|---|---:|---:|---:|---:|---|
| main | 0.859 | 0.812 | 0.117 | 0.055 | True |
| no property-family head | 0.849 | 0.812 | 0.114 | 0.071 | True |
| sensor only | 0.740 | 0.479 | 0.162 | 0.050 | False |
| vision only | 0.802 | 0.729 | 0.126 | 0.064 | True |
| context only | 0.250 | 0.021 | 0.211 | 0.052 | False |
| no sigma calibration | 0.859 | 0.792 | 0.117 | 0.164 | True |

The context-only result is near four-way chance, so the current split does not
appear to leak material identity through action context. Vision-only remains
stronger than sensor-only, so the current benchmark is primarily driven by
visual deformation summaries. The property-family head helps only marginally,
while sigma calibration substantially improves posterior coverage.

### 5.4 Remaining Failure Mode

The main residual error is crunching misclassified as sand. This is plausible:
under some crush events, the material loses structure and then flows, producing
a sand-like late trajectory. The hard rollout preview deliberately shows this
ambiguous case.

![Figure 2: Hard wild rollout preview](../outputs/wild_material_robustness_stress/wild_rollout_preview.png)

## 6. Limitations

The current claim is intentionally narrow.

- The benchmark is synthetic. It is hostile, but it is not real-world Real2Sim.
- "Crunching" is represented by brittle force drops, compaction, and visual
  shrinkage, but no real audio or fracture geometry is modeled.
- The visual features are summary vectors, not raw image encoders.
- The material families are generated by procedural nuisance variables. A real
  dataset could reveal shortcuts that this benchmark does not cover.
- The policy-level result still uses a heuristic excavation controller, not a
  learned closed-loop controller trained against this wild posterior.

## 7. Next Experiments

The next paper-quality experiments should be:

1. Real tabletop data for the four material families with synchronized RGB-D,
   wrench, robot state, and audio if crunching materials are included.
2. Cross-simulator evaluation: train with this synthetic generator and test in
   a different MPM or DEM configuration.
3. Raw image and height-field encoders instead of hand-authored visual
   summaries.
4. Closed-loop excavation policies conditioned on the wild posterior.
5. Ablations: no property-posterior family branch, no validation sigma
   calibration, force-only, vision-only, no held-out action validation.

## 8. Conclusion

The wild benchmark breaks the earlier friendly setting and forces evaluation
under unseen materials, unseen actions, overlapping family ranges, and corrupted
observations. After adding stronger train-time corruption, held-out-action
validation, posterior sigma calibration, and a property-aware family head, the
model passes the stress gates. The current result is strong enough for a
simulation benchmark paper draft, but the honest next step is real-data
validation.
