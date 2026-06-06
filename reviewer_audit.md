# Reviewer Audit: Wild Material Robustness

This audit reviews the paper draft from a skeptical reviewer stance. The goal
is to find plausible reject reasons and test them, not to defend the result by
default.

## Likely Reject Reasons

1. **Synthetic-only evidence.** The paper can be rejected for implying real
   robustness while only proving hostile synthetic Sim2Sim robustness.
2. **Shortcut leakage.** The generator may leak material family through action
   context or procedural family signatures.
3. **Missing ablations.** The method needs evidence that sensor, vision,
   posterior-property classification, and calibration each matter.
4. **Uncertainty overclaim.** Posterior bands must be calibrated, not only
   visually plausible.
5. **Hand-crafted visual features.** The current result does not prove that raw
   camera observations are sufficient.
6. **No independent simulator or real materials.** The benchmark is hostile,
   but it is still one procedural simulator family.

## Audit Experiment

Command:

```bash
python scripts/run_wild_review_audit.py \
  --config configs/learning/wild_material_robustness_stress.json \
  --quick \
  --output-dir outputs/wild_review_audit
```

The audit uses the stress quick benchmark to run targeted ablations:

| Variant | Family Acc | Worst Family Acc | nMAE | Coverage Err | Paper Gate |
|---|---:|---:|---:|---:|---|
| main | 0.859 | 0.812 | 0.117 | 0.055 | True |
| no_property_family_head | 0.849 | 0.812 | 0.114 | 0.071 | True |
| sensor_only | 0.740 | 0.479 | 0.162 | 0.050 | False |
| vision_only | 0.802 | 0.729 | 0.126 | 0.064 | True |
| context_only | 0.250 | 0.021 | 0.211 | 0.052 | False |
| no_sigma_calibration | 0.859 | 0.792 | 0.117 | 0.164 | True |

## Reviewer Findings

- **Context shortcut risk is not supported.** Context-only accuracy is `0.250`,
  essentially chance for four classes. This weakens the label-leakage objection.
- **Vision is doing most of the class work.** Vision-only passes the quick
  stress gate, while sensor-only fails worst-family accuracy. The paper should
  not claim balanced sensor/vision contribution.
- **The property-posterior family branch is helpful but not decisive.**
  Removing it drops accuracy from `0.859` to `0.849`, so the branch is not the
  main reason the model works.
- **Calibration helps but is not the only reason the model passes.** Removing
  sigma calibration worsens coverage error from `0.055` to `0.164`; it still
  passes the loose gate, but the calibrated version is much safer to report.
- **The main remaining reject reason is external validity.** A reviewer can
  still reject on the grounds that the system has no real material data and no
  independent simulator.

## Required Paper Edits

- State the claim as hostile synthetic Sim2Sim robustness, not real-world
  robustness.
- Add the audit table to the paper or supplementary material.
- Do not oversell the property-family head.
- Add a limitation that raw image learning and real material trials remain
  undone.
- For a stronger submission, add real or second-simulator validation before
  claiming general material robustness.

