# Excavation Policy With Property Prediction

This artifact demonstrates how material-property prediction can be used for an
excavation behavior, not only for passive material identification.

## Command

```bash
python scripts/render_excavation_policy_compare.py --config configs/rendering/excavation_policy_compare.json
```

On systems with `make`:

```bash
make excavation-policy
```

## Comparison

Both sides use the same GT MPM material. The difference is the action policy:

```text
left:  fixed nominal excavation, no material model
right: property-aware excavation, using predicted rho/phi/delta/cohesion
```

The property-aware policy reads the final posterior from
`outputs/online_mohr_coulomb_bestval_quick/rollout_predictions.csv`.
For stronger predicted material, it chooses a shallower cut, slower insertion,
and shorter push distance to reduce force-limit violations.

## Current Output

```text
outputs/excavation_policy_compare/excavation_policy_compare.mp4
outputs/excavation_policy_compare/excavation_policy_compare_preview.png
outputs/excavation_policy_compare/excavation_policy_compare_sheet.png
outputs/excavation_policy_compare/excavation_policy_compare_metrics.csv
outputs/excavation_policy_compare/excavation_policy_compare_metadata.json
```

## Current Result

For the current selected test material:

```text
GT:   rho=1261.5, phi=30.7, delta=23.7, cohesion=8.5
pred: rho=1268.6, phi=32.9, delta=22.2, cohesion=8.7
```

Final-frame summary:

```text
fixed/no-model peak force:      3128.56
property-aware peak force:      2697.16
peak force reduction:            431.41

fixed/no-model work:            1122.86
property-aware work:             828.94
work reduction:                  293.91

fixed/no-model force violation:  228.56
property-aware force violation:    0.00

fixed/no-model target mass:     7471.46
property-aware target mass:    11724.00

fixed/no-model safety reward:    878.84
property-aware safety reward:   1154.08
reward delta:                    275.24
```

This should be interpreted as a first policy-conditioning sanity check. The
current policy is a transparent heuristic, not a trained reinforcement-learning
controller. Its purpose is to show the path from:

```text
property posterior -> action parameter choice -> safer excavation rollout
```
