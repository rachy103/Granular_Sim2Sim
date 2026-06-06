# Property Estimation Performance Upgrade Notes

## Current Interpretation

The upgraded quick run is encouraging, but it should not be read as real-world
generalization yet. The current dataset is synthetic, generated from one compact
analytic blade-interaction model. Several features have direct monotonic
relationships to the target Mohr-Coulomb parameters, so a neural posterior model
can look unusually strong even with a material-grouped split.

The next performance goal is therefore not only lower MAE. The goal is:

```text
lower held-out material/action error
+ calibrated posterior uncertainty
+ robustness to unseen probing trajectories
+ robustness to simulator/rendering mismatch
```

The training code now reports final Gaussian NLL and 1-sigma/2-sigma posterior
coverage in addition to MAE. Use those calibration metrics before trusting a
new checkpoint.

Latest best-validation quick check:

```text
output: outputs/online_mohr_coulomb_bestval_quick
final_frame_mae_mean: 14.3765
mae_improvement_fraction: 0.5910
final_nll_mean: 3.1405
coverage_1sigma_error_mean: 0.1071
coverage_2sigma_error_mean: 0.0476
best_validation_epoch: 6
best_validation_loss: 1.1981
```

The final epoch overfits this quick synthetic split, so the training loop now
restores the best validation checkpoint before writing the rollout CSV and
model artifact. The restored checkpoint improves both point accuracy and
posterior calibration relative to the older calibrated quick run.

For the current selected rollout row:

```text
GT:   rho=1261.5, phi=30.7, delta=23.7, cohesion=8.5
pred: rho=1268.6, phi=32.9, delta=22.2, cohesion=8.7
abs:  rho=7.1,    phi=2.2,  delta=1.5,  cohesion=0.24
```

## Research Signals

- Interactive granular identification work reports that force-only material recognition improves when raw force is paired with time-domain dynamics and high-frequency magnitude features, not only instantaneous force values: https://arxiv.org/abs/2403.17606
- Differentiable multimodal filters for manipulation keep the recursive Bayesian filtering structure while learning how to fuse heterogeneous force, proprioception, and vision signals with uncertainty: https://arxiv.org/pdf/2010.13021
- BayesSim-style simulation parameter inference treats simulator parameters as a posterior distribution rather than a single point estimate, which matches the online Mohr-Coulomb belief-state objective here: https://arxiv.org/abs/1906.01728
- Differentiable physics-based system identification for elastoplastic manipulation shows that physically meaningful parameters can be recovered from simple robot interactions and incomplete visual observations, but visual occlusion and simulator mismatch need explicit handling: https://arxiv.org/abs/2411.00554
- Visual granular property inference with likelihood-free Bayesian inference is
  a close precedent for treating particle/surface observations as simulator
  parameter evidence instead of a purely supervised label lookup:
  https://arxiv.org/abs/2003.08032
- Flexible simulation-based posterior heads should move beyond diagonal
  Gaussians once the dataset is large enough. Neural spline flows are a practical
  candidate for correlated, non-Gaussian posteriors:
  https://arxiv.org/abs/1906.04032

## Implemented Changes

- Added rolling force descriptors to the synthetic multimodal stream:
  - force norm and torque norm
  - rolling force RMS
  - rolling force slope
  - rolling high-frequency power ratio
  - rolling spectral centroid
- Added a crossmodal reliability gate to `OnlineMohrCoulombBeliefNet`.
  The network now learns per-frame weights for sensor, vision, and context tokens before the posterior update.
- Kept the cumulative precision-weighted posterior update.
  The model still carries belief across frames and does not reset material predictions frame by frame.
- Kept the late-frame-weighted NLL.
  This favors accurate posterior convergence after the interaction has accumulated enough evidence.
- Added posterior calibration metrics:
  - final Gaussian NLL
  - final 1-sigma coverage
  - final 2-sigma coverage
  - mean coverage error against Gaussian nominal coverage

## Practical Next Steps

### 1. Make the evaluation harder before making the model larger

- Add three splits:
  - held-out material split
  - held-out action split
  - held-out material-action split
- Report MAE, normalized MAE, final NLL, 1-sigma/2-sigma coverage, and coverage
  error for each split.
- Keep the current quick synthetic generator as a smoke test only.

### 2. Generate a factorial probing dataset

For every material, repeat multiple controlled probes:

```text
depth: shallow / medium / deep
speed: slow / nominal / fast
rake angle: low / nominal / high
seed: repeated packing realizations
```

This is the fastest way to improve identifiability. A single trajectory can make
`rho`, `phi`, `delta`, and `cohesion` trade off against each other. Multiple
actions break that ambiguity.

### 3. Replace synthetic shortcuts with measured features

- Compute rolling force RMS, force slope, high-frequency power ratio, and
  spectral centroid from the actual wrench logs.
- Cache surface deformation summaries from the fixed density renderer:
  pile-front movement, height-field change, contact-edge density, and flow proxy.
- Add sensor noise, timing jitter, camera crop jitter, and particle initialization
  randomness to the synthetic pipeline so the network cannot memorize an exact
  generator.

### 4. Upgrade the posterior head after the data improves

- Stage A: keep the current online Gaussian posterior for speed and calibration.
- Add a posterior temperature or precision damping term before increasing model
  capacity. The current quick run shrinks sigma faster than the empirical error
  justifies.
- Stage B: replace the diagonal Gaussian evidence head with a mixture density
  head.
- Stage C: move to a conditional spline-flow posterior once there are enough
  material/action samples to support correlated uncertainty.

### 5. Use physics as a consistency loss

Add a lightweight Mohr-Coulomb/RFT residual head:

```text
predicted theta + action context -> predicted reaction envelope
observed wrench should stay near that envelope
```

This should reduce physically impossible parameter combinations, especially
when vision is occluded or weak.

### 6. Use the posterior to choose better simulations

Instead of broad uniform domain randomization, sample new simulations from the
current posterior's uncertain regions. This follows the BayesSim idea: train the
inference model, inspect the posterior, then allocate simulation budget where
the posterior is still wide or biased.

## Immediate Experiment Queue

1. Re-run the upgraded quick training after the calibration metric patch and
   record MAE/NLL/coverage as the new baseline. The current best-validation
   quick baseline is `outputs/online_mohr_coulomb_bestval_quick`.
2. Add a non-quick config with at least `96` materials and `6` actions per
   material. The starting config is
   `configs/learning/online_mohr_coulomb_next.json`.
3. Add held-out action evaluation.
4. Generate fixed-render density summaries from
   `configs/rendering/density_mujoco_eef_render_fixed.json`.
5. Run ablations:
   force-only, vision-only, no rolling force features, no modality gate, no
   late-frame weighting.
