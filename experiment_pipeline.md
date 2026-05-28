# Experiment Pipeline

The experiment wrapper creates one named sequence directory per run. Each
sequence is meant to contain everything needed to inspect, train from, and
compare a granular interaction experiment.

## Run

Reference sequence:

```bash
python scripts/run_experiment_sequence.py --config configs/experiments/reference_heightfield_intrusion.json
```

Fast CPU-oriented smoke run without the MuJoCo-Newton bridge:

```bash
python scripts/run_experiment_sequence.py --quick --skip-bridge
```

Make targets:

```bash
make experiment-smoke
make experiment
make pipeline-smoke
make pipeline
make sweep-smoke
make sweep
```

## Output Layout

For sequence `reference_heightfield_intrusion_v001`, the wrapper writes:

```text
outputs/experiments/reference_heightfield_intrusion_v001/
  config/
    source_experiment_config.json
    resolved_experiment_config.json
    blade_demo_config.json
    newton_bridge_config.json
    git_info.json
    workspace_scan.json
  video_set/
    density_render/
    blade_demo/
    newton_bridge/
  dataset_metrics/
    dataset_summary.json
    normalization_stats.json
    probing_tensor_metrics.json
    video_metrics.json
  training_metrics/
    baseline_force_model.json
    mdn_training_metrics.json
    model_config.json
    representation_metrics.json
    temporal_mdn.pt
  inference_results/
    baseline_force_predictions.csv
    inference_metrics.json
    learning_inference_metrics.json
    mdn_predictions.csv
    posterior_phi_deg.png
    posterior_cohesion_kpa.png
    posterior_summary.json
  logs/
    density_render.log
    blade_demo.log
    newton_bridge.log
  runs/
    density_render/
    blade_demo/
    newton_bridge/
    probing_dataset/
      probing_windows.npz
  experiment_manifest.json
```

## Metrics

Dataset metrics summarize generated videos, force signals, tool path length,
particle count, particle height range, and mean particle displacement when a
Newton bridge log is present.

The probing dataset converts synchronized wrench and end-effector kinematics
into windows:

```text
X in R^(N x T x 12)
feature order = fx, fy, fz, tx, ty, tz, px, py, pz, vx, vy, vz
```

The default sensor rate is 50 Hz. Normalization is train-split z-score by
default and is written to `dataset_metrics/normalization_stats.json`. Quick
smoke runs use a one-step window only to verify the filesystem and training
plumbing; full runs use the configured `dataset.window_length`.

Training metrics currently compute a deterministic baseline over the available
force sequence. The default baseline predicts the next force norm using the
previous force norm. This is intentionally simple: it gives future learned
models a reproducible floor to beat.

The learned path runs a compact 1D-CNN plus Transformer encoder. Phase 1 uses
InfoNCE on augmented windows for representation learning. Phase 2 fine-tunes an
MDN head whose output is a diagonal Gaussian mixture over material targets. Set
`learning.num_mixtures` to `1` to use the same path as a single Gaussian
decoder.

Inference results write per-sample baseline predictions, MDN predictions,
aggregate MAE/RMSE, inference latency per window, and posterior plots.

## Config Controls

The reference experiment config exposes:

- density renderer frames, substeps, and device
- standalone Warp MPM material settings
- MuJoCo-Newton bridge render mode and visual smoothing
- Newton sand density, friction, stiffness, damping, and jitter scale
- dataset sensor rate, window length, stride, normalization, and target labels
- train/validation/test split for baseline and learned metrics
- learning model size, epochs, MDN mixture count, and inference timing repeats

The bridge runner also accepts these controls directly, for example:

```bash
python scripts/run_mujoco_newton_mpm_bridge.py \
  --config configs/newton_bridge_heightfield.json \
  --sand-friction 0.82 \
  --sand-young-modulus 1500000 \
  --sand-jitter-scale 1.4
```

## Property Sweep

For amortized inference, a single material label is not enough. The sweep
runner creates many sequence configs by Latin-hypercube sampling material
targets and action nuisance variables:

```bash
python scripts/run_property_sweep.py --config configs/sweeps/lhs_property_sweep.json
```

Fast smoke check:

```bash
python scripts/run_property_sweep.py --quick --skip-bridge --count 2 --sweep-name smoke_lhs_sweep
```

The default sweep ranges are:

```text
phi_deg:       25 to 45
cohesion_kpa: 0 to 15
speed_scale, depth_scale, drag_distance_scale, angle_offset, y_offset, z_offset
```

Each sampled material label is mapped into the compact MPM model before the
sequence is run. `phi_deg` controls the Drucker-Prager friction coefficient and
tool friction; `cohesion_kpa` controls a simple yield offset and stiffness
scale. This is still a compact research proxy, not a calibrated soil law, but it
does make the labels influence the force traces.

The sweep layout is:

```text
outputs/sweeps/<sweep_name>/
  configs/              source, resolved, and per-sample configs
  sequences/            normal experiment-sequence folders
  dataset/              aggregate probing_windows.npz and normalization stats
  training_metrics/     aggregate representation and MDN metrics
  inference_results/    aggregate predictions and posterior plots
  logs/                 per-sample and aggregate training logs
  samples.csv           LHS material/action table with derived solver controls
  sweep_manifest.json
```

The aggregate dataset is rebuilt from every sample's `x_raw` and normalized
globally using the aggregate train split. This preserves cross-material scale
differences before applying the required z-score normalization.
