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
  video_set/
    density_render/
    blade_demo/
    newton_bridge/
  dataset_metrics/
    dataset_summary.json
    video_metrics.json
  training_metrics/
    baseline_force_model.json
  inference_results/
    baseline_force_predictions.csv
    inference_metrics.json
  logs/
    density_render.log
    blade_demo.log
    newton_bridge.log
  runs/
    density_render/
    blade_demo/
    newton_bridge/
  experiment_manifest.json
```

## Metrics

Dataset metrics summarize generated videos, force signals, tool path length,
particle count, particle height range, and mean particle displacement when a
Newton bridge log is present.

Training metrics currently compute a deterministic baseline over the available
force sequence. The default baseline predicts the next force norm using the
previous force norm. This is intentionally simple: it gives future learned
models a reproducible floor to beat.

Inference results write per-sample baseline predictions and aggregate validation
MAE/RMSE. When a learned model is added, it should write its predictions into the
same `inference_results/` folder and its training curves into
`training_metrics/`.

## Config Controls

The reference experiment config exposes:

- density renderer frames, substeps, and device
- standalone Warp MPM material settings
- MuJoCo-Newton bridge render mode and visual smoothing
- Newton sand density, friction, stiffness, damping, and jitter scale
- train/validation split for baseline learning metrics

The bridge runner also accepts these controls directly, for example:

```bash
python scripts/run_mujoco_newton_mpm_bridge.py \
  --config configs/newton_bridge_heightfield.json \
  --sand-friction 0.82 \
  --sand-young-modulus 1500000 \
  --sand-jitter-scale 1.4
```
