# Reproducibility Contract

This project follows a source/asset/artifact split so the paper page can stay
pretty without hiding how each result is regenerated.

## Source Repository

Tracked in git:

- Python source under `src/granular_mpm/`
- Entry points under `scripts/`
- Versioned configs under `configs/`
- Project page, paper draft, audit, and reproducibility docs under `docs/`
- Browser-ready page assets under `docs/assets/`
- Install, smoke, and Makefile targets

Ignored by git:

- `outputs/` generated videos, logs, CSVs, USD, PLY, and previews
- `dist/` packaged artifact bundles
- `mujoco_menagerie/` external robot assets
- Python virtual environments and caches

## Fresh Clone

Reference path:

```bash
git clone https://github.com/rachy103/Granular_Sim2Sim.git
cd Granular_Sim2Sim
./install.sh --locked
```

The installer creates `.venv`, installs the package with MuJoCo, Newton,
learning, and test extras, downloads MuJoCo Menagerie at the commit recorded in
`configs/external_assets.json`, and runs import tests.

The reference lock is:

```text
constraints/reference-linux-py310-cu128.txt
```

It is a WSL2/Linux Python 3.10 CUDA reference constraints file. If pip cannot
resolve CUDA-specific wheels on another machine, run:

```bash
./install.sh
```

For a light CPU smoke setup:

```bash
./install.sh --lite --no-menagerie
```

## Minimal Validation

Fast CPU-oriented check:

```bash
make smoke
```

GPU smoke check:

```bash
GRANULAR_SMOKE_DEVICE=cuda:0 make smoke
```

MuJoCo bridge smoke check after Menagerie is installed:

```bash
make smoke-bridge
```

## Headline Reproduction

Regenerate the project-page videos and figures:

```bash
make render-density-eef
make sim2sim-wedge
make excavation-policy
make wild-robustness-stress
make wild-review-audit
```

Expected headline metrics for the full wild-material stress run:

| Metric | Expected value |
| --- | ---: |
| Family accuracy | 0.931 |
| Worst-family accuracy | 0.877 |
| Mean property nMAE | 0.082 |
| Coverage error | 0.046 |

Expected reviewer-audit quick variants:

| Variant | Family Acc | Worst | Gate |
| --- | ---: | ---: | --- |
| main | 0.859 | 0.812 | pass |
| no property-family head | 0.849 | 0.812 | pass |
| sensor only | 0.740 | 0.479 | fail |
| vision only | 0.802 | 0.729 | pass |
| context only | 0.250 | 0.021 | fail |
| no sigma calibration | 0.859 | 0.792 | pass |

Small floating-point and render differences can occur across CPU/GPU backends,
but the pass/fail interpretation should remain stable.

## Artifact Locations

Canonical generated outputs:

```text
outputs/density_mujoco_eef_render/
outputs/sim2sim_bulldozing_wedge/
outputs/excavation_policy_compare/
outputs/wild_material_robustness_stress/
outputs/wild_review_audit/
```

Project-page copies:

```text
docs/assets/videos/excavation_policy_compare.mp4
docs/assets/videos/density_mujoco_eef_property_overlay.mp4
docs/assets/videos/sim2sim_bulldozing_wedge.mp4
docs/assets/videos/density_mujoco_eef_render.mp4
docs/assets/posters/
docs/assets/figures/
docs/assets/brand/
```

The committed project-page videos are re-encoded as H.264/yuv420p with faststart
metadata so GitHub Pages can display them without the black `0:00` playback
failure.

## Packaging

Build a shareable bundle:

```bash
python scripts/package_demo_artifacts.py
```

This writes:

```text
dist/granular-robot-demo-artifacts-<git-sha>.zip
```

The bundle includes videos, previews, logs, configs, and a SHA-256 manifest.
Experiment sequence outputs and very large USD/PLY files are excluded by
default. Include them only when those artifacts are under review:

```bash
python scripts/package_demo_artifacts.py --include-experiments
python scripts/package_demo_artifacts.py --include-heavy-usd
```

## Hardware Expectations

The compact Warp MPM demos can run wherever Warp initializes. The Newton bridge
and full MuJoCo rendering path are expected to run best on WSL2/Linux with a
CUDA-capable NVIDIA GPU and EGL-capable headless rendering.

Reference machine:

```text
warp-lang 1.13.0
mujoco 3.8.1
mujoco-warp 3.8.1
newton 1.2.0
torch 2.11.0+cu128
NVIDIA RTX 5060
```

## Reproducibility Boundaries

Claimed now:

- Synthetic Sim2Sim material-family and property estimation.
- GT-versus-estimated material validation in MPM tasks.
- Property-aware excavation behavior in simulation.
- Reviewer-audit checks for leakage, modality dependence, and calibration.

Not claimed yet:

- Real-world granular calibration.
- Real2Sim transfer.
- Fully validated SI-unit soil mechanics.
- A deployed robot excavation controller.
