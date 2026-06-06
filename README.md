# Granular Robot MPM Sandbox

This repository is a reproducible sandbox for studying robot interaction with granular media. The current baseline is a standalone 3D Warp MLS-MPM sand engine with an SDF blade, 6D reaction wrench logging, and diagnostic renders. The forward path is Newton-first: use Newton's maintained Warp MPM implementation for richer granular physics, USD export, and robot coupling, while keeping the local Warp engine as a compact reference model.

The goal is not to hide granular media behind a robot policy. The goal is to make the granular medium physically inspectable first, then build inference and policy layers on top.

## Fresh Setup

The repo is intended to run from a normal Python environment. The tested path is
WSL2 plus an NVIDIA CUDA-capable GPU, but the code does not depend on the local
venv name used during development.

```bash
git clone https://github.com/rachy103/Granular_Robot.git
cd Granular_Robot

chmod +x install.sh
./install.sh
```

The installer creates `.venv`, installs the Python package with MuJoCo, Newton,
learning, and test extras, shallow-clones `google-deepmind/mujoco_menagerie`,
and runs the import tests. For the exact tested package constraints, use:

```bash
./install.sh --locked
```

The lock lives at `constraints/reference-linux-py310-cu128.txt`. It is a WSL2 /
Linux Python 3.10 CUDA reference constraints file, so use the normal installer on
other platforms if pip cannot resolve CUDA-specific wheels. For a lighter
install:

```bash
./install.sh --lite --no-menagerie
```

If EGL rendering is available, the MuJoCo scripts default to headless rendering
through `MUJOCO_GL=egl`. On a CPU-only machine, start with the standalone density
demo before running the Newton bridge.

Fast checks and reproducible demo runs are available through `make`:

```bash
make smoke          # pytest plus a tiny density-render run
make smoke-bridge   # tiny MuJoCo-Newton bridge run
make demo           # regenerate the publishable demo videos and artifact zip
```

`make smoke` uses Warp CPU by default. For GPU smoke runs, use
`GRANULAR_SMOKE_DEVICE=cuda:0 make smoke`.

See `docs/reproducibility.md` for the full source/asset/artifact contract.

For the pinned reference environment and full demo reproduction, run:

```bash
git clone https://github.com/rachy103/Granular_Robot.git
cd Granular_Robot
./install.sh --locked
make smoke
make demo
```

## Experiment Pipeline

Named experiment sequences are run through a single wrapper:

```bash
python scripts/run_experiment_sequence.py --config configs/experiments/reference_heightfield_intrusion.json
```

For a quick CPU-oriented check:

```bash
make experiment-smoke
make pipeline-smoke
make sweep-smoke
```

Each sequence writes a fixed layout under `outputs/experiments/<sequence_name>/`:

```text
config/              resolved experiment and stage configs
video_set/           rendered videos and previews grouped by stage
dataset_metrics/     dataset/video/force/particle summaries
training_metrics/    baseline metrics, representation loss, MDN metrics, model
inference_results/   prediction CSVs, posterior plots, inference metrics
logs/                command logs from each stage
runs/                raw per-stage outputs
```

The wrapper also builds the learning tensor
`X in R^(N x T x 12)` from 6D wrench plus 6D end-effector kinematics, using
50 Hz resampling and train-split z-score normalization by default. The learned
path is a compact 1D-CNN/Transformer encoder with InfoNCE representation
learning followed by an MDN decoder for `phi_deg` and `cohesion_kpa`.
Set `learning.num_mixtures` to `1` in the experiment config for a single
Gaussian decoder.

See `docs/experiment_pipeline.md` for the full wrapper contract.

For the next multimodal learning phase, see
`docs/multimodal_mohr_coulomb_architecture.md`. It specifies the proposed
2D-vision plus robot-sensor architecture for estimating four
Mohr-Coulomb-style soil parameters from blade digging interactions.
The first implementation is an online belief-update model that carries the
material posterior forward frame by frame instead of re-inferring material
parameters independently at each frame:

```bash
python scripts/train_online_mohr_coulomb.py --quick --output-dir outputs/online_mohr_coulomb_bestval_quick
```

The current canonical quick result restores the best validation checkpoint
before export and writes the posterior rollout used by the render demos to
`outputs/online_mohr_coulomb_bestval_quick/rollout_predictions.csv`.

For amortized inference data generation, run a Latin-hypercube sweep over
material and action nuisance variables:

```bash
python scripts/run_property_sweep.py --config configs/sweeps/lhs_property_sweep.json
```

The sweep writes `outputs/sweeps/<sweep_name>/samples.csv`, per-sample sequence
folders, one globally normalized aggregate tensor dataset, aggregate training
metrics, and aggregate inference results.

Inspect whether material labels actually separate the force traces:

```bash
python scripts/analyze_sweep_scatter.py --sweep-root outputs/sweeps/<sweep_name>
```

The default sweep uses a paired material-action design: multiple action
variations are repeated for each sampled material, and aggregate train/test
splits are grouped by material.

## Artifact Policy

Large files are treated as reproducible artifacts, not source. The repo does not
track `outputs/` or `mujoco_menagerie/`: MuJoCo Menagerie is downloaded by
`install.sh` at the pinned commit recorded in `configs/external_assets.json`,
and demo videos/logs are regenerated by the scripts below.

To share a ready-made run through Google Drive or a GitHub Release, build a small
artifact bundle:

```bash
source .venv/bin/activate
python scripts/package_demo_artifacts.py
```

This writes `dist/granular-robot-demo-artifacts-<git-sha>.zip` with videos,
previews, logs, configs, and a SHA-256 manifest. Experiment/sweep outputs and
very large Newton USD/PLY files are excluded by default; include them explicitly
when needed:

```bash
python scripts/package_demo_artifacts.py --include-experiments
python scripts/package_demo_artifacts.py --include-heavy-usd
```

Use Git LFS only if a generated USD/PLY/video must live inside the git history.
For normal reproduction, prefer the installer plus artifact bundle.

## Current Demo

Run the 3D blade interaction demo:

```bash
python scripts/run_3d_blade_demo.py --config configs/sand3d_blade_demo.json
```

Run the MuJoCo Franka render coupled to the 3D MPM sand engine:

```bash
python scripts/run_mujoco_3d_mpm_cosim.py
```

Run the density-style renderer, which avoids drawing MPM material points as bead-like spheres:

```bash
python scripts/run_3d_density_render_demo.py
```

Render the fixed density-plus-MuJoCo-EEF view with property-estimation graphs:

```bash
make render-density-eef
```

This uses `configs/rendering/density_mujoco_eef_render_fixed.json` and writes
the canonical overlay artifacts under `outputs/density_mujoco_eef_render/`.
See `docs/rendering_environment.md` for the fixed rendering contract.

Compare GT granular properties against estimated properties in the same MPM
bulldozing-wedge probing task:

```bash
python scripts/render_sim2sim_property_compare.py --config configs/rendering/sim2sim_bulldozing_wedge.json
```

This writes a side-by-side Sim2Sim validation video under
`outputs/sim2sim_bulldozing_wedge/`. See
`docs/sim2sim_property_validation.md` for the material mapping and
interpretation.

Compare an excavation behavior that ignores the property model against a
property-aware excavation behavior:

```bash
python scripts/render_excavation_policy_compare.py --config configs/rendering/excavation_policy_compare.json
```

This writes the model/no-model excavation comparison under
`outputs/excavation_policy_compare/`. See
`docs/excavation_policy_with_property_prediction.md` for the policy heuristic
and metrics.

The current comparison uses the same GT granular material on both sides. The
left side ignores the estimated material and runs a fixed nominal excavation;
the right side uses the estimated four-property posterior to choose a shallower,
safer excavation plan.

Run the hostile four-family material robustness benchmark:

```bash
make wild-robustness-stress
```

This trains and evaluates `gravel`, `sand`, `soil`, and `crunching` under
held-out actions, held-out materials, overlapping property ranges, and sensor
/ vision corruptions. The current stress result writes figures and metrics to
`outputs/wild_material_robustness_stress/`; see
`docs/wild_material_robustness_paper.md` for the paper-style draft.

Run the Newton MPM spike and produce a local preview from Newton's USD particles:

```bash
python scripts/run_newton_mpm_spike.py --example mpm_granular --num-frames 48 --voxel-size 0.05
```

Run Newton's rigid-MPM two-way coupling example:

```bash
python scripts/run_newton_mpm_spike.py --example mpm_twoway_coupling --output-dir outputs/newton_mpm_twoway --num-frames 48
```

Run the MuJoCo Franka to Newton MPM bridge:

```bash
python scripts/run_mujoco_newton_mpm_bridge.py --config configs/newton_bridge_heightfield.json
```

Use screen-density rendering or point-splat rendering for comparison/debug:

```bash
python scripts/run_mujoco_newton_mpm_bridge.py --config configs/newton_bridge_heightfield.json --sand-render-mode density
python scripts/run_mujoco_newton_mpm_bridge.py --config configs/newton_bridge_heightfield.json --sand-render-mode point --render-radius 2 --render-blur 0.85 --alpha-blur 0.45
```

Generated artifacts:

```text
outputs/3d_mpm_blade/sand3d_blade_interaction.mp4
outputs/3d_mpm_blade/sand3d_blade_preview.png
outputs/3d_mpm_blade/sand3d_blade_contact_sheet.png
outputs/3d_mpm_blade/wrench_log.csv
outputs/3d_mpm_blade/final_state_and_wrench_log.npz
outputs/3d_mpm_blade/resolved_config.json
outputs/mujoco_3d_mpm_cosim/mujoco_franka_3d_mpm_interaction.mp4
outputs/3d_mpm_density_render/sand3d_density_render.mp4
outputs/newton_mpm_spike/mpm_granular.usd
outputs/newton_mpm_spike/mpm_granular_preview.mp4
outputs/newton_mpm_twoway/mpm_twoway_coupling.usd
outputs/newton_mpm_twoway/mpm_twoway_coupling_preview.mp4
outputs/mujoco_newton_mpm_bridge/mujoco_franka_newton_mpm_bridge.mp4
outputs/mujoco_newton_mpm_bridge/mujoco_robot_pass.mp4
outputs/mujoco_newton_mpm_bridge/newton_mpm_sand_camera_layer.mp4
outputs/mujoco_newton_mpm_bridge/newton_mpm_bridge_log.npz
```

The video contains top, side, and front projections of the same 3D material point state. The orange arrow and force plot show the reaction wrench computed from MPM contact impulses.

## Repository Layout

```text
configs/                         Reproducible run configs
docs/                            Research basis and modeling notes
scripts/                         Entry points for demos and experiments
src/granular_mpm/                MPM kernels, solver wrappers, visualization
outputs/                         Generated videos, logs, and snapshots
dist/                            Packaged generated artifact bundles
```

Legacy prototypes are kept at the repository root:

```text
warp_sand_mpm.py                 2D standalone MPM prototype
warp_sand_mpm_coupled.py         2D MPM with shovel body feedback
mujoco_mpm_cosim.py              2D MPM coupled to MuJoCo Franka
```

## Dependencies

The tested environment is the WSL distro `Ubuntu-Human2Robot` with:

```text
warp-lang 1.13.0
mujoco 3.8.1
mujoco-warp 3.8.1
opencv-python-headless
numpy
```

For the standalone 3D MPM demo, only `warp-lang`, `numpy`, and `opencv-python-headless` are required. MuJoCo is needed for the older Franka coupling prototype.

Optional Newton spike dependencies:

```bash
python -m pip install -e ".[newton-examples]"
```

## Newton Direction

Newton is the preferred backend for the next phase because it already provides maintained 3D MPM granular examples, rigid-MPM two-way coupling, MuJoCo/Warp-adjacent infrastructure, and USD export for serious rendering/analysis pipelines.

The MuJoCo-Newton bridge writes separate robot, sand, and composite videos. Height-field mode reconstructs a world-space sand slab from Newton particles, extruding the surface down to a base plane so the rendered sand has visible volume instead of only a thin top sheet. Density mode renders a camera-space density layer; point-splat mode is kept only for debugging material-point positions. The older standalone `sand3d_density_render.mp4` can still look better for sand alone because it is an orthographic top/side diagnostic that can directly shade a height field. The bridge render has the harder job of sharing a perspective camera and depth ordering with the robot.

Immediate next targets:

```text
replace the bridge's preview renderer with a proper USD/Blender/Omniverse render path
turn the kinematic intrusion sequence into configurable trajectory primitives
log particle state, tool pose, and reaction-like contact signals per frame
connect the tool trajectory to a Franka/MJCF or Newton robot controller
validate intrusion and drag force curves against material parameters
```

## Model Scope

Implemented now:

```text
3D MLS-MPM P2G/grid/G2P loop
3D deformation gradient and APIC affine field
Drucker-Prager-like log-strain plastic projection
oriented-box SDF blade contact
Coulomb tangential projection
6D tool wrench from contact impulse
top/side/front diagnostic renders
wrench CSV and final state NPZ export
```

Not yet claimed:

```text
calibrated SI-unit sand
full Drucker-Prager return mapping
cohesive/moist soil
3D MuJoCo robot coupling to this new 3D engine
validated real-world transfer
```

The next research step is to validate intrusion and drag force curves against material parameters before adding vision or learned force sensing.
