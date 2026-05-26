# Granular Robot MPM Sandbox

This repository is a reproducible sandbox for studying robot interaction with granular media. The current baseline is a standalone 3D Warp MLS-MPM sand engine with an SDF blade, 6D reaction wrench logging, and diagnostic renders. The forward path is Newton-first: use Newton's maintained Warp MPM implementation for richer granular physics, USD export, and robot coupling, while keeping the local Warp engine as a compact reference model.

The goal is not to hide granular media behind a robot policy. The goal is to make the granular medium physically inspectable first, then build inference and policy layers on top.

## Current Demo

Run the 3D blade interaction demo:

```bash
/root/human2robot/.venvs/act_diverse/bin/python scripts/run_3d_blade_demo.py --config configs/sand3d_blade_demo.json
```

Run the MuJoCo Franka render coupled to the 3D MPM sand engine:

```bash
/root/human2robot/.venvs/act_diverse/bin/python scripts/run_mujoco_3d_mpm_cosim.py
```

Run the density-style renderer, which avoids drawing MPM material points as bead-like spheres:

```bash
/root/human2robot/.venvs/act_diverse/bin/python scripts/run_3d_density_render_demo.py
```

Run the Newton MPM spike and produce a local preview from Newton's USD particles:

```bash
/root/human2robot/.venvs/act_diverse/bin/python scripts/run_newton_mpm_spike.py --example mpm_granular --num-frames 48 --voxel-size 0.05
```

Run Newton's rigid-MPM two-way coupling example:

```bash
/root/human2robot/.venvs/act_diverse/bin/python scripts/run_newton_mpm_spike.py --example mpm_twoway_coupling --output-dir outputs/newton_mpm_twoway --num-frames 48
```

Run the MuJoCo Franka to Newton MPM bridge:

```bash
/root/human2robot/.venvs/act_diverse/bin/python scripts/run_mujoco_newton_mpm_bridge.py --voxel-size 0.032 --particles-per-cell 3.0
```

Use point-splat rendering only as a particle/debug view:

```bash
/root/human2robot/.venvs/act_diverse/bin/python scripts/run_mujoco_newton_mpm_bridge.py --sand-render-mode point --voxel-size 0.032 --particles-per-cell 3.0 --render-radius 2 --render-blur 0.85 --alpha-blur 0.45
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
opencv-python
numpy
```

For the standalone 3D MPM demo, only `warp-lang`, `numpy`, and `opencv-python` are required. MuJoCo is needed for the older Franka coupling prototype.

Optional Newton spike dependencies:

```bash
pip install "newton[examples]>=1.2" "usd-core>=26.5"
```

## Newton Direction

Newton is the preferred backend for the next phase because it already provides maintained 3D MPM granular examples, rigid-MPM two-way coupling, MuJoCo/Warp-adjacent infrastructure, and USD export for serious rendering/analysis pipelines.

The MuJoCo-Newton bridge defaults to a camera-space density renderer. The older standalone `sand3d_density_render.mp4` can look better for sand alone because it is an orthographic top/side diagnostic that can directly shade a height field. The bridge render has the harder job of sharing a perspective camera and depth ordering with the robot. Point-splat mode is kept only for debugging material-point positions.

Immediate next targets:

```text
wrap Newton MPM runs in project-native configs
replace the generic Newton collider with a controllable intrusion tool
log particle state, tool pose, and reaction-like contact signals per frame
connect the tool trajectory to a Franka/MJCF or Newton robot controller
render camera RGB/depth from USD or an external renderer while keeping particle logs
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
