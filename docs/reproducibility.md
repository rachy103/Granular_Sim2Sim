# Reproducibility Contract

This project separates source, external assets, and generated artifacts.

## Source Repository

Tracked in git:

- Python source under `src/granular_mpm/`
- Demo and packaging entry points under `scripts/`
- Versioned configs under `configs/`
- Research notes under `docs/`
- Install, smoke, and CI definitions

Ignored by git:

- `outputs/` generated videos, logs, USD, PLY, and previews
- `dist/` packaged artifact bundles
- `mujoco_menagerie/` external robot assets
- Python virtual environments and caches

## Install Path

Fresh clone path:

```bash
git clone https://github.com/rachy103/Granular_Robot.git
cd Granular_Robot
./install.sh
```

The installer creates `.venv`, installs the package, downloads MuJoCo Menagerie
at the commit recorded in `configs/external_assets.json`, and runs import tests.

For the reference Python package versions tested in this repo:

```bash
./install.sh --locked
```

This applies `constraints/reference-linux-py310-cu128.txt` during installation.
It is a WSL2/Linux Python 3.10 CUDA reference lock; if pip cannot resolve a
CUDA-specific wheel on a different machine, use the normal installer and compare
the resulting environment against the constraints file.

A CPU-oriented install can skip MuJoCo assets:

```bash
./install.sh --lite --no-menagerie
```

## Reproduction Commands

Fast check using Warp CPU by default:

```bash
make smoke
```

Use `GRANULAR_SMOKE_DEVICE=cuda:0 make smoke` to run the same smoke path on GPU.

Bridge smoke check after MuJoCo Menagerie is installed:

```bash
make smoke-bridge
```

Publishable local bundle:

```bash
make demo
```

The full demo regenerates:

- `outputs/3d_mpm_density_render/sand3d_density_render.mp4`
- `outputs/3d_mpm_blade/sand3d_blade_interaction.mp4`
- `outputs/mujoco_newton_mpm_bridge/mujoco_franka_newton_mpm_bridge.mp4`
- `outputs/mujoco_newton_mpm_bridge/mujoco_robot_pass.mp4`
- `outputs/mujoco_newton_mpm_bridge/newton_mpm_sand_camera_layer.mp4`
- `outputs/mujoco_newton_mpm_bridge/newton_mpm_bridge_log.npz`

Then it writes a shareable zip:

```text
dist/granular-robot-demo-artifacts-<git-sha>.zip
```

## Artifact Policy

The source repo should remain lightweight. Generated videos and logs can be
shared through Google Drive or GitHub Releases using:

```bash
python scripts/package_demo_artifacts.py
```

Very large USD/PLY files are excluded by default. Include them only when the
renderer or particle export itself is under review:

```bash
python scripts/package_demo_artifacts.py --include-heavy-usd
```

Use Git LFS only for assets that are not practically reproducible and must be
versioned with code. Current MuJoCo assets are reproducible from the pinned
`google-deepmind/mujoco_menagerie` commit in `configs/external_assets.json`, so
they stay outside git.

## Known Hardware Expectations

The compact Warp MPM demos can run wherever Warp initializes. The Newton bridge
is expected to run on WSL2/Linux with a CUDA-capable NVIDIA GPU and EGL-capable
headless MuJoCo rendering. The current reference machine used:

```text
warp-lang 1.13.0
mujoco 3.8.1
mujoco-warp 3.8.1
newton 1.2.0
NVIDIA RTX 5060
```
