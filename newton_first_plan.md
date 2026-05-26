# Newton-First Granular Robot Plan

This repo will use Newton as the primary simulation backend for the next phase.
The local Warp MPM implementation stays useful as a small, inspectable reference,
but Newton is the better path for maintained granular MPM, rigid coupling, USD
export, and eventual robot integration.

## Why Newton

Newton already exposes the pieces this project needs:

```text
3D granular MPM examples
rigid-MPM two-way coupling examples
USD export for particle-level rendering and analysis
Warp-based execution path
MuJoCo/Warp-adjacent tooling
```

Genesis remains a comparison point for quick native RGB previews, but its PyPI
install currently gives only rasterized particle/reconstruction views in this
environment. Newton's USD path is a better fit for high-quality sand rendering
through Omniverse, Houdini, Blender/USD tooling, or a custom point renderer.

## Near-Term Milestones

1. Project-native Newton configs

   Move the spike script from command-line-only parameters to checked-in JSON
   configs for granular material, voxel size, tool trajectory, frame count, and
   output paths.

2. Controlled intrusion tool

   Replace the stock Newton example collider with a tool primitive we control:
   blade, shovel, flat plate, cylinder, or gripper fingertip. Log pose and
   velocity for every frame.

3. Sensor-quality logs

   Export particle position, velocity, width, color/material id, tool pose, and
   contact/impulse proxies into NPZ/CSV alongside USD. The learning stack should
   train from these logs before relying on camera-only RGB.

4. Robot trajectory bridge

   Drive the Newton tool from a Franka/MJCF or Newton robot controller. The first
   version can be kinematic end-effector replay; only after that should it become
   dynamically coupled.

5. Rendering bridge

   Treat USD as the canonical render artifact. Generate quick MP4 previews in
   this repo, and leave hooks for external renderers that can produce better
   RGB/depth without losing particle observability.

## Research Guardrails

Do not treat the sand as a visual effect. Each rendered frame should be backed by
particle logs and material parameters. Vision can come later, but the first
research objects are particle state, intrusion geometry, reaction signals, and
material sensitivity curves.

Before policy learning, validate:

```text
intrusion drag versus depth and velocity
pile deformation under repeated sweeps
parameter sensitivity to friction/yield/voxel size
repeatability under fixed seed and fixed tool trajectory
camera view registration to the particle/USD state
```
