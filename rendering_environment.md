# Fixed Rendering Environment

This document fixes the canonical rendering path for showing the MPM density
rollout with a Franka end-effector overlay.

## Canonical Command

```bash
python scripts/render_density_mujoco_eef_render.py --config configs/rendering/density_mujoco_eef_render_fixed.json
```

The same command is available through:

```bash
make render-density-eef
```

## Fixed Inputs

The config lives at:

```text
configs/rendering/density_mujoco_eef_render_fixed.json
```

It fixes:

- density video: `outputs/3d_mpm_density_render/sand3d_density_render.mp4`
- property rollout CSV:
  `outputs/online_mohr_coulomb_bestval_quick/rollout_predictions.csv`
- frame count: `60`
- MPM timing: `substeps_per_frame=34`, `mpm_dt=0.0008`
- crop mode: `eef_path`
- renderer alpha: `0.98`

## Geometry Contract

The renderer does not use a MuJoCo perspective camera and then crop pixels into
the MPM diagnostic view. That path caused scale and contact-point mismatches.

Instead, the script loads the Panda hand from MuJoCo, extracts only the
end-effector visual mesh plus the custom `mpm3d_tool_handle` and `mpm3d_blade`
geoms, converts those surfaces into blade-site local coordinates, and rasterizes
the actual mesh faces into the density renderer's top and side orthographic
projection coordinates.

This keeps one shared interaction point:

```text
MPM blade center == MuJoCo mpm3d_blade_center site == rendered EEF origin
```

The robot base and full arm are intentionally excluded. The fixed view is the
local sand/tool interaction region, not a whole-robot camera shot.

## Outputs

The canonical run writes:

```text
outputs/density_mujoco_eef_render/density_mujoco_eef_render.mp4
outputs/density_mujoco_eef_render/density_mujoco_eef_render_preview.png
outputs/density_mujoco_eef_render/density_mujoco_eef_render_sheet.png
outputs/density_mujoco_eef_render/density_mujoco_eef_property_overlay.mp4
outputs/density_mujoco_eef_render/density_mujoco_eef_property_overlay_preview.png
outputs/density_mujoco_eef_render/density_mujoco_eef_render_metadata.json
```

The metadata records the config path, MuJoCo version, crop rectangles, output
video size, and every rendered mesh part with vertex and face counts.

## What To Avoid

- Do not use the older camera-composite scripts as the main result path.
  They are useful for debugging, but they reintroduce camera-scale alignment
  problems.
- Do not show the full Franka base for this density rollout. The density view is
  zoomed to the tray/tool interaction, so a physically consistent robot view
  should show only the EEF/tool region.
- Do not treat the black blade as an artifact. It is the actual custom
  `mpm3d_blade` geom driving the MPM interaction.
