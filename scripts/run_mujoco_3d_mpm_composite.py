from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm import SandMPM3D, SandMPM3DConfig
from granular_mpm.composite_render import composite_sand, render_sand_layer
from scripts.run_mujoco_3d_mpm_cosim import (
    MENAGERIE_PANDA,
    PANDA_XML,
    draw_overlay,
    planned_ctrl,
    solve_waypoints,
    tool_state_from_mujoco,
    write_sheet,
    write_video,
)


OUT_DIR = ROOT / "outputs" / "mujoco_3d_mpm_composite"
SCENE_XML = OUT_DIR / "franka_3d_mpm_composite_scene.xml"
VIDEO_PATH = OUT_DIR / "mujoco_franka_mpm_composite.mp4"
PREVIEW_PATH = OUT_DIR / "mujoco_franka_mpm_composite_preview.png"
SHEET_PATH = OUT_DIR / "mujoco_franka_mpm_composite_sheet.png"


def make_scene_xml() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = PANDA_XML.read_text(encoding="utf-8")
    text = text.replace(
        'meshdir="assets"',
        f'meshdir="{(MENAGERIE_PANDA / "assets").as_posix()}"',
        1,
    )
    text = text.replace(
        '<option integrator="implicitfast"/>',
        '<option timestep="0.004" gravity="0 0 -9.81" cone="pyramidal" '
        'solver="CG" iterations="35" ls_iterations="40" integrator="implicitfast"/>',
        1,
    )
    text = text.replace(
        "  <asset>",
        '  <visual>\n'
        '    <headlight diffuse="0.58 0.58 0.58" ambient="0.31 0.31 0.31" specular="0.1 0.1 0.1"/>\n'
        '    <global offwidth="1280" offheight="720" azimuth="124" elevation="-25"/>\n'
        '  </visual>\n\n'
        "  <asset>",
        1,
    )

    tool = """
                      <geom name="mpm3d_tool_handle" type="capsule" fromto="0 0 0.035 0 0 0.145"
                        size="0.007" rgba="0.08 0.09 0.10 1" contype="0" conaffinity="0"/>
                      <geom name="mpm3d_blade" type="box" pos="0 0 0.168"
                        size="0.082 0.122 0.015" rgba="0.02 0.025 0.028 1"
                        contype="0" conaffinity="0"/>
                      <site name="mpm3d_blade_center" pos="0 0 0.168" size="0.012"
                        rgba="1.0 0.58 0.16 1"/>
"""
    text = text.replace(
        '                      <body name="left_finger" pos="0 0 0.0584">',
        tool + '                      <body name="left_finger" pos="0 0 0.0584">',
        1,
    )

    world = """
    <light name="mpm3d_key" pos="0.55 -0.65 1.55" dir="-0.35 0.20 -1" directional="true"/>
    <camera name="mpm3d_overview" pos="1.12 -0.90 0.72" xyaxes="0.63 0.78 0 -0.31 0.25 0.92"/>
    <geom name="floor" type="plane" size="1.30 1.30 0.04" rgba="0.22 0.25 0.27 1"
      condim="3" friction="1.2 0.03 0.0002"/>
    <geom name="tray_back" type="box" pos="0.52 0.465 0.046" size="0.39 0.010 0.046"
      rgba="0.16 0.17 0.18 1" contype="0" conaffinity="0"/>
    <geom name="tray_front" type="box" pos="0.52 0.095 0.046" size="0.39 0.010 0.046"
      rgba="0.16 0.17 0.18 1" contype="0" conaffinity="0"/>
    <geom name="tray_left" type="box" pos="0.145 0.280 0.046" size="0.010 0.195 0.046"
      rgba="0.16 0.17 0.18 1" contype="0" conaffinity="0"/>
    <geom name="tray_right" type="box" pos="0.895 0.280 0.046" size="0.010 0.195 0.046"
      rgba="0.16 0.17 0.18 1" contype="0" conaffinity="0"/>
"""
    text = text.replace("</worldbody>", world + "  </worldbody>", 1)

    replacements = {
        'gainprm="4500" biasprm="0 -4500 -450"': 'gainprm="1700" biasprm="0 -1700 -155"',
        'gainprm="3500" biasprm="0 -3500 -350"': 'gainprm="1450" biasprm="0 -1450 -125"',
        'gainprm="2000" biasprm="0 -2000 -200"': 'gainprm="820" biasprm="0 -820 -78"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    SCENE_XML.write_text(text, encoding="utf-8")


def render_rgb_depth(renderer: mujoco.Renderer, data: mujoco.MjData, camera: str) -> tuple[np.ndarray, np.ndarray]:
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return rgb, depth


def run() -> None:
    make_scene_xml()
    model = mujoco.MjModel.from_xml_path(SCENE_XML.as_posix())
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.qpos[7:9] = 0.02

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mpm3d_blade_center")
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "mpm3d_overview")
    if min(site_id, hand_id, camera_id) < 0:
        raise RuntimeError("Missing MuJoCo site/body/camera")

    waypoints = solve_waypoints(model, site_id)
    data.qpos[:7] = waypoints[0][:7]
    data.qpos[7:9] = 0.02
    data.ctrl[:] = planned_ctrl(waypoints, 0.0, 1.0)
    mujoco.mj_forward(model, data)

    mpm_cfg = SandMPM3DConfig(dt=8.0e-4, seed=7)
    mpm = SandMPM3D(mpm_cfg, device="cuda:0")
    mj_dt = float(model.opt.timestep)
    mpm_substeps = max(1, int(round(mj_dt / mpm_cfg.dt)))
    frames_n = 60
    steps_per_frame = 7
    total_time = frames_n * steps_per_frame * mj_dt
    force_scale = 0.0012
    torque_scale = 0.0006
    print(
        f"composite render MuJoCo dt={mj_dt:.4f} MPM dt={mpm_cfg.dt:.4f} "
        f"particles={mpm.n_particles} mpm_substeps/mj_step={mpm_substeps}"
    )

    renderer = mujoco.Renderer(model, height=720, width=1280)
    frames: list[np.ndarray] = []
    prev_tool = tool_state_from_mujoco(
        data,
        site_id,
        data.site_xpos[site_id].astype(np.float32).copy(),
        0.0,
        mj_dt,
    )
    applied_force_world = np.zeros(3, dtype=np.float64)

    for frame_id in range(frames_n):
        for _ in range(steps_per_frame):
            sim_time = float(data.time)
            data.ctrl[:] = planned_ctrl(waypoints, sim_time, total_time)
            mujoco.mj_forward(model, data)
            tool = tool_state_from_mujoco(data, site_id, prev_tool.center, prev_tool.angle, mj_dt)
            raw_wrench = mpm.step(tool, substeps=mpm_substeps)
            force = raw_wrench[:3].astype(np.float64) * force_scale
            mag = float(np.linalg.norm(force))
            if mag > 85.0:
                force *= 85.0 / mag
            site_pos = data.site_xpos[site_id].copy()
            body_pos = data.xpos[hand_id].copy()
            torque = np.cross(site_pos - body_pos, force) + raw_wrench[3:6].astype(np.float64) * torque_scale
            data.xfrc_applied[:] = 0.0
            data.xfrc_applied[hand_id, :3] = force
            data.xfrc_applied[hand_id, 3:6] = np.clip(torque, -13.0, 13.0)
            mujoco.mj_step(model, data)
            applied_force_world = force
            prev_tool = tool

        mujoco.mj_forward(model, data)
        robot_rgb, robot_depth = render_rgb_depth(renderer, data, "mpm3d_overview")
        pos = mpm.positions()
        sand_rgb, sand_alpha, sand_depth = render_sand_layer(model, data, camera_id, pos, 1280, 720, radius=5)
        composite = composite_sand(robot_rgb, robot_depth, sand_rgb, sand_alpha, sand_depth)
        desired = planned_ctrl(waypoints, float(data.time), total_time)
        qerr = float(np.linalg.norm(desired[:7] - data.qpos[:7]))
        composite_bgr = draw_overlay(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR), applied_force_world, qerr, frame_id)
        frames.append(cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB))

        if frame_id % 10 == 0:
            print(
                f"frame={frame_id:03d} t={data.time:.3f} "
                f"|F|={np.linalg.norm(applied_force_world):.2f} qerr={qerr:.3f}"
            )

    renderer.close()
    write_video(VIDEO_PATH, frames)
    cv2.imwrite(PREVIEW_PATH.as_posix(), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    write_sheet(SHEET_PATH, frames)
    print(f"scene={SCENE_XML}")
    print(f"video={VIDEO_PATH}")
    print(f"preview={PREVIEW_PATH}")
    print(f"sheet={SHEET_PATH}")


if __name__ == "__main__":
    run()
