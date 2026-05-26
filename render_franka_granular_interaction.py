from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import cv2
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp


ROOT = Path(__file__).resolve().parent
MENAGERIE_PANDA = ROOT / "mujoco_menagerie" / "franka_emika_panda"
PANDA_XML = MENAGERIE_PANDA / "panda.xml"
OUT_DIR = ROOT / "outputs" / "franka_granular"
SCENE_XML = OUT_DIR / "franka_granular_scene.xml"
VIDEO_PATH = OUT_DIR / "franka_granular_interaction.mp4"
PREVIEW_PATH = OUT_DIR / "franka_granular_preview.png"


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def grain_bed() -> str:
    rng = np.random.default_rng(22)
    lines: list[str] = []
    radius = 0.014
    spacing = radius * 2.08
    count_x, count_y, count_z = 9, 7, 3
    for ix in range(count_x):
        for iy in range(count_y):
            for iz in range(count_z):
                x = 0.52 + (ix - (count_x - 1) * 0.5) * spacing
                y = (iy - (count_y - 1) * 0.5) * spacing
                z = 0.025 + iz * spacing
                x += rng.uniform(-0.003, 0.003)
                y += rng.uniform(-0.003, 0.003)
                name = f"grain_{ix}_{iy}_{iz}"
                color = rng.choice(
                    [
                        "0.70 0.54 0.32 1",
                        "0.60 0.45 0.27 1",
                        "0.78 0.63 0.38 1",
                    ]
                )
                lines.append(
                    f'    <body name="{name}" pos="{x:.5f} {y:.5f} {z:.5f}">\n'
                    f'      <freejoint name="{name}_free"/>\n'
                    f'      <geom name="{name}_geom" type="sphere" size="{radius:.5f}" '
                    'density="1550" condim="3" friction="1.2 0.04 0.0003" '
                    'solref="0.006 1" solimp="0.92 0.97 0.001" '
                    f'rgba="{color}"/>\n'
                    "    </body>"
                )
    return "\n".join(lines)


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
        '<option timestep="0.002" gravity="0 0 -9.81" cone="pyramidal" '
        'solver="CG" iterations="30" ls_iterations="40" integrator="implicitfast"/>',
        1,
    )
    text = text.replace(
        "  <asset>",
        '  <visual>\n'
        '    <headlight diffuse="0.55 0.55 0.55" ambient="0.30 0.30 0.30" specular="0.1 0.1 0.1"/>\n'
        '    <global offwidth="960" offheight="720" azimuth="135" elevation="-22"/>\n'
        '  </visual>\n\n'
        "  <asset>",
        1,
    )
    tool = """
                      <geom name="granular_rake_handle" type="capsule" fromto="0 0 0.055 0 0 0.150"
                        size="0.007" density="1200" condim="3" friction="1.1 0.03 0.0002"
                        rgba="0.08 0.09 0.10 1"/>
                      <geom name="granular_rake_blade" type="box" pos="0 0 0.175"
                        size="0.078 0.008 0.024" density="1600" condim="3"
                        friction="1.3 0.04 0.0002" rgba="0.02 0.03 0.035 1"/>
                      <site name="tool_tip" pos="0 0 0.175" size="0.01" rgba="1 0.15 0.05 1"/>
"""
    text = text.replace(
        '                      <body name="left_finger" pos="0 0 0.0584">',
        tool + '                      <body name="left_finger" pos="0 0 0.0584">',
        1,
    )
    world = f"""
    <light name="granular_key" pos="0.6 -0.7 1.4" dir="-0.4 0.3 -1" directional="true"/>
    <camera name="overview" pos="1.12 -1.02 0.72" xyaxes="0.68 0.73 0 -0.30 0.28 0.91"/>
    <geom name="floor" type="plane" size="1.3 1.3 0.04" rgba="0.22 0.25 0.27 1"
      condim="3" friction="1.2 0.03 0.0002"/>
    <geom name="tray_back" type="box" pos="0.52 0.145 0.045" size="0.24 0.010 0.045"
      rgba="0.18 0.19 0.20 1" condim="3" friction="1.1 0.03 0.0002"/>
    <geom name="tray_front" type="box" pos="0.52 -0.145 0.045" size="0.24 0.010 0.045"
      rgba="0.18 0.19 0.20 1" condim="3" friction="1.1 0.03 0.0002"/>
    <geom name="tray_left" type="box" pos="0.275 0 0.045" size="0.010 0.155 0.045"
      rgba="0.18 0.19 0.20 1" condim="3" friction="1.1 0.03 0.0002"/>
    <geom name="tray_right" type="box" pos="0.765 0 0.045" size="0.010 0.155 0.045"
      rgba="0.18 0.19 0.20 1" condim="3" friction="1.1 0.03 0.0002"/>
{grain_bed()}
"""
    text = text.replace("</worldbody>", world + "  </worldbody>", 1)
    SCENE_XML.write_text(text, encoding="utf-8")


def joint_ranges(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(model.nq, -np.inf)
    hi = np.full(model.nq, np.inf)
    for j in range(model.njnt):
        qadr = int(model.jnt_qposadr[j])
        if model.jnt_limited[j]:
            lo[qadr] = model.jnt_range[j, 0]
            hi[qadr] = model.jnt_range[j, 1]
    return lo, hi


def seed_grain_freejoints(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not name.startswith("grain_") or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qadr = int(model.jnt_qposadr[j])
        body_id = int(model.jnt_bodyid[j])
        data.qpos[qadr : qadr + 3] = model.body_pos[body_id]
        data.qpos[qadr + 3 : qadr + 7] = model.body_quat[body_id]


def solve_tool_waypoints(model: mujoco.MjModel) -> list[np.ndarray]:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    seed_grain_freejoints(model, data)
    q = data.qpos.copy()
    q[7:9] = 0.02
    lo, hi = joint_ranges(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
    targets = [
        np.array([0.34, 0.00, 0.16]),
        np.array([0.36, 0.00, 0.067]),
        np.array([0.70, 0.00, 0.063]),
        np.array([0.74, 0.00, 0.20]),
    ]
    waypoints: list[np.ndarray] = []
    for target in targets:
        for _ in range(180):
            data.qpos[:] = q
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            err = target - data.site_xpos[site_id]
            if np.linalg.norm(err) < 0.002:
                break
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            jac = jacp[:, :7]
            damping = 0.025
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), err)
            q[:7] += np.clip(dq, -0.045, 0.045)
            q[:7] = np.clip(q[:7], lo[:7], hi[:7])
            q[7:9] = 0.02
        waypoints.append(q.copy())
    return waypoints


def planned_ctrl(waypoints: list[np.ndarray], frame: int, total_frames: int) -> np.ndarray:
    # linger, descend, rake, lift
    breaks = [0.0, 0.16, 0.37, 0.78, 1.0]
    t = frame / max(total_frames - 1, 1)
    if t < breaks[1]:
        a, b, u = waypoints[0], waypoints[0], 0.0
    elif t < breaks[2]:
        a, b = waypoints[0], waypoints[1]
        u = smoothstep((t - breaks[1]) / (breaks[2] - breaks[1]))
    elif t < breaks[3]:
        a, b = waypoints[1], waypoints[2]
        u = smoothstep((t - breaks[2]) / (breaks[3] - breaks[2]))
    else:
        a, b = waypoints[2], waypoints[3]
        u = smoothstep((t - breaks[3]) / (breaks[4] - breaks[3]))
    q = (1.0 - u) * a + u * b
    ctrl = np.zeros(8, dtype=np.float32)
    ctrl[:7] = q[:7]
    ctrl[7] = 125.0
    return ctrl


def render_video() -> None:
    make_scene_xml()
    model = mujoco.MjModel.from_xml_path(SCENE_XML.as_posix())
    host_data = mujoco.MjData(model)
    render_data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, host_data, key_id)
    seed_grain_freejoints(model, host_data)
    host_data.qpos[7:9] = 0.02
    mujoco.mj_forward(model, host_data)

    waypoints = solve_tool_waypoints(model)
    host_data.qpos[:7] = waypoints[0][:7]
    host_data.qpos[7:9] = 0.02
    host_data.ctrl[:] = planned_ctrl(waypoints, 0, 1)
    mujoco.mj_forward(model, host_data)

    width, height = 960, 720
    total_frames = 150
    substeps = 7
    frames: list[np.ndarray] = []

    wp.init()
    renderer = None
    with wp.ScopedDevice("cuda:0"):
        model_wp = mjw.put_model(model)
        data_wp = mjw.put_data(model, host_data, nworld=1, nconmax=4096, njmax=8192)
        renderer = mujoco.Renderer(model, height=height, width=width)
        scene_option = getattr(renderer, "scene_option", getattr(renderer, "_scene_option", None))
        if scene_option is not None:
            scene_option.geomgroup[3] = 0
        for frame_idx in range(total_frames):
            ctrl = planned_ctrl(waypoints, frame_idx, total_frames).reshape(1, -1)
            data_wp.ctrl.assign(ctrl.astype(np.float32))
            for _ in range(substeps):
                mjw.step(model_wp, data_wp)
            wp.synchronize()
            mjw.get_data_into(render_data, model, data_wp, world_id=0)
            mujoco.mj_forward(model, render_data)
            renderer.update_scene(render_data, camera="overview")
            rgb = renderer.render()
            frames.append(rgb.copy())

    if renderer is not None:
        renderer.close()

    writer = cv2.VideoWriter(
        VIDEO_PATH.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {VIDEO_PATH}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    cv2.imwrite(PREVIEW_PATH.as_posix(), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    print(f"scene={SCENE_XML}")
    print(f"video={VIDEO_PATH}")
    print(f"preview={PREVIEW_PATH}")


if __name__ == "__main__":
    render_video()
