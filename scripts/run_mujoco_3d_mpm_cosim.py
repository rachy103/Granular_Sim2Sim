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
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm import SandMPM3D, SandMPM3DConfig, ToolState3D


MENAGERIE_PANDA = ROOT / "mujoco_menagerie" / "franka_emika_panda"
PANDA_XML = MENAGERIE_PANDA / "panda.xml"
OUT_DIR = ROOT / "outputs" / "mujoco_3d_mpm_cosim"
SCENE_XML = OUT_DIR / "franka_3d_mpm_scene.xml"
VIDEO_PATH = OUT_DIR / "mujoco_franka_3d_mpm_interaction.mp4"
PREVIEW_PATH = OUT_DIR / "mujoco_franka_3d_mpm_preview.png"
SHEET_PATH = OUT_DIR / "mujoco_franka_3d_mpm_contact_sheet.png"
VIS_PARTICLES = 2600


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def sand_particle_geoms(count: int = VIS_PARTICLES) -> str:
    lines: list[str] = []
    for i in range(count):
        lines.append(
            f'    <geom name="mpm3d_grain_{i:04d}" type="sphere" '
            'pos="0.10 0.10 -0.10" size="0.0046" '
            'rgba="0.74 0.58 0.32 0.92" contype="0" conaffinity="0"/>'
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

    world = f"""
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
{sand_particle_geoms()}
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


def joint_ranges(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(model.nq, -np.inf)
    hi = np.full(model.nq, np.inf)
    for j in range(model.njnt):
        qadr = int(model.jnt_qposadr[j])
        if model.jnt_limited[j]:
            lo[qadr] = model.jnt_range[j, 0]
            hi[qadr] = model.jnt_range[j, 1]
    return lo, hi


def solve_waypoints(model: mujoco.MjModel, site_id: int) -> list[np.ndarray]:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    q = data.qpos.copy()
    q[7:9] = 0.02
    lo, hi = joint_ranges(model)
    targets = [
        (np.array([0.30, 0.280, 0.390]), -0.78),
        (np.array([0.405, 0.280, 0.225]), -0.42),
        (np.array([0.735, 0.315, 0.225]), -0.31),
        (np.array([0.780, 0.280, 0.365]), -0.31),
    ]
    waypoints: list[np.ndarray] = []
    for pos_target, angle_target in targets:
        desired_axis = np.array([np.cos(angle_target), 0.0, -np.sin(angle_target)])
        for _ in range(320):
            data.qpos[:] = q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            pos_err = pos_target - data.site_xpos[site_id]
            mat = data.site_xmat[site_id].reshape(3, 3)
            long_axis = mat @ np.array([1.0, 0.0, 0.0])
            ori_err = np.cross(long_axis, desired_axis)
            err = np.concatenate([pos_err, 0.16 * ori_err])
            if np.linalg.norm(pos_err) < 0.006 and np.linalg.norm(ori_err) < 0.10:
                break
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            jac = np.vstack([jacp[:, :7], 0.16 * jacr[:, :7]])
            damping = 0.045
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
            q[:7] += np.clip(dq, -0.040, 0.040)
            q[:7] = np.clip(q[:7], lo[:7], hi[:7])
            q[7:9] = 0.02
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        waypoints.append(q.copy())
        print(
            f"waypoint target={pos_target} solved={data.site_xpos[site_id]} "
            f"err={np.linalg.norm(pos_target - data.site_xpos[site_id]):.4f}"
        )
    return waypoints


def planned_ctrl(waypoints: list[np.ndarray], sim_time: float, total_time: float) -> np.ndarray:
    breaks = [0.0, 0.16, 0.37, 0.79, 1.0]
    t = float(np.clip(sim_time / total_time, 0.0, 1.0))
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
    ctrl = np.zeros(8, dtype=np.float64)
    ctrl[:7] = q[:7]
    ctrl[7] = 125.0
    return ctrl


def tool_state_from_mujoco(
    data: mujoco.MjData,
    site_id: int,
    prev_center: np.ndarray,
    prev_angle: float,
    dt: float,
) -> ToolState3D:
    center = data.site_xpos[site_id].astype(np.float32).copy()
    mat = data.site_xmat[site_id].reshape(3, 3)
    long_axis = mat @ np.array([1.0, 0.0, 0.0])
    angle = float(np.arctan2(-long_axis[2], long_axis[0]))
    dangle = angle - prev_angle
    while dangle > np.pi:
        dangle -= 2.0 * np.pi
    while dangle < -np.pi:
        dangle += 2.0 * np.pi
    return ToolState3D(
        center=center,
        velocity=((center - prev_center) / max(dt, 1.0e-6)).astype(np.float32),
        angle=angle,
        angular_velocity=float(dangle / max(dt, 1.0e-6)),
        half=np.array([0.082, 0.122, 0.015], dtype=np.float32),
    )


def update_sand_particles(model: mujoco.MjModel, pos: np.ndarray, sample_ids: np.ndarray, geom_ids: np.ndarray) -> None:
    sampled = pos[sample_ids]
    z_min = float(sampled[:, 2].min())
    z_max = float(sampled[:, 2].max())
    denom = max(1.0e-6, z_max - z_min)
    for slot, geom_id in enumerate(geom_ids):
        p = sampled[slot]
        model.geom_pos[geom_id] = p
        u = np.clip((float(p[2]) - z_min) / denom, 0.0, 1.0)
        model.geom_rgba[geom_id] = np.array(
            [0.56 + 0.27 * u, 0.40 + 0.25 * u, 0.20 + 0.15 * u, 0.94],
            dtype=np.float32,
        )


def draw_overlay(frame: np.ndarray, force_world: np.ndarray, qerr: float, frame_id: int) -> np.ndarray:
    out = frame.copy()
    mag = float(np.linalg.norm(force_world))
    cv2.putText(out, "MuJoCo Franka + external 3D Warp MPM sand", (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (235, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(out, f"frame {frame_id:03d}   sand reaction {mag:5.1f} N   joint tracking err {qerr:5.3f}", (26, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (242, 226, 190), 2, cv2.LINE_AA)
    start = np.array([1110, 118], dtype=np.int32)
    end = start + np.array(
        [
            int(np.clip(force_world[0] * 1.6, -130, 130)),
            int(np.clip(-force_world[2] * 1.6, -100, 100)),
        ],
        dtype=np.int32,
    )
    cv2.arrowedLine(out, tuple(start), tuple(end), (70, 178, 255), 4, tipLength=0.22, line_type=cv2.LINE_AA)
    cv2.circle(out, tuple(start), 5, (70, 178, 255), -1, lineType=cv2.LINE_AA)
    return out


def write_video(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        raise RuntimeError("No frames to write")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_sheet(path: Path, frames: list[np.ndarray]) -> None:
    ids = np.linspace(0, len(frames) - 1, 5).astype(int)
    thumbs = []
    for fid in ids:
        thumb = cv2.resize(cv2.cvtColor(frames[fid], cv2.COLOR_RGB2BGR), (384, 216), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"frame {fid:03d}", (16, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 245, 245), 2, cv2.LINE_AA)
        thumbs.append(thumb)
    cv2.imwrite(path.as_posix(), np.hstack(thumbs))


def run() -> None:
    make_scene_xml()
    model = mujoco.MjModel.from_xml_path(SCENE_XML.as_posix())
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.qpos[7:9] = 0.02

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mpm3d_blade_center")
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if site_id < 0 or hand_id < 0:
        raise RuntimeError("Could not find blade site or hand body")

    waypoints = solve_waypoints(model, site_id)
    data.qpos[:7] = waypoints[0][:7]
    data.qpos[7:9] = 0.02
    data.ctrl[:] = planned_ctrl(waypoints, 0.0, 1.0)
    mujoco.mj_forward(model, data)

    mpm_cfg = SandMPM3DConfig(dt=8.0e-4, seed=7)
    mpm = SandMPM3D(mpm_cfg, device="cuda:0")
    rng = np.random.default_rng(19)
    sample_ids = np.sort(rng.choice(mpm.n_particles, size=min(VIS_PARTICLES, mpm.n_particles), replace=False))
    grain_geom_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"mpm3d_grain_{i:04d}")
            for i in range(sample_ids.size)
        ],
        dtype=np.int32,
    )
    if np.any(grain_geom_ids < 0):
        raise RuntimeError("Could not find MuJoCo visual grain geoms")
    mj_dt = float(model.opt.timestep)
    mpm_substeps = max(1, int(round(mj_dt / mpm_cfg.dt)))
    frames_n = 60
    steps_per_frame = 7
    total_time = frames_n * steps_per_frame * mj_dt
    force_scale = 0.0012
    torque_scale = 0.0006
    print(
        f"MuJoCo dt={mj_dt:.4f} 3D MPM dt={mpm_cfg.dt:.4f} "
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
                mag = 85.0
            site_pos = data.site_xpos[site_id].copy()
            body_pos = data.xpos[hand_id].copy()
            torque = np.cross(site_pos - body_pos, force) + raw_wrench[3:6].astype(np.float64) * torque_scale
            data.xfrc_applied[:] = 0.0
            data.xfrc_applied[hand_id, :3] = force
            data.xfrc_applied[hand_id, 3:6] = np.clip(torque, -13.0, 13.0)
            mujoco.mj_step(model, data)
            applied_force_world = force
            prev_tool = tool

        pos = mpm.positions()
        update_sand_particles(model, pos, sample_ids, grain_geom_ids)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="mpm3d_overview")
        desired = planned_ctrl(waypoints, float(data.time), total_time)
        qerr = float(np.linalg.norm(desired[:7] - data.qpos[:7]))
        frame = renderer.render().copy()
        frame = cv2.cvtColor(draw_overlay(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), applied_force_world, qerr, frame_id), cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if frame_id % 10 == 0:
            print(
                f"frame={frame_id:03d} t={data.time:.3f} "
                f"|F|={np.linalg.norm(applied_force_world):.2f} qerr={qerr:.3f} "
                f"tool=({prev_tool.center[0]:.3f},{prev_tool.center[1]:.3f},{prev_tool.center[2]:.3f})"
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
