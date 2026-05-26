from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import warp as wp

from warp_sand_mpm_coupled import (
    NGRID,
    ToolState,
    create_particles,
    draw_frame,
    g2p_kernel,
    grid_update_kernel,
    p2g_kernel,
)


ROOT = Path(__file__).resolve().parent
MENAGERIE_PANDA = ROOT / "mujoco_menagerie" / "franka_emika_panda"
PANDA_XML = MENAGERIE_PANDA / "panda.xml"
OUT_DIR = ROOT / "outputs" / "mujoco_mpm_cosim"
SCENE_XML = OUT_DIR / "franka_mpm_cosim_scene.xml"
MUJOCO_VIDEO = OUT_DIR / "mujoco_franka_mpm_coupled.mp4"
MPM_VIDEO = OUT_DIR / "mpm_external_sand_coupled.mp4"
MUJOCO_PREVIEW = OUT_DIR / "mujoco_franka_mpm_preview.png"
MPM_PREVIEW = OUT_DIR / "mpm_external_sand_preview.png"
SHEET_PATH = OUT_DIR / "mujoco_mpm_cosim_contact_sheet.png"


@dataclass
class MpmState:
    x: wp.array
    v: wp.array
    C: wp.array
    F: wp.array
    Jp: wp.array
    grid_vx: wp.array
    grid_vy: wp.array
    grid_m: wp.array
    force_accum: wp.array
    n_particles: int


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def sand_proxy_geoms(ncols: int = 48) -> str:
    x_min, x_max = 0.16, 0.86
    dx = (x_max - x_min) / ncols
    lines = []
    for i in range(ncols):
        x = x_min + (i + 0.5) * dx
        lines.append(
            f'    <geom name="mpm_sand_col_{i:02d}" type="box" '
            f'pos="{x:.5f} 0 0.08000" size="{0.5 * dx:.5f} 0.13200 0.04000" '
            'rgba="0.72 0.56 0.31 0.86" contype="0" conaffinity="0"/>'
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
        '    <headlight diffuse="0.55 0.55 0.55" ambient="0.30 0.30 0.30" specular="0.1 0.1 0.1"/>\n'
        '    <global offwidth="960" offheight="720" azimuth="125" elevation="-24"/>\n'
        '  </visual>\n\n'
        "  <asset>",
        1,
    )

    # Keep the tool non-colliding inside MuJoCo; the MPM engine owns sand contact.
    tool = """
                      <geom name="mpm_shovel_handle" type="capsule" fromto="0 0 0.045 0 0 0.145"
                        size="0.007" rgba="0.08 0.09 0.10 1" contype="0" conaffinity="0"/>
                      <geom name="mpm_shovel_blade" type="box" pos="0 0 0.168"
                        size="0.078 0.014 0.018" rgba="0.02 0.025 0.028 1"
                        contype="0" conaffinity="0"/>
                      <site name="mpm_shovel_center" pos="0 0 0.168" size="0.010"
                        rgba="1.0 0.56 0.16 1"/>
"""
    text = text.replace(
        '                      <body name="left_finger" pos="0 0 0.0584">',
        tool + '                      <body name="left_finger" pos="0 0 0.0584">',
        1,
    )

    world = f"""
    <light name="mpm_key" pos="0.65 -0.8 1.6" dir="-0.4 0.35 -1" directional="true"/>
    <camera name="cosim_overview" pos="1.12 -1.05 0.72" xyaxes="0.68 0.73 0 -0.30 0.28 0.91"/>
    <geom name="floor" type="plane" size="1.25 1.25 0.04" rgba="0.22 0.25 0.27 1"
      condim="3" friction="1.2 0.03 0.0002"/>
    <geom name="tray_back" type="box" pos="0.52 0.145 0.045" size="0.37 0.010 0.045"
      rgba="0.17 0.18 0.19 1" contype="0" conaffinity="0"/>
    <geom name="tray_front" type="box" pos="0.52 -0.145 0.045" size="0.37 0.010 0.045"
      rgba="0.17 0.18 0.19 1" contype="0" conaffinity="0"/>
    <geom name="tray_left" type="box" pos="0.145 0 0.045" size="0.010 0.155 0.045"
      rgba="0.17 0.18 0.19 1" contype="0" conaffinity="0"/>
    <geom name="tray_right" type="box" pos="0.895 0 0.045" size="0.010 0.155 0.045"
      rgba="0.17 0.18 0.19 1" contype="0" conaffinity="0"/>
{sand_proxy_geoms()}
"""
    text = text.replace("</worldbody>", world + "  </worldbody>", 1)

    # Soften the position servos enough that sand reaction produces visible tracking error.
    replacements = {
        'gainprm="4500" biasprm="0 -4500 -450"': 'gainprm="1800" biasprm="0 -1800 -160"',
        'gainprm="3500" biasprm="0 -3500 -350"': 'gainprm="1500" biasprm="0 -1500 -130"',
        'gainprm="2000" biasprm="0 -2000 -200"': 'gainprm="850" biasprm="0 -850 -80"',
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
        (np.array([0.30, 0.0, 0.345]), -0.82),
        (np.array([0.38, 0.0, 0.225]), -0.46),
        (np.array([0.68, 0.0, 0.222]), -0.36),
        (np.array([0.74, 0.0, 0.335]), -0.34),
    ]
    waypoints: list[np.ndarray] = []
    for pos_target, angle_target in targets:
        desired_axis = np.array([np.cos(angle_target), 0.0, np.sin(angle_target)])
        for _ in range(260):
            data.qpos[:] = q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            pos_err = pos_target - data.site_xpos[site_id]
            mat = data.site_xmat[site_id].reshape(3, 3)
            long_axis = mat @ np.array([1.0, 0.0, 0.0])
            ori_err = np.cross(long_axis, desired_axis)
            err = np.concatenate([pos_err, 0.18 * ori_err])
            if np.linalg.norm(pos_err) < 0.004 and np.linalg.norm(ori_err) < 0.08:
                break
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            jac = np.vstack([jacp[:, :7], 0.18 * jacr[:, :7]])
            damping = 0.035
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
            q[:7] += np.clip(dq, -0.045, 0.045)
            q[:7] = np.clip(q[:7], lo[:7], hi[:7])
            q[7:9] = 0.02
        waypoints.append(q.copy())
        print(
            f"waypoint target={pos_target} solved_pos={data.site_xpos[site_id]} "
            f"pos_err={np.linalg.norm(pos_target - data.site_xpos[site_id]):.4f}"
        )
    return waypoints


def planned_ctrl(waypoints: list[np.ndarray], sim_time: float, total_time: float) -> np.ndarray:
    breaks = [0.0, 0.16, 0.37, 0.78, 1.0]
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


def init_mpm() -> tuple[MpmState, dict[str, float]]:
    wp.init()
    wp.set_device("cuda:0")
    pos_np = create_particles()
    n_particles = int(pos_np.shape[0])
    state = MpmState(
        x=wp.array(pos_np, dtype=wp.vec2, device="cuda:0"),
        v=wp.zeros(n_particles, dtype=wp.vec2, device="cuda:0"),
        C=wp.zeros(n_particles, dtype=wp.mat22, device="cuda:0"),
        F=wp.array(np.tile(np.eye(2, dtype=np.float32), (n_particles, 1, 1)), dtype=wp.mat22, device="cuda:0"),
        Jp=wp.ones(n_particles, dtype=float, device="cuda:0"),
        grid_vx=wp.zeros(NGRID, dtype=float, device="cuda:0"),
        grid_vy=wp.zeros(NGRID, dtype=float, device="cuda:0"),
        grid_m=wp.zeros(NGRID, dtype=float, device="cuda:0"),
        force_accum=wp.zeros(3, dtype=float, device="cuda:0"),
        n_particles=n_particles,
    )
    young = 2.4e3
    nu = 0.24
    params = {
        "dx": 1.0 / 160.0,
        "dt": 4.0e-4,
        "p_mass": 1.0,
        "p_vol": 1.0,
        "mu0": young / (2.0 * (1.0 + nu)),
        "lam0": young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
    }
    return state, params


def mpm_step(state: MpmState, params: dict[str, float], tool: ToolState, substeps: int) -> np.ndarray:
    dx = params["dx"]
    inv_dx = 1.0 / dx
    dt = params["dt"]
    total_force = np.zeros(3, dtype=np.float32)
    for _ in range(substeps):
        state.grid_vx.zero_()
        state.grid_vy.zero_()
        state.grid_m.zero_()
        state.force_accum.zero_()
        wp.launch(
            p2g_kernel,
            dim=state.n_particles,
            inputs=[
                state.x,
                state.v,
                state.C,
                state.F,
                state.Jp,
                state.grid_vx,
                state.grid_vy,
                state.grid_m,
                dt,
                dx,
                inv_dx,
                params["p_mass"],
                params["p_vol"],
                params["mu0"],
                params["lam0"],
                0.46,
            ],
            device="cuda:0",
        )
        wp.launch(
            grid_update_kernel,
            dim=NGRID,
            inputs=[
                state.grid_vx,
                state.grid_vy,
                state.grid_m,
                dt,
                dx,
                -1.2,
                wp.vec2(float(tool.center[0]), float(tool.center[1])),
                wp.vec2(float(tool.velocity[0]), float(tool.velocity[1])),
                float(tool.angular_velocity),
                float(np.cos(tool.angle)),
                float(np.sin(tool.angle)),
                wp.vec2(float(tool.half[0]), float(tool.half[1])),
                0.34,
                1.15,
                state.force_accum,
            ],
            device="cuda:0",
        )
        wp.launch(
            g2p_kernel,
            dim=state.n_particles,
            inputs=[state.x, state.v, state.C, state.grid_vx, state.grid_vy, state.grid_m, dt, dx, inv_dx],
            device="cuda:0",
        )
        total_force += state.force_accum.numpy().astype(np.float32)
    return total_force / max(1, substeps)


def tool_state_from_mujoco(
    data: mujoco.MjData,
    site_id: int,
    prev_center: np.ndarray,
    prev_angle: float,
    dt: float,
) -> ToolState:
    pos = data.site_xpos[site_id].copy()
    center = np.array([pos[0], pos[2]], dtype=np.float32)
    mat = data.site_xmat[site_id].reshape(3, 3)
    long_axis = mat @ np.array([1.0, 0.0, 0.0])
    angle = float(np.arctan2(long_axis[2], long_axis[0]))
    dangle = angle - prev_angle
    while dangle > np.pi:
        dangle -= 2.0 * np.pi
    while dangle < -np.pi:
        dangle += 2.0 * np.pi
    return ToolState(
        center=center,
        velocity=((center - prev_center) / max(dt, 1.0e-6)).astype(np.float32),
        angle=angle,
        angular_velocity=float(dangle / max(dt, 1.0e-6)),
        half=np.array([0.078, 0.014], dtype=np.float32),
    )


def update_mujoco_sand_proxy(model: mujoco.MjModel, pos: np.ndarray) -> None:
    x_min, x_max = 0.16, 0.86
    base = 0.038
    ncols = 48
    top = np.full(ncols, base + 0.006, dtype=np.float32)
    valid = (pos[:, 0] >= x_min) & (pos[:, 0] <= x_max)
    ids = np.clip(((pos[valid, 0] - x_min) / (x_max - x_min) * (ncols - 1)).astype(np.int32), 0, ncols - 1)
    np.maximum.at(top, ids, pos[valid, 1])
    for i, y in enumerate(top):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"mpm_sand_col_{i:02d}")
        if geom_id < 0:
            continue
        half_h = max(0.002, 0.5 * (float(y) - base))
        model.geom_size[geom_id, 2] = half_h
        model.geom_pos[geom_id, 2] = base + half_h


def draw_force_overlay(frame: np.ndarray, reaction: np.ndarray, err: float) -> np.ndarray:
    out = frame.copy()
    mag = float(np.linalg.norm(reaction))
    cv2.putText(out, "MuJoCo Franka driven by external Warp MPM sand force", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(out, f"sand reaction {mag:5.1f} N   tool tracking err {err:5.3f} m", (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (242, 224, 190), 2, cv2.LINE_AA)
    start = np.array([790, 112], dtype=np.int32)
    end = start + np.array([int(np.clip(reaction[0] * 2.2, -120, 120)), int(np.clip(-reaction[2] * 2.2, -90, 90))], dtype=np.int32)
    cv2.arrowedLine(out, tuple(start), tuple(end), (60, 176, 255), 4, tipLength=0.2, line_type=cv2.LINE_AA)
    cv2.circle(out, tuple(start), 5, (60, 176, 255), -1, lineType=cv2.LINE_AA)
    return out


def write_video(path: Path, frames: list[np.ndarray], size: tuple[int, int], rgb: bool) -> None:
    writer = cv2.VideoWriter(path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if rgb else frame)
    writer.release()


def make_contact_sheet(mj_frames: list[np.ndarray], mpm_frames: list[np.ndarray]) -> None:
    ids = [0, len(mj_frames) // 4, len(mj_frames) // 2, 3 * len(mj_frames) // 4, len(mj_frames) - 1]
    pairs = []
    for fid in ids:
        mj = cv2.resize(cv2.cvtColor(mj_frames[fid], cv2.COLOR_RGB2BGR), (384, 288), interpolation=cv2.INTER_AREA)
        mpm = cv2.resize(mpm_frames[fid], (384, 230), interpolation=cv2.INTER_AREA)
        pad = np.full((58, 384, 3), (24, 27, 29), dtype=np.uint8)
        cv2.putText(pad, f"frame {fid:02d}", (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
        pairs.append(np.vstack([mj, mpm, pad]))
    sheet = np.hstack(pairs)
    cv2.imwrite(SHEET_PATH.as_posix(), sheet)


def run() -> None:
    make_scene_xml()
    model = mujoco.MjModel.from_xml_path(SCENE_XML.as_posix())
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.qpos[7:9] = 0.02
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mpm_shovel_center")
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if site_id < 0 or hand_id < 0:
        raise RuntimeError("Could not find shovel site or hand body")

    waypoints = solve_waypoints(model, site_id)
    data.qpos[:7] = waypoints[0][:7]
    data.qpos[7:9] = 0.02
    data.ctrl[:] = planned_ctrl(waypoints, 0.0, 1.0)
    mujoco.mj_forward(model, data)

    mpm, mpm_params = init_mpm()
    mj_dt = float(model.opt.timestep)
    mpm_substeps = max(1, int(round(mj_dt / mpm_params["dt"])))
    frames = 56
    steps_per_frame = 8
    total_time = frames * steps_per_frame * mj_dt
    print(
        f"MuJoCo dt={mj_dt:.4f} MPM dt={mpm_params['dt']:.4f} "
        f"mpm_substeps/mj_step={mpm_substeps} frames={frames}"
    )

    renderer = mujoco.Renderer(model, height=720, width=960)
    mj_frames: list[np.ndarray] = []
    mpm_frames: list[np.ndarray] = []
    force_history: list[float] = []
    prev_tool = tool_state_from_mujoco(
        data,
        site_id,
        np.array([data.site_xpos[site_id, 0], data.site_xpos[site_id, 2]], dtype=np.float32),
        0.0,
        mj_dt,
    )
    applied_force_world = np.zeros(3, dtype=np.float64)
    last_mpm_force = np.zeros(2, dtype=np.float32)
    last_err = 0.0
    force_scale = 0.016
    torque_scale = 0.004

    for frame_id in range(frames):
        for _ in range(steps_per_frame):
            sim_time = float(data.time)
            data.ctrl[:] = planned_ctrl(waypoints, sim_time, total_time)
            mujoco.mj_forward(model, data)
            tool = tool_state_from_mujoco(data, site_id, prev_tool.center, prev_tool.angle, mj_dt)
            raw_force = mpm_step(mpm, mpm_params, tool, mpm_substeps)

            mpm_force = raw_force[:2].astype(np.float64) * force_scale
            mag = float(np.linalg.norm(mpm_force))
            if mag > 85.0:
                mpm_force *= 85.0 / mag
                mag = 85.0
            applied_force_world = np.array([mpm_force[0], 0.0, mpm_force[1]], dtype=np.float64)
            site_pos = data.site_xpos[site_id].copy()
            body_pos = data.xpos[hand_id].copy()
            lever = site_pos - body_pos
            torque = np.cross(lever, applied_force_world)
            torque[1] += float(raw_force[2]) * torque_scale

            data.xfrc_applied[:] = 0.0
            data.xfrc_applied[hand_id, :3] = applied_force_world
            data.xfrc_applied[hand_id, 3:6] = np.clip(torque, -14.0, 14.0)
            mujoco.mj_step(model, data)

            prev_tool = tool
            last_mpm_force = mpm_force.astype(np.float32)

        wp.synchronize()
        pos = mpm.x.numpy()
        update_mujoco_sand_proxy(model, pos)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="cosim_overview")
        desired = planned_ctrl(waypoints, float(data.time), total_time)
        last_err = float(np.linalg.norm(desired[:7] - data.qpos[:7]))
        mj_rgb = renderer.render().copy()
        mj_rgb = cv2.cvtColor(draw_force_overlay(cv2.cvtColor(mj_rgb, cv2.COLOR_RGB2BGR), applied_force_world, last_err), cv2.COLOR_BGR2RGB)
        mj_frames.append(mj_rgb)

        force_history.append(float(np.linalg.norm(last_mpm_force)))
        mpm_tool = tool_state_from_mujoco(data, site_id, prev_tool.center, prev_tool.angle, mj_dt)
        mpm_frame = draw_frame(pos, mpm_tool, reaction=last_mpm_force, force_history=force_history)
        cv2.putText(mpm_frame, "tool pose and force come from live MuJoCo step", (24, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (225, 228, 220), 2, cv2.LINE_AA)
        mpm_frames.append(mpm_frame)

        if frame_id % 10 == 0:
            print(
                f"frame={frame_id:03d} time={data.time:.3f} "
                f"force={np.linalg.norm(applied_force_world):.2f}N qerr={last_err:.3f} "
                f"tool=({mpm_tool.center[0]:.3f},{mpm_tool.center[1]:.3f})"
            )

    renderer.close()
    write_video(MUJOCO_VIDEO, mj_frames, (960, 720), rgb=True)
    write_video(MPM_VIDEO, mpm_frames, (960, 576), rgb=False)
    cv2.imwrite(MUJOCO_PREVIEW.as_posix(), cv2.cvtColor(mj_frames[-1], cv2.COLOR_RGB2BGR))
    cv2.imwrite(MPM_PREVIEW.as_posix(), mpm_frames[-1])
    make_contact_sheet(mj_frames, mpm_frames)
    print(f"scene={SCENE_XML}")
    print(f"mujoco_video={MUJOCO_VIDEO}")
    print(f"mpm_video={MPM_VIDEO}")
    print(f"sheet={SHEET_PATH}")


if __name__ == "__main__":
    run()
