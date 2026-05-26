"""Bridge MuJoCo Franka motion to Newton MPM sand and render a composite video.

The coupling in this script is intentionally explicit:

1. MuJoCo controls and renders the Franka arm and the intrusion blade.
2. The MuJoCo blade site pose drives a kinematic box collider in Newton MPM.
3. Newton MPM advances the sand and returns collider impulses.
4. A scaled sand reaction force is applied back to the MuJoCo hand body.
5. Newton particles are rendered through the MuJoCo camera and depth-composited.

This gives us a reproducible first Newton-first MuJoCo bridge without hiding the
granular state behind MuJoCo visual spheres.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm.composite_render import composite_sand, render_sand_layer
from scripts.run_mujoco_3d_mpm_cosim import (
    MENAGERIE_PANDA,
    PANDA_XML,
    joint_ranges,
    smoothstep,
    write_sheet,
    write_video,
)


OUT_DIR = ROOT / "outputs" / "mujoco_newton_mpm_bridge"
SCENE_XML = OUT_DIR / "franka_newton_mpm_scene.xml"
VIDEO_PATH = OUT_DIR / "mujoco_franka_newton_mpm_bridge.mp4"
SAND_VIDEO_PATH = OUT_DIR / "newton_mpm_sand_camera_layer.mp4"
PREVIEW_PATH = OUT_DIR / "mujoco_franka_newton_mpm_bridge_preview.png"
SHEET_PATH = OUT_DIR / "mujoco_franka_newton_mpm_bridge_sheet.png"
LOG_PATH = OUT_DIR / "newton_mpm_bridge_log.npz"


@wp.kernel
def sum_collider_impulses(
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    target_collider: int,
    impulse_sum: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if collider_ids[i] == target_collider:
        wp.atomic_add(impulse_sum, 0, collider_impulses[i])


def quat_xyzw_from_matrix(mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to Warp/MuJoCo xyzw quaternion order."""
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif axis == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1.0e-12
    return quat.astype(np.float32)


def angular_velocity_from_mats(prev: np.ndarray, curr: np.ndarray, dt: float) -> np.ndarray:
    rel = curr @ prev.T
    skew = 0.5 * np.array(
        [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]],
        dtype=np.float64,
    )
    return (skew / max(dt, 1.0e-6)).astype(np.float32)


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
                      <geom name="newton_tool_handle" type="capsule" fromto="0 0 0.035 0 0 0.145"
                        size="0.007" rgba="0.08 0.09 0.10 1" contype="0" conaffinity="0"/>
                      <geom name="newton_blade" type="box" pos="0 0 0.168"
                        size="0.082 0.122 0.015" rgba="0.015 0.018 0.022 1"
                        contype="0" conaffinity="0"/>
                      <site name="newton_blade_center" pos="0 0 0.168" size="0.012"
                        rgba="1.0 0.58 0.16 1"/>
"""
    text = text.replace(
        '                      <body name="left_finger" pos="0 0 0.0584">',
        tool + '                      <body name="left_finger" pos="0 0 0.0584">',
        1,
    )

    world = """
    <light name="newton_key" pos="0.55 -0.65 1.55" dir="-0.35 0.20 -1" directional="true"/>
    <camera name="newton_bridge_cam" pos="1.12 -0.90 0.72" xyaxes="0.63 0.78 0 -0.31 0.25 0.92"/>
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
        'gainprm="4500" biasprm="0 -4500 -450"': 'gainprm="1650" biasprm="0 -1650 -150"',
        'gainprm="3500" biasprm="0 -3500 -350"': 'gainprm="1400" biasprm="0 -1400 -120"',
        'gainprm="2000" biasprm="0 -2000 -200"': 'gainprm="800" biasprm="0 -800 -75"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    SCENE_XML.write_text(text, encoding="utf-8")


def solve_bridge_waypoints(model: mujoco.MjModel, site_id: int) -> list[np.ndarray]:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    q = data.qpos.copy()
    q[7:9] = 0.02
    lo, hi = joint_ranges(model)
    targets = [
        (np.array([0.30, 0.280, 0.365]), -0.76),
        (np.array([0.405, 0.280, 0.188]), -0.43),
        (np.array([0.735, 0.315, 0.184]), -0.32),
        (np.array([0.780, 0.280, 0.355]), -0.32),
    ]
    waypoints: list[np.ndarray] = []
    for pos_target, angle_target in targets:
        desired_axis = np.array([np.cos(angle_target), 0.0, -np.sin(angle_target)])
        for _ in range(360):
            data.qpos[:] = q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            pos_err = pos_target - data.site_xpos[site_id]
            mat = data.site_xmat[site_id].reshape(3, 3)
            long_axis = mat @ np.array([1.0, 0.0, 0.0])
            ori_err = np.cross(long_axis, desired_axis)
            err = np.concatenate([pos_err, 0.16 * ori_err])
            if np.linalg.norm(pos_err) < 0.007 and np.linalg.norm(ori_err) < 0.11:
                break
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            jac = np.vstack([jacp[:, :7], 0.16 * jacr[:, :7]])
            damping = 0.048
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
            q[:7] += np.clip(dq, -0.040, 0.040)
            q[:7] = np.clip(q[:7], lo[:7], hi[:7])
            q[7:9] = 0.02
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        waypoints.append(q.copy())
        print(
            f"bridge waypoint target={pos_target} solved={data.site_xpos[site_id]} "
            f"err={np.linalg.norm(pos_target - data.site_xpos[site_id]):.4f}"
        )
    return waypoints


def planned_bridge_ctrl(waypoints: list[np.ndarray], sim_time: float, total_time: float) -> np.ndarray:
    breaks = [0.0, 0.14, 0.35, 0.81, 1.0]
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


class NewtonSandBridge:
    def __init__(self, device: str = "cuda:0", voxel_size: float = 0.045):
        self.device = device
        self.voxel_size = voxel_size
        self.tool_half = np.asarray([0.082, 0.122, 0.015], dtype=np.float32)

        collider_builder = newton.ModelBuilder()
        self.tool_body = collider_builder.add_body(
            xform=wp.transform(wp.vec3(0.30, 0.28, 0.24), wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
            label="mujoco_blade",
        )
        collider_builder.add_shape_box(
            self.tool_body,
            hx=float(self.tool_half[0]),
            hy=float(self.tool_half[1]),
            hz=float(self.tool_half[2]),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.82),
        )
        self._add_static_box(collider_builder, (0.52, 0.465, 0.046), (0.39, 0.010, 0.046), mu=0.65)
        self._add_static_box(collider_builder, (0.52, 0.095, 0.046), (0.39, 0.010, 0.046), mu=0.65)
        self._add_static_box(collider_builder, (0.145, 0.280, 0.046), (0.010, 0.195, 0.046), mu=0.65)
        self._add_static_box(collider_builder, (0.895, 0.280, 0.046), (0.010, 0.195, 0.046), mu=0.65)
        collider_builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.8))
        self.collider_model = collider_builder.finalize(device=device)

        sand_builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand_builder)
        self._emit_sand_particles(sand_builder)
        self.sand_model = sand_builder.finalize(device=device)
        self._set_material("young_modulus", 1.0e6)
        self._set_material("poisson_ratio", 0.28)
        self._set_material("friction", 0.74)
        self._set_material("yield_pressure", 1.0e6)
        self._set_material("tensile_yield_ratio", 0.0)
        self._set_material("damping", 0.02)

        opts = SolverImplicitMPM.Config()
        opts.voxel_size = voxel_size
        opts.grid_type = "fixed"
        opts.grid_padding = 32
        opts.max_active_cell_count = 1 << 16
        opts.strain_basis = "P0"
        opts.transfer_scheme = "apic"
        opts.integration_scheme = "pic"
        opts.max_iterations = 45
        opts.tolerance = 2.0e-4
        opts.critical_fraction = 0.0
        self.solver = SolverImplicitMPM(self.sand_model, opts)
        self.solver.setup_collider(model=self.collider_model)

        self.state = self.sand_model.state()
        rigid_state = self.collider_model.state()
        self.state.body_q = wp.empty_like(rigid_state.body_q)
        self.state.body_qd = wp.empty_like(rigid_state.body_qd)
        self.state.body_f = wp.empty_like(rigid_state.body_f)
        self.state.body_q.assign(rigid_state.body_q)
        self.state.body_qd.assign(rigid_state.body_qd)

        self.impulse_sum = wp.zeros(1, dtype=wp.vec3, device=device)
        self.last_pos: np.ndarray | None = None
        self.last_mat: np.ndarray | None = None

    @staticmethod
    def _add_static_box(builder: newton.ModelBuilder, pos: tuple[float, float, float], half: tuple[float, float, float], mu: float) -> None:
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*pos), wp.quat_identity()),
            hx=half[0],
            hy=half[1],
            hz=half[2],
            cfg=newton.ModelBuilder.ShapeConfig(mu=mu),
        )

    def _emit_sand_particles(self, builder: newton.ModelBuilder) -> None:
        ppc = 3.0
        density = 1550.0
        lo = np.asarray([0.205, 0.125, 0.035], dtype=np.float64)
        hi = np.asarray([0.835, 0.435, 0.205], dtype=np.float64)
        res = np.asarray(np.ceil(ppc * (hi - lo) / self.voxel_size), dtype=int)
        cell = (hi - lo) / res
        mass = float(np.prod(cell) * density)
        radius = float(np.max(cell) * 0.50)
        builder.add_particle_grid(
            pos=wp.vec3(lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(res[0]) + 1,
            dim_y=int(res[1]) + 1,
            dim_z=int(res[2]) + 1,
            cell_x=float(cell[0]),
            cell_y=float(cell[1]),
            cell_z=float(cell[2]),
            mass=mass,
            jitter=1.8 * radius,
            radius_mean=radius,
            custom_attributes={"mpm:friction": 0.74},
        )

    def _set_material(self, name: str, value: float) -> None:
        if hasattr(self.sand_model.mpm, name):
            getattr(self.sand_model.mpm, name).fill_(value)

    @property
    def particle_count(self) -> int:
        return int(self.sand_model.particle_count)

    def set_tool_pose(self, pos: np.ndarray, mat: np.ndarray, dt: float) -> None:
        pos = np.asarray(pos, dtype=np.float32)
        mat = np.asarray(mat, dtype=np.float32).reshape(3, 3)
        quat = quat_xyzw_from_matrix(mat)
        if self.last_pos is None:
            linear = np.zeros(3, dtype=np.float32)
            angular = np.zeros(3, dtype=np.float32)
        else:
            linear = ((pos - self.last_pos) / max(dt, 1.0e-6)).astype(np.float32)
            angular = angular_velocity_from_mats(self.last_mat, mat, dt)

        body_q = wp.array(
            [wp.transform(wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), wp.quatf(*quat))],
            dtype=wp.transform,
            device=self.device,
        )
        body_qd = wp.array(
            [
                wp.spatial_vector(
                    float(linear[0]),
                    float(linear[1]),
                    float(linear[2]),
                    float(angular[0]),
                    float(angular[1]),
                    float(angular[2]),
                )
            ],
            dtype=wp.spatial_vector,
            device=self.device,
        )
        self.state.body_q.assign(body_q)
        self.state.body_qd.assign(body_qd)
        self.last_pos = pos.copy()
        self.last_mat = mat.copy()

    def step(self, dt: float) -> np.ndarray:
        self.solver.step(self.state, self.state, None, None, dt)
        impulses, _positions, collider_ids = self.solver.collect_collider_impulses(self.state)
        self.impulse_sum.fill_(wp.vec3(0.0, 0.0, 0.0))
        wp.launch(
            sum_collider_impulses,
            dim=collider_ids.shape[0],
            inputs=[collider_ids, impulses, int(self.tool_body), self.impulse_sum],
            device=self.device,
        )
        wp.synchronize()
        impulse = self.impulse_sum.numpy()[0].astype(np.float64)
        return impulse / max(dt, 1.0e-6)

    def positions(self) -> np.ndarray:
        return self.state.particle_q.numpy()

    def velocities(self) -> np.ndarray:
        return self.state.particle_qd.numpy()


def render_rgb_depth(renderer: mujoco.Renderer, data: mujoco.MjData, camera: str) -> tuple[np.ndarray, np.ndarray]:
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return rgb, depth


def draw_bridge_overlay(frame_bgr: np.ndarray, force_world: np.ndarray, qerr: float, frame_id: int) -> np.ndarray:
    out = frame_bgr.copy()
    mag = float(np.linalg.norm(force_world))
    cv2.putText(
        out,
        "MuJoCo Franka driving Newton MPM sand",
        (26, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        (235, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"frame {frame_id:03d}   Newton reaction {mag:6.1f} N   joint err {qerr:5.3f}",
        (26, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (242, 226, 190),
        2,
        cv2.LINE_AA,
    )
    start = np.array([1110, 118], dtype=np.int32)
    end = start + np.array(
        [
            int(np.clip(force_world[0] * 1.5, -130, 130)),
            int(np.clip(-force_world[2] * 1.5, -100, 100)),
        ],
        dtype=np.int32,
    )
    cv2.arrowedLine(out, tuple(start), tuple(end), (70, 178, 255), 4, tipLength=0.22, line_type=cv2.LINE_AA)
    cv2.circle(out, tuple(start), 5, (70, 178, 255), -1, lineType=cv2.LINE_AA)
    return out


def run() -> None:
    make_scene_xml()
    model = mujoco.MjModel.from_xml_path(SCENE_XML.as_posix())
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.qpos[7:9] = 0.02

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "newton_blade_center")
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "newton_bridge_cam")
    if min(site_id, hand_id, camera_id) < 0:
        raise RuntimeError("Missing MuJoCo site/body/camera for Newton bridge")

    waypoints = solve_bridge_waypoints(model, site_id)
    data.qpos[:7] = waypoints[0][:7]
    data.qpos[7:9] = 0.02
    data.ctrl[:] = planned_bridge_ctrl(waypoints, 0.0, 1.0)
    mujoco.mj_forward(model, data)

    bridge = NewtonSandBridge(device="cuda:0", voxel_size=0.045)
    mj_dt = float(model.opt.timestep)
    frames_n = 72
    steps_per_frame = 5
    total_time = frames_n * steps_per_frame * mj_dt
    force_feedback_scale = 0.018
    print(
        f"MuJoCo dt={mj_dt:.4f} Newton particles={bridge.particle_count} "
        f"frames={frames_n} steps/frame={steps_per_frame}"
    )

    renderer = mujoco.Renderer(model, height=720, width=1280)
    frames: list[np.ndarray] = []
    sand_frames: list[np.ndarray] = []
    log_frames: list[int] = []
    log_force: list[np.ndarray] = []
    log_tool: list[np.ndarray] = []
    log_particle_pos: list[np.ndarray] = []

    force_for_mujoco = np.zeros(3, dtype=np.float64)
    displayed_force = np.zeros(3, dtype=np.float64)
    for frame_id in range(frames_n):
        for _ in range(steps_per_frame):
            sim_time = float(data.time)
            data.ctrl[:] = planned_bridge_ctrl(waypoints, sim_time, total_time)
            mujoco.mj_forward(model, data)

            tool_pos = data.site_xpos[site_id].copy()
            tool_mat = data.site_xmat[site_id].reshape(3, 3).copy()
            bridge.set_tool_pose(tool_pos, tool_mat, mj_dt)
            newton_force = bridge.step(mj_dt)

            force_for_mujoco = np.clip(newton_force * force_feedback_scale, -80.0, 80.0)
            site_pos = data.site_xpos[site_id].copy()
            body_pos = data.xpos[hand_id].copy()
            torque = np.cross(site_pos - body_pos, force_for_mujoco)
            data.xfrc_applied[:] = 0.0
            data.xfrc_applied[hand_id, :3] = force_for_mujoco
            data.xfrc_applied[hand_id, 3:6] = np.clip(torque, -12.0, 12.0)
            mujoco.mj_step(model, data)
            displayed_force = force_for_mujoco.copy()

        mujoco.mj_forward(model, data)
        particle_pos = bridge.positions()
        robot_rgb, robot_depth = render_rgb_depth(renderer, data, "newton_bridge_cam")
        sand_rgb, sand_alpha, sand_depth = render_sand_layer(
            model,
            data,
            camera_id,
            particle_pos,
            1280,
            720,
            radius=5,
        )
        composite = composite_sand(robot_rgb, robot_depth, sand_rgb, sand_alpha, sand_depth)
        desired = planned_bridge_ctrl(waypoints, float(data.time), total_time)
        qerr = float(np.linalg.norm(desired[:7] - data.qpos[:7]))
        frame = cv2.cvtColor(
            draw_bridge_overlay(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR), displayed_force, qerr, frame_id),
            cv2.COLOR_BGR2RGB,
        )
        sand_only = cv2.cvtColor(
            draw_bridge_overlay(cv2.cvtColor(sand_rgb, cv2.COLOR_RGB2BGR), displayed_force, qerr, frame_id),
            cv2.COLOR_BGR2RGB,
        )
        frames.append(frame)
        sand_frames.append(sand_only)

        if frame_id % 6 == 0 or frame_id == frames_n - 1:
            log_frames.append(frame_id)
            log_force.append(displayed_force.copy())
            log_tool.append(data.site_xpos[site_id].copy())
            log_particle_pos.append(particle_pos.astype(np.float32).copy())

        if frame_id % 12 == 0:
            print(
                f"frame={frame_id:03d} t={data.time:.3f} "
                f"|F|={np.linalg.norm(displayed_force):.2f} qerr={qerr:.3f} "
                f"tool=({data.site_xpos[site_id,0]:.3f},{data.site_xpos[site_id,1]:.3f},{data.site_xpos[site_id,2]:.3f})"
            )

    renderer.close()
    write_video(VIDEO_PATH, frames)
    write_video(SAND_VIDEO_PATH, sand_frames)
    cv2.imwrite(PREVIEW_PATH.as_posix(), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    write_sheet(SHEET_PATH, frames)
    np.savez_compressed(
        LOG_PATH,
        frame=np.asarray(log_frames, dtype=np.int32),
        force=np.asarray(log_force, dtype=np.float32),
        tool_pos=np.asarray(log_tool, dtype=np.float32),
        particle_pos=np.asarray(log_particle_pos, dtype=object),
        voxel_size=np.asarray(bridge.voxel_size, dtype=np.float32),
    )
    print(f"scene={SCENE_XML}")
    print(f"video={VIDEO_PATH}")
    print(f"sand_video={SAND_VIDEO_PATH}")
    print(f"preview={PREVIEW_PATH}")
    print(f"sheet={SHEET_PATH}")
    print(f"log={LOG_PATH}")


if __name__ == "__main__":
    run()
