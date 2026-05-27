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

import argparse
import json
import os
import sys
from dataclasses import dataclass
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

from granular_mpm.composite_render import (
    composite_sand,
    render_sand_density_layer,
    render_sand_heightfield_layer,
    render_sand_layer,
)
from scripts.run_mujoco_3d_mpm_cosim import (
    MENAGERIE_PANDA,
    PANDA_XML,
    joint_ranges,
    smoothstep,
    write_sheet,
    write_video,
)


OUT_DIR = ROOT / "outputs" / "mujoco_newton_mpm_bridge"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "newton_bridge_heightfield.json"


@dataclass(frozen=True)
class BridgePaths:
    out_dir: Path
    scene_xml: Path
    video: Path
    robot_video: Path
    sand_video: Path
    preview: Path
    sheet: Path
    log: Path


def bridge_paths(out_dir: Path) -> BridgePaths:
    return BridgePaths(
        out_dir=out_dir,
        scene_xml=out_dir / "franka_newton_mpm_scene.xml",
        video=out_dir / "mujoco_franka_newton_mpm_bridge.mp4",
        robot_video=out_dir / "mujoco_robot_pass.mp4",
        sand_video=out_dir / "newton_mpm_sand_camera_layer.mp4",
        preview=out_dir / "mujoco_franka_newton_mpm_bridge_preview.png",
        sheet=out_dir / "mujoco_franka_newton_mpm_bridge_sheet.png",
        log=out_dir / "newton_mpm_bridge_log.npz",
    )


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


def _bridge_defaults(config_path: Path | None) -> dict[str, float | int | str]:
    defaults: dict[str, float | int | str] = {
        "device": "cuda:0",
        "output_dir": "outputs/mujoco_newton_mpm_bridge",
        "voxel_size": 0.045,
        "particles_per_cell": 3.0,
        "sand_density": 1550.0,
        "sand_friction": 0.74,
        "sand_young_modulus": 1.0e6,
        "sand_poisson": 0.28,
        "sand_yield_pressure": 1.0e6,
        "sand_damping": 0.02,
        "sand_jitter_scale": 1.8,
        "frames": 72,
        "steps_per_frame": 5,
        "sand_render_mode": "heightfield",
        "render_radius": 5,
        "render_blur": 2.4,
        "alpha_blur": 1.6,
        "alpha_cutoff": 0.060,
        "alpha_gain": 0.48,
        "force_feedback_scale": 0.018,
    }
    if config_path is None:
        return defaults

    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    for key, value in raw.items():
        normalized = key.replace("-", "_")
        if normalized in defaults:
            defaults[normalized] = value
    return defaults


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None)
    pre_args, _ = pre_parser.parse_known_args()
    defaults = _bridge_defaults(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / str(defaults["output_dir"]))
    parser.add_argument("--voxel-size", type=float, default=defaults["voxel_size"])
    parser.add_argument("--particles-per-cell", type=float, default=defaults["particles_per_cell"])
    parser.add_argument("--sand-density", type=float, default=defaults["sand_density"])
    parser.add_argument("--sand-friction", type=float, default=defaults["sand_friction"])
    parser.add_argument("--sand-young-modulus", type=float, default=defaults["sand_young_modulus"])
    parser.add_argument("--sand-poisson", type=float, default=defaults["sand_poisson"])
    parser.add_argument("--sand-yield-pressure", type=float, default=defaults["sand_yield_pressure"])
    parser.add_argument("--sand-damping", type=float, default=defaults["sand_damping"])
    parser.add_argument("--sand-jitter-scale", type=float, default=defaults["sand_jitter_scale"])
    parser.add_argument("--frames", type=int, default=defaults["frames"])
    parser.add_argument("--steps-per-frame", type=int, default=defaults["steps_per_frame"])
    parser.add_argument(
        "--sand-render-mode",
        choices=["heightfield", "density", "point"],
        default=defaults["sand_render_mode"],
    )
    parser.add_argument("--render-radius", type=int, default=defaults["render_radius"])
    parser.add_argument("--render-blur", type=float, default=defaults["render_blur"])
    parser.add_argument("--alpha-blur", type=float, default=defaults["alpha_blur"])
    parser.add_argument("--alpha-cutoff", type=float, default=defaults["alpha_cutoff"])
    parser.add_argument("--alpha-gain", type=float, default=defaults["alpha_gain"])
    parser.add_argument("--force-feedback-scale", type=float, default=defaults["force_feedback_scale"])
    return parser.parse_args()


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


def make_scene_xml(paths: BridgePaths) -> None:
    paths.out_dir.mkdir(parents=True, exist_ok=True)
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
    paths.scene_xml.write_text(text, encoding="utf-8")


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
    def __init__(
        self,
        device: str = "cuda:0",
        voxel_size: float = 0.045,
        particles_per_cell: float = 3.0,
        sand_density: float = 1550.0,
        sand_friction: float = 0.74,
        sand_young_modulus: float = 1.0e6,
        sand_poisson: float = 0.28,
        sand_yield_pressure: float = 1.0e6,
        sand_damping: float = 0.02,
        sand_jitter_scale: float = 1.8,
    ):
        self.device = device
        self.voxel_size = voxel_size
        self.particles_per_cell = particles_per_cell
        self.sand_density = sand_density
        self.sand_friction = sand_friction
        self.sand_young_modulus = sand_young_modulus
        self.sand_poisson = sand_poisson
        self.sand_yield_pressure = sand_yield_pressure
        self.sand_damping = sand_damping
        self.sand_jitter_scale = sand_jitter_scale
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
        self._set_material("young_modulus", sand_young_modulus)
        self._set_material("poisson_ratio", sand_poisson)
        self._set_material("friction", sand_friction)
        self._set_material("yield_pressure", sand_yield_pressure)
        self._set_material("tensile_yield_ratio", 0.0)
        self._set_material("damping", sand_damping)

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
        density = self.sand_density
        lo = np.asarray([0.205, 0.125, 0.035], dtype=np.float64)
        hi = np.asarray([0.835, 0.435, 0.205], dtype=np.float64)
        res = np.asarray(np.ceil(self.particles_per_cell * (hi - lo) / self.voxel_size), dtype=int)
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
            jitter=float(self.sand_jitter_scale) * radius,
            radius_mean=radius,
            custom_attributes={"mpm:friction": self.sand_friction},
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
    args = parse_args()
    paths = bridge_paths(args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir)
    make_scene_xml(paths)
    model = mujoco.MjModel.from_xml_path(paths.scene_xml.as_posix())
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

    bridge = NewtonSandBridge(
        device=args.device,
        voxel_size=args.voxel_size,
        particles_per_cell=args.particles_per_cell,
        sand_density=args.sand_density,
        sand_friction=args.sand_friction,
        sand_young_modulus=args.sand_young_modulus,
        sand_poisson=args.sand_poisson,
        sand_yield_pressure=args.sand_yield_pressure,
        sand_damping=args.sand_damping,
        sand_jitter_scale=args.sand_jitter_scale,
    )
    mj_dt = float(model.opt.timestep)
    frames_n = args.frames
    steps_per_frame = args.steps_per_frame
    total_time = frames_n * steps_per_frame * mj_dt
    force_feedback_scale = args.force_feedback_scale
    print(
        f"MuJoCo dt={mj_dt:.4f} Newton particles={bridge.particle_count} "
        f"frames={frames_n} steps/frame={steps_per_frame}"
    )

    renderer = mujoco.Renderer(model, height=720, width=1280)
    frames: list[np.ndarray] = []
    robot_frames: list[np.ndarray] = []
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
        if args.sand_render_mode == "heightfield":
            sand_rgb, sand_alpha, sand_depth = render_sand_heightfield_layer(
                model,
                data,
                camera_id,
                particle_pos,
                1280,
                720,
                density_blur_sigma=max(0.6, args.render_blur * 0.48),
                height_blur_sigma=max(0.8, args.render_blur * 0.67),
                alpha_cutoff=args.alpha_cutoff,
                alpha_gain=args.alpha_gain,
            )
        elif args.sand_render_mode == "density":
            sand_rgb, sand_alpha, sand_depth = render_sand_density_layer(
                model,
                data,
                camera_id,
                particle_pos,
                1280,
                720,
                blur_sigma=args.render_blur,
                alpha_blur_sigma=args.alpha_blur,
                alpha_cutoff=args.alpha_cutoff,
                alpha_gain=args.alpha_gain,
            )
        else:
            sand_rgb, sand_alpha, sand_depth = render_sand_layer(
                model,
                data,
                camera_id,
                particle_pos,
                1280,
                720,
                radius=args.render_radius,
                blur_sigma=args.render_blur,
                alpha_blur_sigma=args.alpha_blur,
                alpha_cutoff=args.alpha_cutoff,
                alpha_gain=args.alpha_gain,
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
        robot_frames.append(robot_rgb)
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
    write_video(paths.video, frames)
    write_video(paths.robot_video, robot_frames)
    write_video(paths.sand_video, sand_frames)
    cv2.imwrite(paths.preview.as_posix(), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    write_sheet(paths.sheet, frames)
    np.savez_compressed(
        paths.log,
        frame=np.asarray(log_frames, dtype=np.int32),
        force=np.asarray(log_force, dtype=np.float32),
        tool_pos=np.asarray(log_tool, dtype=np.float32),
        particle_pos=np.asarray(log_particle_pos, dtype=object),
        voxel_size=np.asarray(bridge.voxel_size, dtype=np.float32),
        particles_per_cell=np.asarray(bridge.particles_per_cell, dtype=np.float32),
        sand_density=np.asarray(bridge.sand_density, dtype=np.float32),
        sand_friction=np.asarray(bridge.sand_friction, dtype=np.float32),
        sand_young_modulus=np.asarray(bridge.sand_young_modulus, dtype=np.float32),
        sand_poisson=np.asarray(bridge.sand_poisson, dtype=np.float32),
        sand_yield_pressure=np.asarray(bridge.sand_yield_pressure, dtype=np.float32),
        sand_damping=np.asarray(bridge.sand_damping, dtype=np.float32),
        sand_jitter_scale=np.asarray(bridge.sand_jitter_scale, dtype=np.float32),
        device=np.asarray(args.device),
        sand_render_mode=np.asarray(args.sand_render_mode),
        render_radius=np.asarray(args.render_radius, dtype=np.int32),
        render_blur=np.asarray(args.render_blur, dtype=np.float32),
        alpha_blur=np.asarray(args.alpha_blur, dtype=np.float32),
        force_feedback_scale=np.asarray(args.force_feedback_scale, dtype=np.float32),
    )
    print(f"scene={paths.scene_xml}")
    print(f"video={paths.video}")
    print(f"robot_video={paths.robot_video}")
    print(f"sand_video={paths.sand_video}")
    print(f"preview={paths.preview}")
    print(f"sheet={paths.sheet}")
    print(f"log={paths.log}")


if __name__ == "__main__":
    run()
