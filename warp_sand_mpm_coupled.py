from __future__ import annotations

"""Minimal 2D sand engine using Warp MLS-MPM with two-way shovel coupling.

The demo keeps only the pieces needed to prototype shovel-granular coupling:
P2G transfer, grid update, G2P transfer, a Drucker-Prager-like plastic
projection, a shovel SDF collider, and the reaction force returned to a
controlled shovel body.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import warp as wp


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "warp_sand_mpm_coupled"
VIDEO_PATH = OUT_DIR / "sand_mpm_coupled_shovel.mp4"
PREVIEW_PATH = OUT_DIR / "sand_mpm_coupled_preview.png"

NX = wp.constant(160)
NY = wp.constant(96)
NGRID = 160 * 96


@wp.func
def grid_index(i: int, j: int) -> int:
    return i + j * NX


@wp.func
def rot_vec(v: wp.vec2, c: float, s: float) -> wp.vec2:
    return wp.vec2(c * v[0] - s * v[1], s * v[0] + c * v[1])


@wp.func
def rot_vec_t(v: wp.vec2, c: float, s: float) -> wp.vec2:
    return wp.vec2(c * v[0] + s * v[1], -s * v[0] + c * v[1])


@wp.func
def box_sdf_normal(p: wp.vec2, center: wp.vec2, c: float, s: float, half: wp.vec2):
    local = rot_vec_t(p - center, c, s)
    ax = wp.abs(local[0])
    ay = wp.abs(local[1])
    qx = ax - half[0]
    qy = ay - half[1]

    ox = wp.max(qx, 0.0)
    oy = wp.max(qy, 0.0)
    outside = wp.length(wp.vec2(ox, oy))
    inside = wp.min(wp.max(qx, qy), 0.0)
    sdf = outside + inside

    n_local = wp.vec2(0.0, 1.0)
    sx = wp.where(local[0] >= 0.0, 1.0, -1.0)
    sy = wp.where(local[1] >= 0.0, 1.0, -1.0)
    if qx > 0.0 or qy > 0.0:
        g = wp.vec2(wp.where(qx > 0.0, qx * sx, 0.0), wp.where(qy > 0.0, qy * sy, 0.0))
        gl = wp.length(g)
        if gl > 1.0e-8:
            n_local = g / gl
    else:
        if qx > qy:
            n_local = wp.vec2(sx, 0.0)
        else:
            n_local = wp.vec2(0.0, sy)

    return sdf, rot_vec(n_local, c, s)


@wp.kernel
def p2g_kernel(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    C: wp.array(dtype=wp.mat22),
    F: wp.array(dtype=wp.mat22),
    Jp: wp.array(dtype=float),
    grid_vx: wp.array(dtype=float),
    grid_vy: wp.array(dtype=float),
    grid_m: wp.array(dtype=float),
    dt: float,
    dx: float,
    inv_dx: float,
    p_mass: float,
    p_vol: float,
    mu0: float,
    lam0: float,
    dp_alpha: float,
):
    p = wp.tid()
    xp = x[p]
    vp = v[p]
    Cp = C[p]

    I = wp.mat22(1.0, 0.0, 0.0, 1.0)
    F_trial = (I + Cp * dt) * F[p]

    U = wp.mat22()
    sig = wp.vec2()
    V = wp.mat22()
    wp.svd2(F_trial, U, sig, V)

    old_j = wp.max(1.0e-6, sig[0] * sig[1])

    # Drucker-Prager-inspired return mapping in log-strain space.
    e0 = wp.log(wp.max(sig[0], 1.0e-4))
    e1 = wp.log(wp.max(sig[1], 1.0e-4))
    tr = e0 + e1
    mean = 0.5 * tr
    dev0 = e0 - mean
    dev1 = e1 - mean
    dev_norm = wp.sqrt(dev0 * dev0 + dev1 * dev1)

    if tr > 0.0:
        e0 = 0.0
        e1 = 0.0
    else:
        yield_value = dev_norm + dp_alpha * tr
        if yield_value > 0.0 and dev_norm > 1.0e-8:
            scale = wp.max(0.0, -dp_alpha * tr) / dev_norm
            e0 = mean + dev0 * scale
            e1 = mean + dev1 * scale

    s0 = wp.exp(wp.clamp(e0, -0.35, 0.25))
    s1 = wp.exp(wp.clamp(e1, -0.35, 0.25))
    new_j = wp.max(1.0e-6, s0 * s1)
    Jp[p] = wp.clamp(Jp[p] * old_j / new_j, 0.35, 2.5)
    Fp = U * wp.mat22(s0, 0.0, 0.0, s1) * wp.transpose(V)
    F[p] = Fp

    mu = mu0
    lam = lam0
    J = wp.determinant(Fp)
    R = U * wp.transpose(V)
    stress = (Fp - R) * (2.0 * mu) * wp.transpose(Fp) + I * (lam * J * (J - 1.0))
    stress = stress * (-dt * p_vol * 4.0 * inv_dx * inv_dx)
    affine = stress + Cp * p_mass

    grid_pos = xp * inv_dx
    base_x = int(wp.floor(grid_pos[0] - 0.5))
    base_y = int(wp.floor(grid_pos[1] - 0.5))
    fx = grid_pos - wp.vec2(float(base_x), float(base_y))

    wx = wp.vec3(
        0.5 * (1.5 - fx[0]) * (1.5 - fx[0]),
        0.75 - (fx[0] - 1.0) * (fx[0] - 1.0),
        0.5 * (fx[0] - 0.5) * (fx[0] - 0.5),
    )
    wy = wp.vec3(
        0.5 * (1.5 - fx[1]) * (1.5 - fx[1]),
        0.75 - (fx[1] - 1.0) * (fx[1] - 1.0),
        0.5 * (fx[1] - 0.5) * (fx[1] - 0.5),
    )

    for gx in range(3):
        for gy in range(3):
            i = base_x + gx
            j = base_y + gy
            if i >= 0 and i < NX and j >= 0 and j < NY:
                weight = wx[gx] * wy[gy]
                dpos = (wp.vec2(float(gx), float(gy)) - fx) * dx
                momentum = vp * p_mass + affine * dpos
                idx = grid_index(i, j)
                wp.atomic_add(grid_vx, idx, weight * momentum[0])
                wp.atomic_add(grid_vy, idx, weight * momentum[1])
                wp.atomic_add(grid_m, idx, weight * p_mass)


@wp.kernel
def grid_update_kernel(
    grid_vx: wp.array(dtype=float),
    grid_vy: wp.array(dtype=float),
    grid_m: wp.array(dtype=float),
    dt: float,
    dx: float,
    gravity: float,
    tool_center: wp.vec2,
    tool_vel: wp.vec2,
    tool_omega: float,
    tool_c: float,
    tool_s: float,
    tool_half: wp.vec2,
    tool_mu: float,
    max_speed: float,
    force_accum: wp.array(dtype=float),
):
    tid = wp.tid()
    i = tid % NX
    j = tid / NX
    m = grid_m[tid]

    if m <= 0.0:
        return

    vel = wp.vec2(grid_vx[tid] / m, grid_vy[tid] / m)
    vel[1] = vel[1] + gravity * dt

    p = wp.vec2((float(i) + 0.5) * dx, (float(j) + 0.5) * dx)

    # Ground and side walls.
    if (i < 3 or p[0] < 0.155) and vel[0] < 0.0:
        vel[0] = 0.0
    if (i > NX - 4 or p[0] > 0.885) and vel[0] > 0.0:
        vel[0] = 0.0
    if (j < 3 or p[1] < 0.045) and vel[1] < 0.0:
        vel[1] = 0.0
        vel[0] = vel[0] * 0.45
    if j > NY - 4 and vel[1] > 0.0:
        vel[1] = 0.0

    vel_before_tool = vel
    sdf, n = box_sdf_normal(p, tool_center, tool_c, tool_s, tool_half)
    if sdf < 1.25 * dx:
        r = p - tool_center
        contact_vel = tool_vel + wp.vec2(-tool_omega * r[1], tool_omega * r[0])
        rel = vel - contact_vel
        vn = wp.dot(rel, n)
        if vn < 0.0:
            rel = rel - n * vn
            tangent = rel - n * wp.dot(rel, n)
            tl = wp.length(tangent)
            if tl > 1.0e-8:
                rel = rel - tangent * wp.min(1.0, tool_mu * (-vn) / tl)
            vel = contact_vel + rel
        if sdf < 0.0:
            vel = vel + n * (-sdf / wp.max(dt, 1.0e-6)) * 0.04

        dp = (vel - vel_before_tool) * m
        tool_f = -dp / wp.max(dt, 1.0e-6)
        r2 = p - tool_center
        wp.atomic_add(force_accum, 0, tool_f[0])
        wp.atomic_add(force_accum, 1, tool_f[1])
        wp.atomic_add(force_accum, 2, r2[0] * tool_f[1] - r2[1] * tool_f[0])

    vel = vel * 0.997
    speed = wp.length(vel)
    if speed > max_speed:
        vel = vel * (max_speed / speed)

    grid_vx[tid] = vel[0]
    grid_vy[tid] = vel[1]


@wp.kernel
def g2p_kernel(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    C: wp.array(dtype=wp.mat22),
    grid_vx: wp.array(dtype=float),
    grid_vy: wp.array(dtype=float),
    grid_m: wp.array(dtype=float),
    dt: float,
    dx: float,
    inv_dx: float,
):
    p = wp.tid()
    xp = x[p]
    grid_pos = xp * inv_dx
    base_x = int(wp.floor(grid_pos[0] - 0.5))
    base_y = int(wp.floor(grid_pos[1] - 0.5))
    fx = grid_pos - wp.vec2(float(base_x), float(base_y))

    wx = wp.vec3(
        0.5 * (1.5 - fx[0]) * (1.5 - fx[0]),
        0.75 - (fx[0] - 1.0) * (fx[0] - 1.0),
        0.5 * (fx[0] - 0.5) * (fx[0] - 0.5),
    )
    wy = wp.vec3(
        0.5 * (1.5 - fx[1]) * (1.5 - fx[1]),
        0.75 - (fx[1] - 1.0) * (fx[1] - 1.0),
        0.5 * (fx[1] - 0.5) * (fx[1] - 0.5),
    )

    new_v = wp.vec2(0.0, 0.0)
    new_C = wp.mat22(0.0, 0.0, 0.0, 0.0)
    for gx in range(3):
        for gy in range(3):
            i = base_x + gx
            j = base_y + gy
            if i >= 0 and i < NX and j >= 0 and j < NY:
                weight = wx[gx] * wy[gy]
                idx = grid_index(i, j)
                gv = wp.vec2(grid_vx[idx], grid_vy[idx])
                dpos = (wp.vec2(float(gx), float(gy)) - fx) * dx
                new_v = new_v + gv * weight
                new_C = new_C + wp.outer(gv, dpos) * (4.0 * inv_dx * weight)

    xp = xp + new_v * dt
    xp[0] = wp.clamp(xp[0], 0.02, (float(NX) - 2.0) * dx)
    xp[1] = wp.clamp(xp[1], 0.02, (float(NY) - 2.0) * dx)
    x[p] = xp
    v[p] = new_v
    C[p] = new_C


@dataclass
class ToolState:
    center: np.ndarray
    velocity: np.ndarray
    angle: float
    angular_velocity: float
    half: np.ndarray


@dataclass
class ToolBody:
    center: np.ndarray
    velocity: np.ndarray
    angle: float
    angular_velocity: float


def tool_trajectory(t: float, dt: float) -> ToolState:
    def pose(time: float) -> tuple[np.ndarray, float]:
        # Four-phase shovel motion: approach, insert, push, lift.
        if time < 0.18:
            u = time / 0.18
            center = np.array([0.24 + 0.10 * u, 0.36 - 0.08 * u], dtype=np.float32)
            angle = -0.82
        elif time < 0.42:
            u = (time - 0.18) / 0.24
            center = np.array([0.34 + 0.04 * u, 0.28 - 0.06 * u], dtype=np.float32)
            angle = -0.82 + 0.38 * u
        elif time < 1.38:
            u = (time - 0.42) / 0.96
            center = np.array([0.38 + 0.30 * u, 0.222 - 0.004 * np.sin(np.pi * u)], dtype=np.float32)
            angle = -0.44 + 0.08 * u
        else:
            u = min((time - 1.38) / 0.42, 1.0)
            center = np.array([0.68 + 0.06 * u, 0.222 + 0.12 * u], dtype=np.float32)
            angle = -0.36
        return center, angle

    c0, a0 = pose(t)
    c1, a1 = pose(t + dt)
    vel = (c1 - c0) / dt
    omega = (a1 - a0) / dt
    return ToolState(
        center=c0,
        velocity=vel.astype(np.float32),
        angle=a0,
        angular_velocity=float(omega),
        half=np.array([0.078, 0.014], dtype=np.float32),
    )


def create_particles() -> np.ndarray:
    rng = np.random.default_rng(4)
    xs = []
    dx = 1.0 / 160.0
    # Dense material points filling a low tray. These are not rigid spheres.
    for y in np.arange(0.060, 0.255, dx * 0.60):
        for x in np.arange(0.18, 0.84, dx * 0.60):
            if rng.random() < 0.88:
                xs.append(
                    [
                        x + rng.uniform(-0.0012, 0.0012),
                        y + rng.uniform(-0.0012, 0.0012),
                    ]
                )
    return np.asarray(xs, dtype=np.float32)


def draw_frame(
    pos: np.ndarray,
    tool: ToolState,
    command: ToolState | None = None,
    reaction: np.ndarray | None = None,
    force_history: list[float] | None = None,
    width: int = 960,
    height: int = 576,
) -> np.ndarray:
    frame = np.full((height, width, 3), (24, 27, 29), dtype=np.uint8)
    scale = width / 1.0

    def to_px(p: np.ndarray) -> tuple[int, int]:
        return int(p[0] * scale), int(height - p[1] * scale)

    # Tray and ground.
    cv2.rectangle(frame, to_px(np.array([0.14, 0.035])), to_px(np.array([0.88, 0.325])), (48, 53, 56), 6)
    cv2.line(frame, to_px(np.array([0.0, 0.035])), to_px(np.array([1.0, 0.035])), (80, 86, 88), 2)

    # Draw material as a density/height impression rather than visible balls.
    pts = pos[np.argsort(pos[:, 1])]
    colors = np.array([[57, 111, 149], [72, 141, 182], [91, 174, 215], [118, 198, 234]], dtype=np.uint8)
    for idx, p in enumerate(pts):
        px, py = to_px(p)
        if 0 <= px < width and 0 <= py < height:
            c = colors[(idx + int(p[0] * 997.0)) % len(colors)]
            cv2.circle(frame, (px, py), 2, (int(c[0]), int(c[1]), int(c[2])), -1, lineType=cv2.LINE_AA)

    # Height-map silhouette.
    bins = 96
    x_min, x_max = 0.16, 0.86
    top = np.zeros(bins, dtype=np.float32)
    valid = (pos[:, 0] >= x_min) & (pos[:, 0] <= x_max)
    ids = np.clip(((pos[valid, 0] - x_min) / (x_max - x_min) * (bins - 1)).astype(np.int32), 0, bins - 1)
    np.maximum.at(top, ids, pos[valid, 1])
    surface = []
    for i, y in enumerate(top):
        if y > 0.0:
            x = x_min + (x_max - x_min) * i / (bins - 1)
            surface.append(to_px(np.array([x, y + 0.004], dtype=np.float32)))
    if len(surface) > 2:
        cv2.polylines(frame, [np.array(surface, dtype=np.int32)], False, (246, 218, 142), 2, lineType=cv2.LINE_AA)

    def tool_poly(state: ToolState) -> np.ndarray:
        c = float(np.cos(state.angle))
        s = float(np.sin(state.angle))
        corners = []
        for sx in [-1.0, 1.0]:
            for sy in [-1.0, 1.0]:
                local = np.array([sx * state.half[0], sy * state.half[1]], dtype=np.float32)
                world = state.center + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]], dtype=np.float32)
                corners.append(to_px(world))
        return np.array([corners[0], corners[2], corners[3], corners[1]], dtype=np.int32)

    if command is not None:
        cmd_poly = tool_poly(command)
        cv2.polylines(frame, [cmd_poly], True, (94, 112, 122), 1, lineType=cv2.LINE_AA)

    order = tool_poly(tool)
    cv2.fillConvexPoly(frame, order, (16, 18, 20), lineType=cv2.LINE_AA)
    cv2.polylines(frame, [order], True, (226, 235, 236), 1, lineType=cv2.LINE_AA)

    if reaction is not None:
        mag = float(np.linalg.norm(reaction))
        start = np.array(tool.center, dtype=np.float32)
        end = start + np.asarray(reaction, dtype=np.float32) * 0.0048
        cv2.arrowedLine(frame, to_px(start), to_px(end), (88, 178, 255), 3, tipLength=0.22, line_type=cv2.LINE_AA)
        cv2.putText(frame, f"sand reaction {mag:5.1f}", (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 232, 220), 2, cv2.LINE_AA)

    if force_history:
        x0, y0, w, h = 700, 36, 228, 68
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (37, 41, 43), 1)
        hist = np.asarray(force_history[-90:], dtype=np.float32)
        peak = max(1.0, float(np.percentile(hist, 95)))
        pts = []
        for i, f in enumerate(hist):
            xh = x0 + 6 + int(i * (w - 12) / max(1, len(hist) - 1))
            yh = y0 + h - 8 - int(min(float(f) / peak, 1.0) * (h - 18))
            pts.append((xh, yh))
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, (88, 178, 255), 2, lineType=cv2.LINE_AA)

    cv2.putText(frame, "2-way coupled MLS-MPM sand + controlled shovel", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (224, 230, 230), 2, cv2.LINE_AA)
    return frame


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wp.init()
    wp.set_device("cuda:0")

    dx = 1.0 / 160.0
    inv_dx = 1.0 / dx
    dt = 4.0e-4
    frame_dt = 1.0 / 30.0
    substeps = int(frame_dt / dt)
    total_frames = 56

    pos_np = create_particles()
    n_particles = int(pos_np.shape[0])
    print(f"particles={n_particles} grid={int(NX)}x{int(NY)} substeps/frame={substeps}")

    x = wp.array(pos_np, dtype=wp.vec2, device="cuda:0")
    v = wp.zeros(n_particles, dtype=wp.vec2, device="cuda:0")
    C = wp.zeros(n_particles, dtype=wp.mat22, device="cuda:0")
    F = wp.array(np.tile(np.eye(2, dtype=np.float32), (n_particles, 1, 1)), dtype=wp.mat22, device="cuda:0")
    Jp = wp.ones(n_particles, dtype=float, device="cuda:0")
    grid_vx = wp.zeros(NGRID, dtype=float, device="cuda:0")
    grid_vy = wp.zeros(NGRID, dtype=float, device="cuda:0")
    grid_m = wp.zeros(NGRID, dtype=float, device="cuda:0")
    force_accum = wp.zeros(3, dtype=float, device="cuda:0")

    # This core demo uses dimensionless MPM units. Keeping p_mass and p_vol
    # comparable prevents the bed from collapsing into a single contact line.
    p_mass = 1.0
    p_vol = 1.0
    young = 2.4e3
    nu = 0.24
    mu0 = young / (2.0 * (1.0 + nu))
    lam0 = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    cmd0 = tool_trajectory(0.0, dt)
    body = ToolBody(
        center=cmd0.center.copy(),
        velocity=cmd0.velocity.copy(),
        angle=float(cmd0.angle),
        angular_velocity=float(cmd0.angular_velocity),
    )
    coupling_gain = 2.5e-3
    tool_mass = 7.5
    tool_inertia = 0.075
    kp = 360.0
    kd = 30.0
    angle_kp = 36.0
    angle_kd = 6.5
    max_drive = 42.0
    max_sand_force = 60.0
    max_torque = 18.0

    frames: list[np.ndarray] = []
    force_history: list[float] = []
    last_reaction = np.zeros(2, dtype=np.float32)
    sim_t = 0.0
    for frame_id in range(total_frames):
        for _ in range(substeps):
            command = tool_trajectory(sim_t, dt)
            tool = ToolState(
                center=body.center.astype(np.float32),
                velocity=body.velocity.astype(np.float32),
                angle=float(body.angle),
                angular_velocity=float(body.angular_velocity),
                half=command.half,
            )
            grid_vx.zero_()
            grid_vy.zero_()
            grid_m.zero_()
            force_accum.zero_()
            wp.launch(
                p2g_kernel,
                dim=n_particles,
                inputs=[
                    x,
                    v,
                    C,
                    F,
                    Jp,
                    grid_vx,
                    grid_vy,
                    grid_m,
                    dt,
                    dx,
                    inv_dx,
                    p_mass,
                    p_vol,
                    mu0,
                    lam0,
                    0.46,
                ],
                device="cuda:0",
            )
            wp.launch(
                grid_update_kernel,
                dim=NGRID,
                inputs=[
                    grid_vx,
                    grid_vy,
                    grid_m,
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
                    force_accum,
                ],
                device="cuda:0",
            )
            wp.launch(
                g2p_kernel,
                dim=n_particles,
                inputs=[x, v, C, grid_vx, grid_vy, grid_m, dt, dx, inv_dx],
                device="cuda:0",
            )
            raw_force = force_accum.numpy()
            sand_force = raw_force[:2].astype(np.float32) * coupling_gain
            sand_torque = float(raw_force[2]) * coupling_gain
            sand_mag = float(np.linalg.norm(sand_force))
            if sand_mag > max_sand_force:
                sand_force *= max_sand_force / sand_mag
                sand_mag = max_sand_force

            drive = kp * (command.center - body.center) + kd * (command.velocity - body.velocity)
            drive_mag = float(np.linalg.norm(drive))
            if drive_mag > max_drive:
                drive *= max_drive / drive_mag

            body.velocity += ((drive + sand_force) / tool_mass) * dt
            body.velocity *= 0.9995
            body.velocity = np.clip(body.velocity, [-0.65, -0.65], [0.65, 0.65])
            body.center += body.velocity * dt
            body.center[0] = float(np.clip(body.center[0], 0.14, 0.88))
            body.center[1] = float(np.clip(body.center[1], 0.08, 0.45))

            angle_err = float(command.angle - body.angle)
            drive_torque = angle_kp * angle_err + angle_kd * (command.angular_velocity - body.angular_velocity)
            total_torque = float(np.clip(drive_torque + sand_torque, -max_torque, max_torque))
            body.angular_velocity += (total_torque / tool_inertia) * dt
            body.angular_velocity *= 0.998
            body.angular_velocity = float(np.clip(body.angular_velocity, -3.0, 3.0))
            body.angle += body.angular_velocity * dt

            last_reaction = sand_force.copy()
            sim_t += dt
        wp.synchronize()
        pos = x.numpy()
        command = tool_trajectory(sim_t, dt)
        actual = ToolState(
            center=body.center.astype(np.float32),
            velocity=body.velocity.astype(np.float32),
            angle=float(body.angle),
            angular_velocity=float(body.angular_velocity),
            half=command.half,
        )
        force_history.append(float(np.linalg.norm(last_reaction)))
        frames.append(draw_frame(pos, actual, command, last_reaction, force_history))
        if frame_id % 12 == 0:
            finite = np.isfinite(pos).all(axis=1)
            pmin = pos[finite].min(axis=0) if finite.any() else np.array([np.nan, np.nan])
            pmax = pos[finite].max(axis=0) if finite.any() else np.array([np.nan, np.nan])
            err = float(np.linalg.norm(command.center - body.center))
            print(
                f"frame={frame_id:03d} sim_t={sim_t:.3f} finite={finite.sum()} "
                f"min={pmin} max={pmax} reaction={force_history[-1]:.2f} track_err={err:.3f}"
            )

    writer = cv2.VideoWriter(
        VIDEO_PATH.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (960, 576),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    cv2.imwrite(PREVIEW_PATH.as_posix(), frames[-1])
    print(f"video={VIDEO_PATH}")
    print(f"preview={PREVIEW_PATH}")


if __name__ == "__main__":
    run()
