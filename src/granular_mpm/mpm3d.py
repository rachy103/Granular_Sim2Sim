from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


NX = wp.constant(72)
NY = wp.constant(40)
NZ = wp.constant(48)
NGRID = 72 * 40 * 48


@dataclass
class SandMPM3DConfig:
    dx: float = 1.0 / 72.0
    dt: float = 4.0e-4
    p_mass: float = 1.0
    p_vol: float = 1.0
    young: float = 2.2e3
    poisson: float = 0.24
    dp_alpha: float = 0.42
    cohesion: float = 0.0
    gravity: float = -1.35
    tool_mu: float = 0.44
    max_grid_speed: float = 1.3
    damping: float = 0.998
    contact_band_cells: float = 1.35
    penetration_stiffness: float = 0.045
    seed: int = 7


@dataclass
class ToolState3D:
    center: np.ndarray
    velocity: np.ndarray
    angle: float
    angular_velocity: float
    half: np.ndarray

    @property
    def omega(self) -> np.ndarray:
        return np.array([0.0, self.angular_velocity, 0.0], dtype=np.float32)


@wp.func
def grid_index(i: int, j: int, k: int) -> int:
    return i + NX * (j + NY * k)


@wp.func
def rot_y(v: wp.vec3, c: float, s: float) -> wp.vec3:
    return wp.vec3(c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2])


@wp.func
def rot_y_t(v: wp.vec3, c: float, s: float) -> wp.vec3:
    return wp.vec3(c * v[0] - s * v[2], v[1], s * v[0] + c * v[2])


@wp.func
def box_sdf_normal_3d(p: wp.vec3, center: wp.vec3, c: float, s: float, half: wp.vec3):
    local = rot_y_t(p - center, c, s)
    ax = wp.abs(local[0])
    ay = wp.abs(local[1])
    az = wp.abs(local[2])
    qx = ax - half[0]
    qy = ay - half[1]
    qz = az - half[2]

    ox = wp.max(qx, 0.0)
    oy = wp.max(qy, 0.0)
    oz = wp.max(qz, 0.0)
    outside = wp.length(wp.vec3(ox, oy, oz))
    inside = wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)
    sdf = outside + inside

    sx = wp.where(local[0] >= 0.0, 1.0, -1.0)
    sy = wp.where(local[1] >= 0.0, 1.0, -1.0)
    sz = wp.where(local[2] >= 0.0, 1.0, -1.0)
    n_local = wp.vec3(0.0, 0.0, 1.0)
    if qx > 0.0 or qy > 0.0 or qz > 0.0:
        g = wp.vec3(
            wp.where(qx > 0.0, qx * sx, 0.0),
            wp.where(qy > 0.0, qy * sy, 0.0),
            wp.where(qz > 0.0, qz * sz, 0.0),
        )
        gl = wp.length(g)
        if gl > 1.0e-8:
            n_local = g / gl
    else:
        if qx > qy and qx > qz:
            n_local = wp.vec3(sx, 0.0, 0.0)
        elif qy > qz:
            n_local = wp.vec3(0.0, sy, 0.0)
        else:
            n_local = wp.vec3(0.0, 0.0, sz)

    return sdf, rot_y(n_local, c, s)


@wp.kernel
def p2g_kernel_3d(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    F: wp.array(dtype=wp.mat33),
    Jp: wp.array(dtype=float),
    grid_vx: wp.array(dtype=float),
    grid_vy: wp.array(dtype=float),
    grid_vz: wp.array(dtype=float),
    grid_m: wp.array(dtype=float),
    dt: float,
    dx: float,
    inv_dx: float,
    p_mass: float,
    p_vol: float,
    mu0: float,
    lam0: float,
    dp_alpha: float,
    cohesion: float,
):
    p = wp.tid()
    xp = x[p]
    vp = v[p]
    Cp = C[p]

    I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    F_trial = (I + Cp * dt) * F[p]

    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F_trial, U, sig, V)

    old_j = wp.max(1.0e-6, sig[0] * sig[1] * sig[2])

    e0 = wp.log(wp.max(sig[0], 1.0e-4))
    e1 = wp.log(wp.max(sig[1], 1.0e-4))
    e2 = wp.log(wp.max(sig[2], 1.0e-4))
    tr = e0 + e1 + e2
    mean = tr / 3.0
    d0 = e0 - mean
    d1 = e1 - mean
    d2 = e2 - mean
    dev_norm = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)

    if tr > 0.0:
        e0 = 0.0
        e1 = 0.0
        e2 = 0.0
    else:
        yield_value = dev_norm + dp_alpha * tr - cohesion
        if yield_value > 0.0 and dev_norm > 1.0e-8:
            scale = wp.max(0.0, -dp_alpha * tr) / dev_norm
            e0 = mean + d0 * scale
            e1 = mean + d1 * scale
            e2 = mean + d2 * scale

    s0 = wp.exp(wp.clamp(e0, -0.35, 0.22))
    s1 = wp.exp(wp.clamp(e1, -0.35, 0.22))
    s2 = wp.exp(wp.clamp(e2, -0.35, 0.22))
    new_j = wp.max(1.0e-6, s0 * s1 * s2)
    Jp[p] = wp.clamp(Jp[p] * old_j / new_j, 0.28, 3.2)
    Fp = U * wp.mat33(s0, 0.0, 0.0, 0.0, s1, 0.0, 0.0, 0.0, s2) * wp.transpose(V)
    F[p] = Fp

    J = wp.determinant(Fp)
    R = U * wp.transpose(V)
    stress = (Fp - R) * (2.0 * mu0) * wp.transpose(Fp) + I * (lam0 * J * (J - 1.0))
    stress = stress * (-dt * p_vol * 4.0 * inv_dx * inv_dx)
    affine = stress + Cp * p_mass

    grid_pos = xp * inv_dx
    base_x = int(wp.floor(grid_pos[0] - 0.5))
    base_y = int(wp.floor(grid_pos[1] - 0.5))
    base_z = int(wp.floor(grid_pos[2] - 0.5))
    fx = grid_pos - wp.vec3(float(base_x), float(base_y), float(base_z))

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
    wz = wp.vec3(
        0.5 * (1.5 - fx[2]) * (1.5 - fx[2]),
        0.75 - (fx[2] - 1.0) * (fx[2] - 1.0),
        0.5 * (fx[2] - 0.5) * (fx[2] - 0.5),
    )

    for gx in range(3):
        for gy in range(3):
            for gz in range(3):
                i = base_x + gx
                j = base_y + gy
                k = base_z + gz
                if i >= 0 and i < NX and j >= 0 and j < NY and k >= 0 and k < NZ:
                    weight = wx[gx] * wy[gy] * wz[gz]
                    dpos = (wp.vec3(float(gx), float(gy), float(gz)) - fx) * dx
                    momentum = vp * p_mass + affine * dpos
                    idx = grid_index(i, j, k)
                    wp.atomic_add(grid_vx, idx, weight * momentum[0])
                    wp.atomic_add(grid_vy, idx, weight * momentum[1])
                    wp.atomic_add(grid_vz, idx, weight * momentum[2])
                    wp.atomic_add(grid_m, idx, weight * p_mass)


@wp.kernel
def grid_update_kernel_3d(
    grid_vx: wp.array(dtype=float),
    grid_vy: wp.array(dtype=float),
    grid_vz: wp.array(dtype=float),
    grid_m: wp.array(dtype=float),
    dt: float,
    dx: float,
    gravity: float,
    damping: float,
    tool_center: wp.vec3,
    tool_vel: wp.vec3,
    tool_omega: wp.vec3,
    tool_c: float,
    tool_s: float,
    tool_half: wp.vec3,
    tool_mu: float,
    contact_band_cells: float,
    penetration_stiffness: float,
    max_speed: float,
    force_accum: wp.array(dtype=float),
):
    tid = wp.tid()
    i = tid % NX
    j = (tid / NX) % NY
    k = tid / (NX * NY)
    m = grid_m[tid]

    if m <= 0.0:
        return

    vel = wp.vec3(grid_vx[tid] / m, grid_vy[tid] / m, grid_vz[tid] / m)
    vel[2] = vel[2] + gravity * dt
    p = wp.vec3((float(i) + 0.5) * dx, (float(j) + 0.5) * dx, (float(k) + 0.5) * dx)

    # Rectangular tray boundaries.
    if (i < 3 or p[0] < 0.145) and vel[0] < 0.0:
        vel[0] = 0.0
    if (i > NX - 4 or p[0] > 0.895) and vel[0] > 0.0:
        vel[0] = 0.0
    if (j < 3 or p[1] < 0.095) and vel[1] < 0.0:
        vel[1] = 0.0
    if (j > NY - 4 or p[1] > 0.460) and vel[1] > 0.0:
        vel[1] = 0.0
    if (k < 3 or p[2] < 0.040) and vel[2] < 0.0:
        vel[2] = 0.0
        vel[0] = vel[0] * 0.50
        vel[1] = vel[1] * 0.50
    if k > NZ - 4 and vel[2] > 0.0:
        vel[2] = 0.0

    vel_before_tool = vel
    sdf, n = box_sdf_normal_3d(p, tool_center, tool_c, tool_s, tool_half)
    if sdf < contact_band_cells * dx:
        r = p - tool_center
        contact_vel = tool_vel + wp.cross(tool_omega, r)
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
            vel = vel + n * (-sdf / wp.max(dt, 1.0e-6)) * penetration_stiffness

        dp = (vel - vel_before_tool) * m
        tool_f = -dp / wp.max(dt, 1.0e-6)
        tau = wp.cross(r, tool_f)
        wp.atomic_add(force_accum, 0, tool_f[0])
        wp.atomic_add(force_accum, 1, tool_f[1])
        wp.atomic_add(force_accum, 2, tool_f[2])
        wp.atomic_add(force_accum, 3, tau[0])
        wp.atomic_add(force_accum, 4, tau[1])
        wp.atomic_add(force_accum, 5, tau[2])

    vel = vel * damping
    speed = wp.length(vel)
    if speed > max_speed:
        vel = vel * (max_speed / speed)

    grid_vx[tid] = vel[0]
    grid_vy[tid] = vel[1]
    grid_vz[tid] = vel[2]


@wp.kernel
def g2p_kernel_3d(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    grid_vx: wp.array(dtype=float),
    grid_vy: wp.array(dtype=float),
    grid_vz: wp.array(dtype=float),
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
    base_z = int(wp.floor(grid_pos[2] - 0.5))
    fx = grid_pos - wp.vec3(float(base_x), float(base_y), float(base_z))

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
    wz = wp.vec3(
        0.5 * (1.5 - fx[2]) * (1.5 - fx[2]),
        0.75 - (fx[2] - 1.0) * (fx[2] - 1.0),
        0.5 * (fx[2] - 0.5) * (fx[2] - 0.5),
    )

    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for gx in range(3):
        for gy in range(3):
            for gz in range(3):
                i = base_x + gx
                j = base_y + gy
                k = base_z + gz
                if i >= 0 and i < NX and j >= 0 and j < NY and k >= 0 and k < NZ:
                    weight = wx[gx] * wy[gy] * wz[gz]
                    idx = grid_index(i, j, k)
                    gv = wp.vec3(grid_vx[idx], grid_vy[idx], grid_vz[idx])
                    dpos = (wp.vec3(float(gx), float(gy), float(gz)) - fx) * dx
                    new_v = new_v + gv * weight
                    new_C = new_C + wp.outer(gv, dpos) * (4.0 * inv_dx * weight)

    xp = xp + new_v * dt
    xp[0] = wp.clamp(xp[0], 0.02, (float(NX) - 2.0) * dx)
    xp[1] = wp.clamp(xp[1], 0.02, (float(NY) - 2.0) * dx)
    xp[2] = wp.clamp(xp[2], 0.02, (float(NZ) - 2.0) * dx)
    x[p] = xp
    v[p] = new_v
    C[p] = new_C


def create_block_particles(config: SandMPM3DConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    dx = config.dx
    spacing = dx * 0.72
    particles: list[list[float]] = []
    for z in np.arange(0.058, 0.258, spacing):
        for y in np.arange(0.135, 0.425, spacing):
            for x in np.arange(0.178, 0.845, spacing):
                if rng.random() < 0.62:
                    particles.append(
                        [
                            float(x + rng.uniform(-0.0038, 0.0038)),
                            float(y + rng.uniform(-0.0038, 0.0038)),
                            float(z + rng.uniform(-0.0032, 0.0032)),
                        ]
                    )
    return np.asarray(particles, dtype=np.float32)


class SandMPM3D:
    def __init__(self, config: SandMPM3DConfig | None = None, device: str = "cuda:0"):
        self.config = config or SandMPM3DConfig()
        self.device = device
        wp.init()
        wp.set_device(device)

        pos_np = create_block_particles(self.config)
        self.n_particles = int(pos_np.shape[0])
        self.x = wp.array(pos_np, dtype=wp.vec3, device=device)
        self.v = wp.zeros(self.n_particles, dtype=wp.vec3, device=device)
        self.C = wp.zeros(self.n_particles, dtype=wp.mat33, device=device)
        eye = np.tile(np.eye(3, dtype=np.float32), (self.n_particles, 1, 1))
        self.F = wp.array(eye, dtype=wp.mat33, device=device)
        self.Jp = wp.ones(self.n_particles, dtype=float, device=device)
        self.grid_vx = wp.zeros(NGRID, dtype=float, device=device)
        self.grid_vy = wp.zeros(NGRID, dtype=float, device=device)
        self.grid_vz = wp.zeros(NGRID, dtype=float, device=device)
        self.grid_m = wp.zeros(NGRID, dtype=float, device=device)
        self.force_accum = wp.zeros(6, dtype=float, device=device)

        nu = self.config.poisson
        young = self.config.young
        self.mu0 = young / (2.0 * (1.0 + nu))
        self.lam0 = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def step(self, tool: ToolState3D, substeps: int = 1) -> np.ndarray:
        cfg = self.config
        inv_dx = 1.0 / cfg.dx
        total_force = np.zeros(6, dtype=np.float32)
        for _ in range(substeps):
            self.grid_vx.zero_()
            self.grid_vy.zero_()
            self.grid_vz.zero_()
            self.grid_m.zero_()
            self.force_accum.zero_()
            wp.launch(
                p2g_kernel_3d,
                dim=self.n_particles,
                inputs=[
                    self.x,
                    self.v,
                    self.C,
                    self.F,
                    self.Jp,
                    self.grid_vx,
                    self.grid_vy,
                    self.grid_vz,
                    self.grid_m,
                    cfg.dt,
                    cfg.dx,
                    inv_dx,
                    cfg.p_mass,
                    cfg.p_vol,
                    self.mu0,
                    self.lam0,
                    cfg.dp_alpha,
                    cfg.cohesion,
                ],
                device=self.device,
            )
            wp.launch(
                grid_update_kernel_3d,
                dim=NGRID,
                inputs=[
                    self.grid_vx,
                    self.grid_vy,
                    self.grid_vz,
                    self.grid_m,
                    cfg.dt,
                    cfg.dx,
                    cfg.gravity,
                    cfg.damping,
                    wp.vec3(float(tool.center[0]), float(tool.center[1]), float(tool.center[2])),
                    wp.vec3(float(tool.velocity[0]), float(tool.velocity[1]), float(tool.velocity[2])),
                    wp.vec3(float(tool.omega[0]), float(tool.omega[1]), float(tool.omega[2])),
                    float(np.cos(tool.angle)),
                    float(np.sin(tool.angle)),
                    wp.vec3(float(tool.half[0]), float(tool.half[1]), float(tool.half[2])),
                    cfg.tool_mu,
                    cfg.contact_band_cells,
                    cfg.penetration_stiffness,
                    cfg.max_grid_speed,
                    self.force_accum,
                ],
                device=self.device,
            )
            wp.launch(
                g2p_kernel_3d,
                dim=self.n_particles,
                inputs=[
                    self.x,
                    self.v,
                    self.C,
                    self.grid_vx,
                    self.grid_vy,
                    self.grid_vz,
                    self.grid_m,
                    cfg.dt,
                    cfg.dx,
                    inv_dx,
                ],
                device=self.device,
            )
            total_force += self.force_accum.numpy().astype(np.float32)
        return total_force / max(1, substeps)

    def positions(self) -> np.ndarray:
        wp.synchronize()
        return self.x.numpy()
