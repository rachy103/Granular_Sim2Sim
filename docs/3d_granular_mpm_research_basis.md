# 3D Granular MPM Research Basis

## Research Position

The project should treat granular media as the object being physically interrogated, not merely as a disturbance for a robot policy. A useful environment therefore needs a first-class 3D granular state before adding camera or force-learning pipelines.

The current 2D MPM + MuJoCo coupling is a good coupling smoke test, but it is not enough for granular analysis. The next core target is a 3D MLS-MPM solver with a rigid SDF tool interface, calibrated material parameters, and measurable reaction signatures.

## State

For material point `p`:

```text
x_p      position in R^3
v_p      velocity in R^3
F_p      elastic deformation gradient in R^{3x3}
C_p      affine velocity field in R^{3x3}
Jp_p     plastic volume/compaction history
m_p      particle mass
V_p      particle volume
```

For grid node `i`:

```text
m_i      nodal mass
v_i      nodal velocity
f_i      nodal external/contact impulse accumulator
```

Hidden material parameters:

```text
theta = {
  rho,              density
  E, nu,            elastic modulus and Poisson ratio
  phi,              internal friction angle
  c,                cohesion, often zero for dry sand
  beta_damp,        numerical/physical damping
  mu_tool,          tool-sand friction
  packing,          initial density/void ratio
  compaction_state  history-dependent plastic compression
}
```

## MLS-MPM Step

Use quadratic B-spline weights `w_ip = N((x_i - x_p) / dx)`.

Particle to grid:

```text
m_i^n = sum_p w_ip m_p

(m v)_i^n = sum_p w_ip [
  m_p v_p^n
  + A_p^n (x_i - x_p^n)
]

A_p^n = -dt V_p (P_p^n F_p^{nT}) (4 / dx^2) + m_p C_p^n
```

Grid update:

```text
v_i^- = (m v)_i / m_i + dt g
v_i^+ = boundary_and_tool_contact(v_i^-, x_i)
```

Grid to particle:

```text
v_p^{n+1} = sum_i w_ip v_i^+

C_p^{n+1} = sum_i (4 / dx^2) w_ip v_i^+ (x_i - x_p^n)^T

x_p^{n+1} = x_p^n + dt v_p^{n+1}

F_trial = (I + dt C_p^{n+1}) F_p^n
F_p^{n+1} = project_plasticity(F_trial, theta)
```

## Granular Plasticity

The minimum useful target is Drucker-Prager plasticity in 3D. With compression-positive pressure convention, a common yield form is:

```text
f(s, p) = ||s|| + alpha p - k <= 0
```

where:

```text
p = tr(sigma) / 3
s = sigma - p I
alpha = 2 sin(phi) / (sqrt(3) (3 - sin(phi)))
k = 6 c cos(phi) / (sqrt(3) (3 - sin(phi)))
```

For dry cohesionless sand, set `c ~= 0`, so the primary parameter is `phi`. The implementation can use the common MPM log-strain return mapping:

```text
F_trial = U Sigma V^T
epsilon = log(Sigma)
epsilon_hat = epsilon - mean(epsilon) 1

yield = ||epsilon_hat|| + alpha tr(epsilon) - k
```

If `yield <= 0`, keep the elastic state. Otherwise, project the log strain back to the yield surface and update `Jp` as the plastic volume history. This is the piece that should become more rigorous than the current DP-lite 2D prototype.

## Tool Contact

Represent the shovel or probe as an SDF:

```text
phi_tool(x, q_tool)     signed distance
n = grad(phi_tool)      contact normal
v_tool(x) = v_com + omega x (x - x_com)
```

For grid nodes inside the contact band:

```text
v_rel = v_i^- - v_tool(x_i)

if phi_tool(x_i) < h and dot(v_rel, n) < 0:
  remove inward normal velocity
  apply Coulomb tangential projection with mu_tool
```

Reaction on the tool:

```text
Delta p_i = m_i (v_i^+ - v_i^-)
f_tool = - sum_i Delta p_i / dt
tau_tool = - sum_i (x_i - x_tool) x Delta p_i / dt
```

This wrench is what gets sent back to MuJoCo.

## What We Should Measure Before Policy

A 3D granular env becomes useful when the following signatures are stable and parameter-sensitive:

```text
intrusion force vs depth
drag force vs speed
drag force vs attack angle
failure wedge / pile-up geometry
sinkage and compaction history
force hysteresis during insert-push-lift
tool stuck / slip transitions
excavated volume and energy
```

The first policies should not use raw camera or noisy learned force encoders. They should use clean simulation observables first:

```text
tool pose and velocity
reaction wrench
surface height field
estimated displaced volume
latent material parameters for supervision
```

Then sensors can be degraded later into realistic vision/force pipelines.

## Implementation Roadmap

1. Build standalone 3D Warp MLS-MPM sand engine.
2. Add 3D SDF tools: flat blade, wedge shovel, cylinder probe.
3. Validate quasi-static intrusion and horizontal drag curves.
4. Couple MuJoCo Franka/tool to the 3D MPM wrench.
5. Add parameter randomization and system identification tasks.
6. Only then add learning policies and sensor models.

The immediate next code target should be `warp_sand_mpm_3d.py`: a 3D solver with a moving SDF blade and separate render outputs for top/side views plus force curves.
