# Sim2Sim Property Validation

The current project does not yet have a real-to-sim loop with real granular
observations. The practical validation step is therefore Sim2Sim:

```text
GT material properties -> MPM rollout under a revealing task
estimated material properties -> MPM rollout under the same task
```

If the two rollouts produce similar deformation, pile-up, and reaction profiles,
the estimated posterior is useful as a simulator state. If the rollouts diverge,
the video shows which physical behavior the estimate fails to reproduce.

## Command

Bulldozing wedge diagnostic, recommended for presentation and Sim2Sim sanity
checks:

```bash
python scripts/render_sim2sim_property_compare.py --config configs/rendering/sim2sim_bulldozing_wedge.json
```

The older deep-rake comparison remains available:

```bash
python scripts/render_sim2sim_property_compare.py --config configs/rendering/sim2sim_property_compare.json
```

The same commands are available on systems with `make`:

```bash
make sim2sim-wedge
make sim2sim-property
```

## Current Wedge Output

```text
outputs/sim2sim_bulldozing_wedge/sim2sim_bulldozing_wedge.mp4
outputs/sim2sim_bulldozing_wedge/sim2sim_bulldozing_wedge_preview.png
outputs/sim2sim_bulldozing_wedge/sim2sim_bulldozing_wedge_sheet.png
outputs/sim2sim_bulldozing_wedge/sim2sim_bulldozing_wedge_metrics.csv
outputs/sim2sim_bulldozing_wedge/sim2sim_bulldozing_wedge_metadata.json
```

## Deep-Rake Output

```text
outputs/sim2sim_property_compare/sim2sim_property_compare.mp4
outputs/sim2sim_property_compare/sim2sim_property_compare_preview.png
outputs/sim2sim_property_compare/sim2sim_property_compare_sheet.png
outputs/sim2sim_property_compare/sim2sim_property_compare_metrics.csv
outputs/sim2sim_property_compare/sim2sim_property_compare_metadata.json
```

## Material Mapping

The CSV row supplies:

```text
rho, phi_deg, delta_deg, cohesion_kpa
```

The local MPM engine currently uses:

```text
p_mass    <- rho / rho_reference
dp_alpha  <- Drucker-Prager alpha from phi_deg
cohesion  <- cohesion_kpa * scale
tool_mu   <- tan(delta_deg) * scale, clipped
young     <- young_base + young_per_kpa * cohesion_kpa
```

This is an approximate simulation-control mapping, not a claim that the local
MPM engine is a calibrated physical soil model. In particular, `rho` mainly
affects mass/reaction behavior here; deformation differences are usually more
visible through `phi`, `delta`, `cohesion`, and `tool_mu`.

## Revealing Tasks

### Bulldozing Wedge

The recommended task uses a vertical blade as a retaining wall/bulldozer blade:

```text
insert_duration = 0.24
push_duration = 1.18
x_start = 0.225
x_end = 0.775
z_work = 0.150
blade_half_x = 0.105
blade_half_y = 0.145
blade_half_z = 0.012
angle = pi / 2
```

This task exposes:

- `phi` through the failure wedge angle and surface slope
- `cohesion` through clumping, retained vertical faces, and pile stability
- `delta/tool_mu` through blade-adjacent sticking and reaction force
- `rho` mostly through reaction magnitude

### Deep Rake

The older comparison task is a deeper and slightly longer rake than the default
density render:

```text
depth_scale = 1.18
drag_distance_scale = 1.12
angle_offset = -0.07
```

It is intended to expose:

- cohesion through pile retention and clumping
- internal friction through shear band and free-surface slope
- interface friction through reaction force and blade-adjacent drag
- density mostly through reaction magnitude

## Current Interpretation

The current best-validation posterior produces very similar GT and EST
bulldozing-wedge rollouts. That visual similarity is expected rather than a
rendering failure: the selected row has small property errors.

```text
GT:   rho=1261.5, phi=30.7, delta=23.7, cohesion=8.5
EST:  rho=1268.6, phi=32.9, delta=22.2, cohesion=8.7
abs:  rho=7.1,    phi=2.2,  delta=1.5,  cohesion=0.24
```

The final wedge metrics stay close as well:

```text
force |F| GT/EST:       1128.42 / 1106.22
force absolute error:     22.21
mean particle divergence:  0.0187 m
p90 particle divergence:   0.0383 m
zmax GT/EST:               0.613 / 0.615
```

This is a useful result: the posterior is close enough to preserve the material
behavior under a property-revealing task. For harder validation, use held-out
action families and materials whose `phi`, `delta`, and `cohesion` differ more
strongly.
