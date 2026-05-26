from __future__ import annotations

import math
import tempfile
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp


AIDIN_RIGHT = Path("/root/human2robot/spider/spider/assets/robots/aidin/right.xml")


def grain_bodies(prefix: str, count_x: int, count_y: int, count_z: int, radius: float, z0: float) -> str:
    lines: list[str] = []
    spacing = radius * 2.12
    for ix in range(count_x):
        for iy in range(count_y):
            for iz in range(count_z):
                x = (ix - (count_x - 1) / 2.0) * spacing
                y = (iy - (count_y - 1) / 2.0) * spacing
                z = z0 + iz * spacing
                name = f"{prefix}_{ix}_{iy}_{iz}"
                lines.append(
                    f'    <body name="{name}" pos="{x:.5f} {y:.5f} {z:.5f}">\n'
                    f'      <freejoint name="{name}_free"/>\n'
                    f'      <geom name="{name}_geom" type="sphere" size="{radius:.5f}" '
                    'density="1700" friction="1.1 0.02 0.0002" condim="3" '
                    'solref="0.008 1" solimp="0.9 0.95 0.001" rgba="0.72 0.56 0.34 1"/>\n'
                    '    </body>'
                )
    return "\n".join(lines)


def write_granular_only_xml(path: Path) -> None:
    grains = grain_bodies("grain", 6, 6, 3, 0.012, 0.045)
    path.write_text(
        f"""<mujoco model="mjwarp_granular_probe">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" cone="pyramidal" solver="CG" iterations="20" ls_iterations="30"/>
  <default>
    <geom contype="1" conaffinity="1"/>
  </default>
  <worldbody>
    <light pos="0 -1 1"/>
    <geom name="floor" type="plane" size="0.45 0.45 0.03" friction="1.2 0.02 0.0002" condim="3"/>
    <geom name="wall_xp" type="box" pos="0.17 0 0.07" size="0.01 0.18 0.07" friction="0.9 0.01 0.0001"/>
    <geom name="wall_xn" type="box" pos="-0.17 0 0.07" size="0.01 0.18 0.07" friction="0.9 0.01 0.0001"/>
    <geom name="wall_yp" type="box" pos="0 0.17 0.07" size="0.18 0.01 0.07" friction="0.9 0.01 0.0001"/>
    <geom name="wall_yn" type="box" pos="0 -0.17 0.07" size="0.18 0.01 0.07" friction="0.9 0.01 0.0001"/>
    <body name="probe_tool" pos="-0.13 0 0.052">
      <joint name="probe_slide" type="slide" axis="1 0 0" range="-0.16 0.16"/>
      <geom name="probe_blade" type="box" size="0.012 0.07 0.035" mass="0.12" friction="1.0 0.02 0.0002" condim="3"/>
    </body>
{grains}
  </worldbody>
  <actuator>
    <motor joint="probe_slide" gear="1" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
""",
        encoding="utf-8",
    )


def write_aidin_grains_xml(path: Path) -> None:
    text = AIDIN_RIGHT.read_text(encoding="utf-8")
    text = text.replace(
        'meshdir="."',
        f'meshdir="{AIDIN_RIGHT.parent.as_posix()}"',
        1,
    )
    text = text.replace(
        '<geom name="floor" type="plane" size="0 0 0.05" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0" />',
        '<geom name="floor" type="plane" size="0.6 0.6 0.05" rgba="0.9 0.9 0.9 1" contype="1" conaffinity="1" friction="1.2 0.02 0.0002" condim="3" />',
        1,
    )
    grains = grain_bodies("aidin_grain", 4, 4, 2, 0.012, 0.045)
    text = text.replace("</worldbody>", grains + "\n  </worldbody>", 1)
    path.write_text(text, encoding="utf-8")


def run_probe(label: str, xml_path: Path, nworld: int, steps: int, nconmax: int, njmax: int) -> None:
    mjm = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    mjd = mujoco.MjData(mjm)
    if mjm.nu:
        mjd.ctrl[:] = 5.0
    with wp.ScopedDevice("cuda:0"):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
        for _ in range(steps):
            mjw.step(m, d)
        wp.synchronize()

    qpos = d.qpos.numpy()
    qvel = d.qvel.numpy()
    nacon = int(np.asarray(d.nacon.numpy())[0])
    print(
        f"{label}: ngeom={mjm.ngeom} nbody={mjm.nbody} nq={mjm.nq} nv={mjm.nv} "
        f"nu={mjm.nu} nworld={nworld} steps={steps}"
    )
    print(
        f"{label}: finite_qpos={bool(np.isfinite(qpos).all())} "
        f"finite_qvel={bool(np.isfinite(qvel).all())} "
        f"nacon_total={nacon} nacon_per_world={nacon / max(nworld, 1):.1f}"
    )


def main() -> None:
    wp.init()
    print(f"warp_device={wp.get_preferred_device()}")
    with tempfile.TemporaryDirectory(prefix="mjwarp_granular_") as tmp:
        tmp_path = Path(tmp)
        granular_xml = tmp_path / "granular_only.xml"
        aidin_xml = tmp_path / "aidin_with_grains.xml"
        write_granular_only_xml(granular_xml)
        write_aidin_grains_xml(aidin_xml)
        run_probe("granular_only", granular_xml, nworld=8, steps=30, nconmax=2048, njmax=4096)
        run_probe("aidin_with_grains", aidin_xml, nworld=1, steps=5, nconmax=4096, njmax=8192)


if __name__ == "__main__":
    main()
