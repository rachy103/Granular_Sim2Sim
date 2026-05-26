from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm import SandMPM3D, SandMPM3DConfig
from granular_mpm.density_render import render_density_frame, write_density_sheet, write_density_video
from scripts.run_3d_blade_demo import blade_state


def run(out_dir: Path, frames_n: int, substeps: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    solver = SandMPM3D(SandMPM3DConfig(dt=8.0e-4, seed=7), device="cuda:0")
    sim_t = 0.0
    frames: list[np.ndarray] = []
    force_history: list[float] = []
    force_scale = 0.0012
    print(f"density render frames={frames_n} particles={solver.n_particles}")

    last_tool = blade_state(0.0, solver.config.dt)
    for frame_id in range(frames_n):
        raw_wrench = np.zeros(6, dtype=np.float32)
        for _ in range(substeps):
            last_tool = blade_state(sim_t, solver.config.dt)
            raw_wrench += solver.step(last_tool, substeps=1)
            sim_t += solver.config.dt
        raw_wrench /= max(1, substeps)
        display_wrench = raw_wrench * force_scale
        force_history.append(float(np.linalg.norm(display_wrench[:3])))
        pos = solver.positions()
        frames.append(render_density_frame(pos, last_tool, display_wrench, force_history, frame_id, sim_t))
        if frame_id % 10 == 0:
            print(
                f"frame={frame_id:03d} t={sim_t:.3f} "
                f"|F|={np.linalg.norm(display_wrench[:3]):.2f} zmax={pos[:,2].max():.3f}"
            )

    video = out_dir / "sand3d_density_render.mp4"
    preview = out_dir / "sand3d_density_preview.png"
    sheet = out_dir / "sand3d_density_contact_sheet.png"
    write_density_video(video, frames, fps=30)
    cv2.imwrite(preview.as_posix(), frames[-1])
    write_density_sheet(sheet, frames)
    print(f"video={video}")
    print(f"preview={preview}")
    print(f"sheet={sheet}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "3d_mpm_density_render")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=34)
    args = parser.parse_args()
    run(args.out, args.frames, args.substeps)


if __name__ == "__main__":
    main()
