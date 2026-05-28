from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm import SandMPM3D, SandMPM3DConfig, ToolState3D
from granular_mpm.viz import render_frame_3d, write_contact_sheet, write_video


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def default_trajectory() -> dict[str, float]:
    return {
        "speed_scale": 1.0,
        "time_offset": 0.0,
        "x_offset": 0.0,
        "y_offset": 0.0,
        "z_offset": 0.0,
        "depth_scale": 1.0,
        "drag_distance_scale": 1.0,
        "angle_offset": 0.0,
        "angle_scale": 1.0,
        "blade_half_x_scale": 1.0,
        "blade_half_y_scale": 1.0,
        "blade_half_z_scale": 1.0,
    }


def trajectory_from_config(raw: dict | None) -> dict[str, float]:
    params = default_trajectory()
    if raw:
        for key, value in raw.items():
            if key in params:
                params[key] = float(value)
    return params


def blade_pose(time: float, trajectory: dict[str, float] | None = None) -> tuple[np.ndarray, float]:
    params = trajectory or default_trajectory()
    time = (time + params["time_offset"]) * params["speed_scale"]
    if time < 0.22:
        u = smoothstep(time / 0.22)
        center = np.array([0.25 + 0.10 * u, 0.280, 0.405 - 0.105 * u], dtype=np.float32)
        angle = -0.78
    elif time < 0.52:
        u = smoothstep((time - 0.22) / 0.30)
        center = np.array([0.35 + 0.055 * u, 0.280, 0.300 - 0.080 * u], dtype=np.float32)
        angle = -0.78 + 0.36 * u
    elif time < 1.42:
        u = smoothstep((time - 0.52) / 0.90)
        center = np.array(
            [
                0.405 + 0.340 * u,
                0.280 + 0.035 * np.sin(np.pi * u),
                0.220 - 0.006 * np.sin(np.pi * u),
            ],
            dtype=np.float32,
        )
        angle = -0.42 + 0.11 * u
    else:
        u = smoothstep(min((time - 1.42) / 0.36, 1.0))
        center = np.array([0.745 + 0.035 * u, 0.280, 0.220 + 0.150 * u], dtype=np.float32)
        angle = -0.31
    anchor = np.array([0.25, 0.280, 0.405], dtype=np.float32)
    center[0] = anchor[0] + (center[0] - anchor[0]) * params["drag_distance_scale"] + params["x_offset"]
    center[1] = center[1] + params["y_offset"]
    center[2] = anchor[2] + (center[2] - anchor[2]) * params["depth_scale"] + params["z_offset"]
    angle = angle * params["angle_scale"] + params["angle_offset"]
    return center, float(angle)


def blade_state(time: float, dt: float, trajectory: dict[str, float] | None = None) -> ToolState3D:
    params = trajectory or default_trajectory()
    center0, angle0 = blade_pose(time, params)
    center1, angle1 = blade_pose(time + dt, params)
    half = np.array(
        [
            0.082 * params["blade_half_x_scale"],
            0.122 * params["blade_half_y_scale"],
            0.015 * params["blade_half_z_scale"],
        ],
        dtype=np.float32,
    )
    return ToolState3D(
        center=center0,
        velocity=((center1 - center0) / max(dt, 1.0e-6)).astype(np.float32),
        angle=angle0,
        angular_velocity=float((angle1 - angle0) / max(dt, 1.0e-6)),
        half=half,
    )


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_mpm_config(raw: dict) -> SandMPM3DConfig:
    allowed = set(SandMPM3DConfig.__dataclass_fields__.keys())
    kwargs = {k: v for k, v in raw.items() if k in allowed}
    return SandMPM3DConfig(**kwargs)


def run(config_path: Path) -> None:
    cfg = load_config(config_path)
    out_dir = ROOT / cfg.get("output_dir", "outputs/3d_mpm_blade")
    out_dir.mkdir(parents=True, exist_ok=True)

    mpm_cfg = build_mpm_config(cfg.get("mpm", {}))
    solver = SandMPM3D(mpm_cfg, device=cfg.get("device", "cuda:0"))
    frames_n = int(cfg.get("frames", 60))
    substeps = int(cfg.get("substeps_per_frame", 36))
    fps = int(cfg.get("fps", 30))
    display_force_scale = float(cfg.get("display_force_scale", 0.0012))
    trajectory = trajectory_from_config(cfg.get("trajectory", {}))

    print(
        f"3D MPM demo particles={solver.n_particles} grid=72x40x48 "
        f"dt={mpm_cfg.dt} substeps/frame={substeps} frames={frames_n}"
    )

    frames: list[np.ndarray] = []
    force_history: list[float] = []
    rows: list[dict[str, float]] = []
    sim_t = 0.0
    last_tool = blade_state(0.0, mpm_cfg.dt, trajectory)

    for frame_id in range(frames_n):
        raw_wrench = np.zeros(6, dtype=np.float32)
        for _ in range(substeps):
            tool = blade_state(sim_t, mpm_cfg.dt, trajectory)
            raw_wrench += solver.step(tool, substeps=1)
            sim_t += mpm_cfg.dt
            last_tool = tool
        raw_wrench /= max(1, substeps)
        display_wrench = raw_wrench * display_force_scale
        force_history.append(float(np.linalg.norm(display_wrench[:3])))
        pos = solver.positions()
        frame = render_frame_3d(pos, last_tool, display_wrench, force_history, frame_id, sim_t)
        frames.append(frame)

        rows.append(
            {
                "frame": frame_id,
                "time": sim_t,
                "tool_x": float(last_tool.center[0]),
                "tool_y": float(last_tool.center[1]),
                "tool_z": float(last_tool.center[2]),
                "tool_angle": float(last_tool.angle),
                "raw_fx": float(raw_wrench[0]),
                "raw_fy": float(raw_wrench[1]),
                "raw_fz": float(raw_wrench[2]),
                "raw_tx": float(raw_wrench[3]),
                "raw_ty": float(raw_wrench[4]),
                "raw_tz": float(raw_wrench[5]),
                "display_force_norm": float(np.linalg.norm(display_wrench[:3])),
                "z_min": float(pos[:, 2].min()),
                "z_max": float(pos[:, 2].max()),
                "trajectory_speed_scale": float(trajectory["speed_scale"]),
                "trajectory_depth_scale": float(trajectory["depth_scale"]),
                "trajectory_angle_offset": float(trajectory["angle_offset"]),
                "trajectory_drag_distance_scale": float(trajectory["drag_distance_scale"]),
            }
        )
        if frame_id % 10 == 0:
            print(
                f"frame={frame_id:03d} t={sim_t:.3f} "
                f"|F|={np.linalg.norm(display_wrench[:3]):.2f} "
                f"z=[{pos[:, 2].min():.3f},{pos[:, 2].max():.3f}]"
            )

    video_path = out_dir / "sand3d_blade_interaction.mp4"
    preview_path = out_dir / "sand3d_blade_preview.png"
    sheet_path = out_dir / "sand3d_blade_contact_sheet.png"
    csv_path = out_dir / "wrench_log.csv"
    npz_path = out_dir / "final_state_and_wrench_log.npz"
    config_copy = out_dir / "resolved_config.json"

    write_video(video_path, frames, fps=fps)
    cv2.imwrite(preview_path.as_posix(), frames[-1])
    write_contact_sheet(sheet_path, frames)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        npz_path,
        final_positions=solver.positions(),
        wrench_log=np.asarray([[row[k] for k in rows[0].keys()] for row in rows], dtype=np.float32),
        wrench_log_columns=np.asarray(list(rows[0].keys())),
    )
    with config_copy.open("w", encoding="utf-8") as f:
        json.dump({"mpm": asdict(mpm_cfg), "trajectory": trajectory, **{k: v for k, v in cfg.items() if k not in {"mpm", "trajectory"}}}, f, indent=2)

    print(f"video={video_path}")
    print(f"preview={preview_path}")
    print(f"sheet={sheet_path}")
    print(f"wrench_log={csv_path}")
    print(f"state={npz_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "sand3d_blade_demo.json")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
