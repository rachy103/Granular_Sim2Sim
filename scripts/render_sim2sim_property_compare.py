from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm import SandMPM3D, SandMPM3DConfig, ToolState3D
from granular_mpm.density_render import render_density_frame
from granular_mpm.sweep import dp_alpha_from_phi
from granular_mpm.viz import write_contact_sheet, write_video
from scripts.run_3d_blade_demo import blade_state, trajectory_from_config


DEFAULT_CONFIG = ROOT / "configs/rendering/sim2sim_property_compare.json"
TARGETS = ["rho", "phi_deg", "delta_deg", "cohesion_kpa"]


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, remaining = pre_parser.parse_known_args()
    config = load_config(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument(
        "--property-csv",
        type=Path,
        default=config_path(config, "property_csv", ROOT / "outputs/online_mohr_coulomb_bestval_quick/rollout_predictions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=config_path(config, "output_dir", ROOT / "outputs/sim2sim_property_compare"))
    parser.add_argument("--row", default=str(config.get("row", "last")))
    parser.add_argument("--device", default=str(config.get("device", "cuda:0")))
    parser.add_argument("--frames", type=int, default=int(config.get("frames", 60)))
    parser.add_argument("--fps", type=int, default=int(config.get("fps", 30)))
    parser.add_argument("--substeps-per-frame", type=int, default=int(config.get("substeps_per_frame", 34)))
    parser.add_argument("--force-scale", type=float, default=float(config.get("force_scale", 0.0012)))
    args = parser.parse_args(remaining)
    args.config = pre_args.config
    args.raw_config = config
    return args


def main() -> None:
    args = parse_args()
    config = dict(args.raw_config)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row = load_rollout_row(resolve(args.property_csv), args.row)
    gt = material_from_row(row, "target")
    est = material_from_row(row, "pred")
    mapping = dict(config.get("material_mapping", {}))
    gt_mpm = material_to_mpm_config(gt, config.get("mpm", {}), mapping)
    est_mpm = material_to_mpm_config(est, config.get("mpm", {}), mapping)
    trajectory = trajectory_from_config(config.get("trajectory", {}))
    trajectory_raw = dict(config.get("trajectory", {}))
    task = str(config.get("task", "rake"))
    artifact_stem = str(config.get("artifact_stem", "sim2sim_property_compare"))

    print("sim2sim_property_compare")
    print(f"property_csv={resolve(args.property_csv)}")
    print(f"row={args.row}")
    print(f"GT={gt}")
    print(f"EST={est}")
    print(f"GT_mpm={asdict(gt_mpm)}")
    print(f"EST_mpm={asdict(est_mpm)}")

    gt_solver = make_solver(gt_mpm, args.device)
    est_solver = make_solver(est_mpm, args.device)
    device = gt_solver.device
    if est_solver.device != device:
        raise RuntimeError("GT and EST solvers ended up on different devices")

    sim_t = 0.0
    force_history_gt: list[float] = []
    force_history_est: list[float] = []
    frames: list[np.ndarray] = []
    rows: list[dict[str, float]] = []
    last_tool = tool_state_for_task(task, 0.0, gt_mpm.dt, trajectory_raw)
    initial_gt = gt_solver.positions()
    initial_est = est_solver.positions()
    if initial_gt.shape != initial_est.shape:
        raise RuntimeError("GT and EST particle sets are not comparable")

    for frame_id in range(int(args.frames)):
        raw_gt = np.zeros(6, dtype=np.float32)
        raw_est = np.zeros(6, dtype=np.float32)
        for _ in range(int(args.substeps_per_frame)):
            last_tool = tool_state_for_task(task, sim_t, gt_mpm.dt, trajectory_raw)
            raw_gt += gt_solver.step(last_tool, substeps=1)
            raw_est += est_solver.step(last_tool, substeps=1)
            sim_t += gt_mpm.dt
        raw_gt /= max(1, int(args.substeps_per_frame))
        raw_est /= max(1, int(args.substeps_per_frame))

        display_gt = raw_gt * float(args.force_scale)
        display_est = raw_est * float(args.force_scale)
        norm_gt = float(np.linalg.norm(display_gt[:3]))
        norm_est = float(np.linalg.norm(display_est[:3]))
        force_history_gt.append(norm_gt)
        force_history_est.append(norm_est)
        pos_gt = gt_solver.positions()
        pos_est = est_solver.positions()
        divergence = np.linalg.norm(pos_gt - pos_est, axis=1)

        gt_frame = render_density_frame(pos_gt, last_tool, display_gt, force_history_gt, frame_id, sim_t)
        est_frame = render_density_frame(pos_est, last_tool, display_est, force_history_est, frame_id, sim_t)
        frames.append(
            compose_compare_frame(
                gt_frame=gt_frame,
                est_frame=est_frame,
                frame_id=frame_id,
                sim_time=sim_t,
                task=task,
                gt=gt,
                est=est,
                gt_mpm=gt_mpm,
                est_mpm=est_mpm,
                force_history_gt=force_history_gt,
                force_history_est=force_history_est,
                mean_particle_divergence=float(np.mean(divergence)),
                p90_particle_divergence=float(np.percentile(divergence, 90)),
                zmax_gt=float(pos_gt[:, 2].max()),
                zmax_est=float(pos_est[:, 2].max()),
            )
        )
        rows.append(
            {
                "frame": float(frame_id),
                "time": float(sim_t),
                "force_norm_gt": norm_gt,
                "force_norm_est": norm_est,
                "force_norm_abs_error": abs(norm_gt - norm_est),
                "force_norm_ratio_est_over_gt": norm_est / max(norm_gt, 1.0e-6),
                "mean_particle_divergence": float(np.mean(divergence)),
                "p90_particle_divergence": float(np.percentile(divergence, 90)),
                "zmax_gt": float(pos_gt[:, 2].max()),
                "zmax_est": float(pos_est[:, 2].max()),
                "zmax_abs_error": abs(float(pos_gt[:, 2].max()) - float(pos_est[:, 2].max())),
                "mean_displacement_gt": float(np.linalg.norm(pos_gt - initial_gt, axis=1).mean()),
                "mean_displacement_est": float(np.linalg.norm(pos_est - initial_est, axis=1).mean()),
            }
        )
        if frame_id % 10 == 0:
            print(
                f"frame={frame_id:03d} t={sim_t:.3f} "
                f"|F| gt/est={norm_gt:.2f}/{norm_est:.2f} "
                f"div={rows[-1]['mean_particle_divergence']:.4f}"
            )

    video_path = out_dir / f"{artifact_stem}.mp4"
    preview_path = out_dir / f"{artifact_stem}_preview.png"
    sheet_path = out_dir / f"{artifact_stem}_sheet.png"
    csv_path = out_dir / f"{artifact_stem}_metrics.csv"
    metadata_path = out_dir / f"{artifact_stem}_metadata.json"

    write_video(video_path, frames, fps=int(args.fps))
    cv2.imwrite(preview_path.as_posix(), frames[len(frames) // 2])
    write_contact_sheet(sheet_path, frames, count=5)
    write_metrics_csv(csv_path, rows)
    metadata = {
        "script": Path(__file__).resolve().as_posix(),
        "config": resolve(args.config).as_posix(),
        "property_csv": resolve(args.property_csv).as_posix(),
        "row": args.row,
        "task": task,
        "device": device,
        "frames": int(args.frames),
        "fps": int(args.fps),
        "substeps_per_frame": int(args.substeps_per_frame),
        "trajectory": trajectory_raw,
        "material_mapping": mapping,
        "gt": gt,
        "est": est,
        "gt_mpm": asdict(gt_mpm),
        "est_mpm": asdict(est_mpm),
        "final_metrics": rows[-1],
        "mean_force_abs_error": float(np.mean([r["force_norm_abs_error"] for r in rows])),
        "mean_particle_divergence": float(np.mean([r["mean_particle_divergence"] for r in rows])),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"video={video_path}")
    print(f"preview={preview_path}")
    print(f"sheet={sheet_path}")
    print(f"metrics={csv_path}")
    print(f"metadata={metadata_path}")


def load_config(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        return {}
    with resolved.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return dict(raw.get("sim2sim_property_compare", raw))


def config_path(config: dict[str, Any], key: str, default: Path) -> Path:
    value = config.get(key)
    return default if value is None else Path(str(value))


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def tool_state_for_task(task: str, time: float, dt: float, trajectory: dict[str, Any] | None = None) -> ToolState3D:
    if task in {"bulldozing_wedge", "wedge", "bulldoze"}:
        return bulldozing_wedge_state(time, dt, trajectory or {})
    return blade_state(time, dt, trajectory_from_config(trajectory or {}))


def smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def bulldozing_wedge_pose(time: float, params: dict[str, Any]) -> tuple[np.ndarray, float, np.ndarray]:
    t = (time + float(params.get("time_offset", 0.0))) * float(params.get("speed_scale", 1.0))
    insert_duration = float(params.get("insert_duration", 0.24))
    push_duration = float(params.get("push_duration", 1.18))
    x_start = float(params.get("x_start", 0.225))
    x_end = float(params.get("x_end", 0.775))
    y = float(params.get("y", 0.280))
    z_high = float(params.get("z_high", 0.330))
    z_work = float(params.get("z_work", 0.150))
    angle = float(params.get("angle", math.pi / 2.0)) + float(params.get("angle_offset", 0.0))

    if t < insert_duration:
        u = smoothstep(t / max(insert_duration, 1.0e-6))
        center = np.asarray([x_start, y, z_high + (z_work - z_high) * u], dtype=np.float32)
    elif t < insert_duration + push_duration:
        u = (t - insert_duration) / max(push_duration, 1.0e-6)
        edge = 0.06
        if u < edge:
            u_motion = edge * smoothstep(u / edge)
        elif u > 1.0 - edge:
            u_motion = 1.0 - edge + edge * smoothstep((u - (1.0 - edge)) / edge)
        else:
            u_motion = u
        center = np.asarray([x_start + (x_end - x_start) * u_motion, y, z_work], dtype=np.float32)
    else:
        center = np.asarray([x_end, y, z_work], dtype=np.float32)

    half = np.asarray(
        [
            float(params.get("blade_half_x", 0.105)),
            float(params.get("blade_half_y", 0.145)),
            float(params.get("blade_half_z", 0.012)),
        ],
        dtype=np.float32,
    )
    return center, angle, half


def bulldozing_wedge_state(time: float, dt: float, params: dict[str, Any]) -> ToolState3D:
    center0, angle0, half = bulldozing_wedge_pose(time, params)
    center1, angle1, _half1 = bulldozing_wedge_pose(time + dt, params)
    return ToolState3D(
        center=center0,
        velocity=((center1 - center0) / max(dt, 1.0e-6)).astype(np.float32),
        angle=float(angle0),
        angular_velocity=float((angle1 - angle0) / max(dt, 1.0e-6)),
        half=half,
    )


def load_rollout_row(path: Path, row_spec: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    if row_spec == "last":
        return rows[-1]
    index = int(row_spec)
    return rows[index]


def material_from_row(row: dict[str, str], prefix: str) -> dict[str, float]:
    material = {}
    for name in TARGETS:
        key = f"{prefix}_{name}"
        if key not in row:
            raise KeyError(f"Missing {key} in rollout row")
        material[name] = float(row[key])
    return material


def material_to_mpm_config(
    material: dict[str, float],
    base: dict[str, Any] | None,
    mapping: dict[str, Any],
) -> SandMPM3DConfig:
    raw = dict(base or {})
    rho = float(material["rho"])
    phi = float(material["phi_deg"])
    delta = float(material["delta_deg"])
    cohesion_kpa = float(material["cohesion_kpa"])
    rho_reference = float(mapping.get("rho_reference", 1550.0))
    rho_mass = rho / max(rho_reference, 1.0e-6)
    rho_mass = float(np.clip(rho_mass, float(mapping.get("rho_mass_min", 0.65)), float(mapping.get("rho_mass_max", 1.35))))
    tool_mu = math.tan(math.radians(delta)) * float(mapping.get("delta_tool_mu_scale", 2.2))
    tool_mu = float(np.clip(tool_mu, float(mapping.get("tool_mu_min", 0.18)), float(mapping.get("tool_mu_max", 0.92))))
    raw.update(
        {
            "p_mass": rho_mass,
            "dp_alpha": dp_alpha_from_phi(phi, float(mapping.get("dp_alpha_scale", 1.0))),
            "cohesion": cohesion_kpa * float(mapping.get("cohesion_kpa_to_mpm", 0.004)),
            "tool_mu": tool_mu,
            "young": float(mapping.get("young_base", 1800.0)) + cohesion_kpa * float(mapping.get("young_per_kpa", 45.0)),
        }
    )
    allowed = set(SandMPM3DConfig.__dataclass_fields__.keys())
    return SandMPM3DConfig(**{key: value for key, value in raw.items() if key in allowed})


def make_solver(config: SandMPM3DConfig, device: str) -> SandMPM3D:
    try:
        return SandMPM3D(config, device=device)
    except Exception as exc:
        if device == "cpu":
            raise
        print(f"falling_back_to_cpu_after_device_error={device}: {exc}")
        return SandMPM3D(config, device="cpu")


def compose_compare_frame(
    gt_frame: np.ndarray,
    est_frame: np.ndarray,
    frame_id: int,
    sim_time: float,
    task: str,
    gt: dict[str, float],
    est: dict[str, float],
    gt_mpm: SandMPM3DConfig,
    est_mpm: SandMPM3DConfig,
    force_history_gt: list[float],
    force_history_est: list[float],
    mean_particle_divergence: float,
    p90_particle_divergence: float,
    zmax_gt: float,
    zmax_est: float,
) -> np.ndarray:
    canvas = np.full((1080, 1920, 3), (18, 20, 21), dtype=np.uint8)
    cv2.putText(
        canvas,
        f"Sim2Sim material check: GT granular vs estimated granular on the same {task.replace('_', ' ')} task",
        (30, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (238, 242, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"frame {frame_id:03d}   t={sim_time:5.3f}s",
        (30, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (210, 216, 216),
        1,
        cv2.LINE_AA,
    )
    left = cv2.resize(gt_frame, (900, 506), interpolation=cv2.INTER_AREA)
    right = cv2.resize(est_frame, (900, 506), interpolation=cv2.INTER_AREA)
    canvas[104:610, 30:930] = left
    canvas[104:610, 990:1890] = right
    draw_label(canvas, "GT material", (30, 96), (80, 180, 255))
    draw_label(canvas, "Estimated material", (990, 96), (110, 220, 150))
    cv2.rectangle(canvas, (30, 104), (930, 610), (66, 70, 70), 1)
    cv2.rectangle(canvas, (990, 104), (1890, 610), (66, 70, 70), 1)
    draw_force_comparison(canvas, force_history_gt, force_history_est, (42, 670, 850, 250))
    draw_property_table(canvas, gt, est, gt_mpm, est_mpm, (960, 650, 900, 360))
    cv2.putText(
        canvas,
        f"particle divergence mean={mean_particle_divergence:.4f}m  p90={p90_particle_divergence:.4f}m   "
        f"zmax GT/EST={zmax_gt:.3f}/{zmax_est:.3f}",
        (42, 1030),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 224, 222),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_label(canvas: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    cv2.rectangle(canvas, (x, y - 30), (x + 285, y - 4), (26, 29, 30), -1)
    cv2.putText(canvas, text, (x + 12, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def draw_force_comparison(canvas: np.ndarray, gt: list[float], est: list[float], rect: tuple[int, int, int, int]) -> None:
    x0, y0, w, h = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (62, 66, 66), 1)
    cv2.putText(canvas, "reaction norm comparison", (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (226, 230, 228), 1, cv2.LINE_AA)
    draw_series(canvas, gt, rect, (80, 180, 255))
    draw_series(canvas, est, rect, (110, 220, 150))
    cv2.putText(canvas, "GT", (x0 + 20, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 180, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "EST", (x0 + 86, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 220, 150), 2, cv2.LINE_AA)
    if gt and est:
        err = abs(float(gt[-1]) - float(est[-1]))
        cv2.putText(
            canvas,
            f"current |F| error={err:.2f}",
            (x0 + 20, y0 + h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 232, 210),
            1,
            cv2.LINE_AA,
        )


def draw_series(canvas: np.ndarray, values: list[float], rect: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    if len(values) < 2:
        return
    x0, y0, w, h = rect
    arr = np.asarray(values[-160:], dtype=np.float32)
    peak = max(1.0, float(np.percentile(arr, 96)))
    pts = []
    for idx, value in enumerate(arr):
        x = x0 + 18 + int(idx * (w - 36) / max(1, arr.shape[0] - 1))
        y = y0 + h - 28 - int(np.clip(float(value) / peak, 0.0, 1.0) * (h - 58))
        pts.append((x, y))
    cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


def draw_property_table(
    canvas: np.ndarray,
    gt: dict[str, float],
    est: dict[str, float],
    gt_mpm: SandMPM3DConfig,
    est_mpm: SandMPM3DConfig,
    rect: tuple[int, int, int, int],
) -> None:
    x0, y0, w, h = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (62, 66, 66), 1)
    cv2.putText(canvas, "material posterior used as simulator parameters", (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (226, 230, 228), 1, cv2.LINE_AA)
    y = y0 + 38
    for name in TARGETS:
        err = abs(gt[name] - est[name])
        cv2.putText(
            canvas,
            f"{name:<13} GT={gt[name]:8.3f}   EST={est[name]:8.3f}   abs err={err:7.3f}",
            (x0 + 22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 234, 232),
            1,
            cv2.LINE_AA,
        )
        y += 36
    y += 12
    for label, gval, eval_ in [
        ("p_mass", gt_mpm.p_mass, est_mpm.p_mass),
        ("dp_alpha", gt_mpm.dp_alpha, est_mpm.dp_alpha),
        ("cohesion", gt_mpm.cohesion, est_mpm.cohesion),
        ("tool_mu", gt_mpm.tool_mu, est_mpm.tool_mu),
        ("young", gt_mpm.young, est_mpm.young),
    ]:
        cv2.putText(
            canvas,
            f"{label:<13} GT={gval:8.4g}   EST={eval_:8.4g}",
            (x0 + 22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (202, 208, 206),
            1,
            cv2.LINE_AA,
        )
        y += 30


def write_metrics_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
