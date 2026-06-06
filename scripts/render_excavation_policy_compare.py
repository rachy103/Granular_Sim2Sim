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


DEFAULT_CONFIG = ROOT / "configs/rendering/excavation_policy_compare.json"
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
    parser.add_argument("--output-dir", type=Path, default=config_path(config, "output_dir", ROOT / "outputs/excavation_policy_compare"))
    parser.add_argument("--row", default=str(config.get("row", "last")))
    parser.add_argument("--device", default=str(config.get("device", "cuda:0")))
    parser.add_argument("--frames", type=int, default=int(config.get("frames", 84)))
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
    stem = str(config.get("artifact_stem", "excavation_policy_compare"))

    row = load_rollout_row(resolve(args.property_csv), str(args.row))
    gt_material = material_from_row(row, "target")
    pred_material = material_from_row(row, "pred")
    mapping = dict(config.get("material_mapping", {}))
    mpm_cfg = material_to_mpm_config(gt_material, dict(config.get("mpm", {})), mapping)
    baseline_policy = dict(config.get("baseline_policy", {}))
    model_policy = property_aware_policy(pred_material, dict(config.get("model_policy", {})))

    print("excavation_policy_compare")
    print(f"property_csv={resolve(args.property_csv)}")
    print(f"row={args.row}")
    print(f"gt_material={gt_material}")
    print(f"pred_material={pred_material}")
    print(f"mpm_cfg={asdict(mpm_cfg)}")
    print(f"baseline_policy={baseline_policy}")
    print(f"model_policy={model_policy}")

    fixed_solver = make_solver(mpm_cfg, args.device)
    model_solver = make_solver(mpm_cfg, args.device)
    device = fixed_solver.device
    initial_fixed = fixed_solver.positions()
    initial_model = model_solver.positions()
    if initial_fixed.shape != initial_model.shape:
        raise RuntimeError("Fixed and model-aware solvers have different particle layouts")

    sim_t = 0.0
    fixed_history: list[float] = []
    model_history: list[float] = []
    fixed_work = 0.0
    model_work = 0.0
    frames: list[np.ndarray] = []
    metrics_rows: list[dict[str, float]] = []
    force_limit = float(config.get("force_limit_display", 2900.0))

    for frame_id in range(int(args.frames)):
        raw_fixed = np.zeros(6, dtype=np.float32)
        raw_model = np.zeros(6, dtype=np.float32)
        path_fixed = 0.0
        path_model = 0.0
        fixed_tool = excavation_state(sim_t, mpm_cfg.dt, baseline_policy)
        model_tool = excavation_state(sim_t, mpm_cfg.dt, model_policy)
        for _ in range(int(args.substeps_per_frame)):
            fixed_tool = excavation_state(sim_t, mpm_cfg.dt, baseline_policy)
            model_tool = excavation_state(sim_t, mpm_cfg.dt, model_policy)
            raw_fixed += fixed_solver.step(fixed_tool, substeps=1)
            raw_model += model_solver.step(model_tool, substeps=1)
            path_fixed += float(np.linalg.norm(fixed_tool.velocity[:3])) * mpm_cfg.dt
            path_model += float(np.linalg.norm(model_tool.velocity[:3])) * mpm_cfg.dt
            sim_t += mpm_cfg.dt
        raw_fixed /= max(1, int(args.substeps_per_frame))
        raw_model /= max(1, int(args.substeps_per_frame))
        display_fixed = raw_fixed * float(args.force_scale)
        display_model = raw_model * float(args.force_scale)
        f_fixed = float(np.linalg.norm(display_fixed[:3]))
        f_model = float(np.linalg.norm(display_model[:3]))
        fixed_work += f_fixed * path_fixed
        model_work += f_model * path_model
        fixed_history.append(f_fixed)
        model_history.append(f_model)
        pos_fixed = fixed_solver.positions()
        pos_model = model_solver.positions()
        fixed_metrics = excavation_metrics(pos_fixed, initial_fixed, mpm_cfg.p_mass, fixed_history, fixed_work, force_limit)
        model_metrics = excavation_metrics(pos_model, initial_model, mpm_cfg.p_mass, model_history, model_work, force_limit)
        frame = compose_frame(
            fixed_frame=render_density_frame(pos_fixed, fixed_tool, display_fixed, fixed_history, frame_id, sim_t),
            model_frame=render_density_frame(pos_model, model_tool, display_model, model_history, frame_id, sim_t),
            frame_id=frame_id,
            sim_time=sim_t,
            gt_material=gt_material,
            pred_material=pred_material,
            baseline_policy=baseline_policy,
            model_policy=model_policy,
            fixed_metrics=fixed_metrics,
            model_metrics=model_metrics,
            fixed_history=fixed_history,
            model_history=model_history,
            force_limit=force_limit,
        )
        frames.append(frame)
        metrics_rows.append(
            {
                "frame": float(frame_id),
                "time": float(sim_t),
                **{f"fixed_{k}": float(v) for k, v in fixed_metrics.items()},
                **{f"model_{k}": float(v) for k, v in model_metrics.items()},
                "reward_delta_model_minus_fixed": float(model_metrics["reward"] - fixed_metrics["reward"]),
                "peak_force_reduction": float(fixed_metrics["peak_force"] - model_metrics["peak_force"]),
                "work_reduction": float(fixed_metrics["work"] - model_metrics["work"]),
            }
        )
        if frame_id % 10 == 0:
            print(
                f"frame={frame_id:03d} t={sim_t:.3f} "
                f"moved fixed/model={fixed_metrics['moved_mass']:.1f}/{model_metrics['moved_mass']:.1f} "
                f"peakF={fixed_metrics['peak_force']:.1f}/{model_metrics['peak_force']:.1f} "
                f"reward={fixed_metrics['reward']:.2f}/{model_metrics['reward']:.2f}"
            )

    video_path = out_dir / f"{stem}.mp4"
    preview_path = out_dir / f"{stem}_preview.png"
    sheet_path = out_dir / f"{stem}_sheet.png"
    csv_path = out_dir / f"{stem}_metrics.csv"
    metadata_path = out_dir / f"{stem}_metadata.json"
    write_video(video_path, frames, fps=int(args.fps))
    cv2.imwrite(preview_path.as_posix(), frames[len(frames) // 2])
    write_contact_sheet(sheet_path, frames, count=5)
    write_metrics_csv(csv_path, metrics_rows)
    metadata = {
        "script": Path(__file__).resolve().as_posix(),
        "config": resolve(args.config).as_posix(),
        "property_csv": resolve(args.property_csv).as_posix(),
        "row": args.row,
        "device": device,
        "frames": int(args.frames),
        "fps": int(args.fps),
        "substeps_per_frame": int(args.substeps_per_frame),
        "gt_material": gt_material,
        "pred_material": pred_material,
        "mpm_config": asdict(mpm_cfg),
        "material_mapping": mapping,
        "baseline_policy": baseline_policy,
        "model_policy": model_policy,
        "force_limit_display": force_limit,
        "final_metrics": metrics_rows[-1],
        "mean_reward_delta_model_minus_fixed": float(np.mean([r["reward_delta_model_minus_fixed"] for r in metrics_rows])),
        "mean_peak_force_reduction": float(np.mean([r["peak_force_reduction"] for r in metrics_rows])),
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
    return dict(raw.get("excavation_policy_compare", raw))


def config_path(config: dict[str, Any], key: str, default: Path) -> Path:
    value = config.get(key)
    return default if value is None else Path(str(value))


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_rollout_row(path: Path, row_spec: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    if row_spec == "last":
        return rows[-1]
    return rows[int(row_spec)]


def material_from_row(row: dict[str, str], prefix: str) -> dict[str, float]:
    return {name: float(row[f"{prefix}_{name}"]) for name in TARGETS}


def material_to_mpm_config(material: dict[str, float], base: dict[str, Any], mapping: dict[str, Any]) -> SandMPM3DConfig:
    raw = dict(base or {})
    rho = float(material["rho"])
    phi = float(material["phi_deg"])
    delta = float(material["delta_deg"])
    cohesion_kpa = float(material["cohesion_kpa"])
    rho_reference = float(mapping.get("rho_reference", 1550.0))
    p_mass = float(np.clip(rho / max(rho_reference, 1.0e-6), float(mapping.get("rho_mass_min", 0.65)), float(mapping.get("rho_mass_max", 1.35))))
    tool_mu = math.tan(math.radians(delta)) * float(mapping.get("delta_tool_mu_scale", 2.2))
    tool_mu = float(np.clip(tool_mu, float(mapping.get("tool_mu_min", 0.18)), float(mapping.get("tool_mu_max", 0.92))))
    raw.update(
        {
            "p_mass": p_mass,
            "dp_alpha": dp_alpha_from_phi(phi, float(mapping.get("dp_alpha_scale", 1.0))),
            "cohesion": cohesion_kpa * float(mapping.get("cohesion_kpa_to_mpm", 0.004)),
            "tool_mu": tool_mu,
            "young": float(mapping.get("young_base", 1800.0)) + cohesion_kpa * float(mapping.get("young_per_kpa", 45.0)),
        }
    )
    allowed = set(SandMPM3DConfig.__dataclass_fields__.keys())
    return SandMPM3DConfig(**{key: value for key, value in raw.items() if key in allowed})


def property_aware_policy(pred: dict[str, float], raw: dict[str, Any]) -> dict[str, Any]:
    policy = dict(raw)
    rho_n = normalize(float(pred["rho"]), 1150.0, 1900.0)
    phi_n = normalize(float(pred["phi_deg"]), 24.0, 46.0)
    delta_n = normalize(float(pred["delta_deg"]), 8.0, 34.0)
    cohesion_n = normalize(float(pred["cohesion_kpa"]), 0.0, 16.0)
    strength = float(np.clip(0.20 * rho_n + 0.28 * phi_n + 0.22 * delta_n + 0.30 * cohesion_n, 0.0, 1.0))
    z_deep = float(policy.pop("z_work_deep", 0.145))
    z_shallow = float(policy.pop("z_work_shallow", 0.178))
    base_insert = float(policy.pop("insert_duration_base", 0.24))
    base_push = float(policy.pop("push_duration_base", 1.18))
    x_start = float(policy.get("x_start", 0.225))
    x_end = float(policy.get("x_end", 0.790))
    x_end_reduction = float(policy.pop("x_end_strength_reduction", 0.25))
    policy["z_work"] = z_deep + (z_shallow - z_deep) * strength
    policy["insert_duration"] = base_insert * (1.0 + 0.25 * strength)
    policy["push_duration"] = base_push * (1.0 + 0.10 * strength)
    policy["x_end"] = x_start + (x_end - x_start) * (1.0 - x_end_reduction * strength)
    policy["estimated_strength_index"] = strength
    return policy


def normalize(value: float, lo: float, hi: float) -> float:
    return float(np.clip((value - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0))


def smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def excavation_pose(time: float, policy: dict[str, Any]) -> tuple[np.ndarray, float, np.ndarray]:
    insert_duration = float(policy.get("insert_duration", 0.24))
    push_duration = float(policy.get("push_duration", 1.18))
    lift_duration = float(policy.get("lift_duration", 0.34))
    x_start = float(policy.get("x_start", 0.225))
    x_end = float(policy.get("x_end", 0.790))
    y = float(policy.get("y", 0.280))
    z_high = float(policy.get("z_high", 0.335))
    z_work = float(policy.get("z_work", 0.155))
    z_lift = float(policy.get("z_lift", 0.305))
    angle = float(policy.get("angle", math.pi / 2.0))
    if time < insert_duration:
        u = smoothstep(time / max(insert_duration, 1.0e-6))
        center = np.asarray([x_start, y, z_high + (z_work - z_high) * u], dtype=np.float32)
    elif time < insert_duration + push_duration:
        u = (time - insert_duration) / max(push_duration, 1.0e-6)
        center = np.asarray([x_start + (x_end - x_start) * smoothstep(u), y, z_work], dtype=np.float32)
    elif time < insert_duration + push_duration + lift_duration:
        u = smoothstep((time - insert_duration - push_duration) / max(lift_duration, 1.0e-6))
        center = np.asarray([x_end, y, z_work + (z_lift - z_work) * u], dtype=np.float32)
    else:
        center = np.asarray([x_end, y, z_lift], dtype=np.float32)
    half = np.asarray(
        [
            float(policy.get("blade_half_x", 0.105)),
            float(policy.get("blade_half_y", 0.145)),
            float(policy.get("blade_half_z", 0.012)),
        ],
        dtype=np.float32,
    )
    return center, angle, half


def excavation_state(time: float, dt: float, policy: dict[str, Any]) -> ToolState3D:
    center0, angle0, half = excavation_pose(time, policy)
    center1, angle1, _ = excavation_pose(time + dt, policy)
    return ToolState3D(
        center=center0,
        velocity=((center1 - center0) / max(dt, 1.0e-6)).astype(np.float32),
        angle=angle0,
        angular_velocity=float((angle1 - angle0) / max(dt, 1.0e-6)),
        half=half,
    )


def make_solver(config: SandMPM3DConfig, device: str) -> SandMPM3D:
    try:
        return SandMPM3D(config, device=device)
    except Exception as exc:
        if device == "cpu":
            raise
        print(f"falling_back_to_cpu_after_device_error={device}: {exc}")
        return SandMPM3D(config, device="cpu")


def excavation_metrics(
    pos: np.ndarray,
    initial_pos: np.ndarray,
    p_mass: float,
    force_history: list[float],
    work: float,
    force_limit: float,
) -> dict[str, float]:
    x0 = initial_pos[:, 0]
    x = pos[:, 0]
    moved_forward = np.maximum(x - x0, 0.0)
    moved_mass = float(np.sum(moved_forward > 0.035) * p_mass)
    target_mass = float(np.sum(x > 0.700) * p_mass)
    transport = float(np.sum(moved_forward) * p_mass)
    peak_force = float(max(force_history) if force_history else 0.0)
    current_force = float(force_history[-1] if force_history else 0.0)
    violation = float(max(0.0, peak_force - force_limit))
    efficiency = transport / (work + 200.0)
    reward = 0.075 * moved_mass + 70.0 * efficiency + 0.006 * target_mass - 0.00012 * work - 0.09 * violation
    return {
        "moved_mass": moved_mass,
        "target_zone_mass": target_mass,
        "forward_transport": transport,
        "peak_force": peak_force,
        "current_force": current_force,
        "work": float(work),
        "force_violation": violation,
        "efficiency": float(efficiency),
        "reward": float(reward),
    }


def compose_frame(
    fixed_frame: np.ndarray,
    model_frame: np.ndarray,
    frame_id: int,
    sim_time: float,
    gt_material: dict[str, float],
    pred_material: dict[str, float],
    baseline_policy: dict[str, Any],
    model_policy: dict[str, Any],
    fixed_metrics: dict[str, float],
    model_metrics: dict[str, float],
    fixed_history: list[float],
    model_history: list[float],
    force_limit: float,
) -> np.ndarray:
    canvas = np.full((1080, 1920, 3), (18, 20, 21), dtype=np.uint8)
    cv2.putText(
        canvas,
        "Excavation policy comparison: fixed no-model plan vs property-aware model plan",
        (30, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (238, 242, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, f"frame {frame_id:03d}   t={sim_time:5.3f}s", (30, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 216, 216), 1, cv2.LINE_AA)
    left = cv2.resize(fixed_frame, (900, 506), interpolation=cv2.INTER_AREA)
    right = cv2.resize(model_frame, (900, 506), interpolation=cv2.INTER_AREA)
    canvas[104:610, 30:930] = left
    canvas[104:610, 990:1890] = right
    draw_label(canvas, "No model: fixed nominal excavation", (30, 96), (80, 180, 255))
    draw_label(canvas, "With model: property-aware excavation", (990, 96), (110, 220, 150))
    cv2.rectangle(canvas, (30, 104), (930, 610), (66, 70, 70), 1)
    cv2.rectangle(canvas, (990, 104), (1890, 610), (66, 70, 70), 1)
    draw_force_plot(canvas, fixed_history, model_history, force_limit, (42, 670, 850, 250))
    draw_score_panel(canvas, fixed_metrics, model_metrics, baseline_policy, model_policy, (960, 650, 900, 360))
    draw_material_line(canvas, gt_material, pred_material, (42, 1030))
    return canvas


def draw_label(canvas: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    cv2.rectangle(canvas, (x, y - 30), (x + 420, y - 4), (26, 29, 30), -1)
    cv2.putText(canvas, text, (x + 12, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def draw_force_plot(canvas: np.ndarray, fixed: list[float], model: list[float], limit: float, rect: tuple[int, int, int, int]) -> None:
    x0, y0, w, h = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (62, 66, 66), 1)
    cv2.putText(canvas, "reaction norm and safety limit", (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (226, 230, 228), 1, cv2.LINE_AA)
    all_values = np.asarray((fixed + model) or [1.0], dtype=np.float32)
    peak = max(1.0, float(np.percentile(all_values, 96)), limit * 1.05)
    limit_y = y0 + h - 28 - int(np.clip(limit / peak, 0.0, 1.0) * (h - 58))
    cv2.line(canvas, (x0 + 18, limit_y), (x0 + w - 18, limit_y), (120, 120, 130), 1, cv2.LINE_AA)
    cv2.putText(canvas, "limit", (x0 + w - 78, limit_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 150, 160), 1, cv2.LINE_AA)
    draw_series(canvas, fixed, rect, peak, (80, 180, 255))
    draw_series(canvas, model, rect, peak, (110, 220, 150))
    cv2.putText(canvas, "fixed", (x0 + 20, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 180, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "model", (x0 + 100, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 220, 150), 2, cv2.LINE_AA)


def draw_series(canvas: np.ndarray, values: list[float], rect: tuple[int, int, int, int], peak: float, color: tuple[int, int, int]) -> None:
    if len(values) < 2:
        return
    x0, y0, w, h = rect
    arr = np.asarray(values[-180:], dtype=np.float32)
    pts = []
    for idx, value in enumerate(arr):
        x = x0 + 18 + int(idx * (w - 36) / max(1, arr.shape[0] - 1))
        y = y0 + h - 28 - int(np.clip(float(value) / peak, 0.0, 1.0) * (h - 58))
        pts.append((x, y))
    cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


def draw_score_panel(
    canvas: np.ndarray,
    fixed: dict[str, float],
    model: dict[str, float],
    baseline_policy: dict[str, Any],
    model_policy: dict[str, Any],
    rect: tuple[int, int, int, int],
) -> None:
    x0, y0, w, h = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (62, 66, 66), 1)
    cv2.putText(canvas, "excavation metrics", (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (226, 230, 228), 1, cv2.LINE_AA)
    rows = [
        ("z_work", float(baseline_policy.get("z_work", 0.0)), float(model_policy.get("z_work", 0.0)), "m"),
        ("push_duration", float(baseline_policy.get("push_duration", 0.0)), float(model_policy.get("push_duration", 0.0)), "s"),
        ("moved_mass", fixed["moved_mass"], model["moved_mass"], ""),
        ("target_zone_mass", fixed["target_zone_mass"], model["target_zone_mass"], ""),
        ("peak_force", fixed["peak_force"], model["peak_force"], ""),
        ("work", fixed["work"], model["work"], ""),
        ("efficiency", fixed["efficiency"], model["efficiency"], ""),
        ("reward", fixed["reward"], model["reward"], ""),
    ]
    y = y0 + 38
    cv2.putText(canvas, "metric               fixed/no-model      property-aware", (x0 + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (186, 192, 190), 1, cv2.LINE_AA)
    y += 34
    for label, fixed_value, model_value, unit in rows:
        delta = model_value - fixed_value
        color = (110, 220, 150) if label == "reward" and delta >= 0 else (230, 234, 232)
        cv2.putText(
            canvas,
            f"{label:<18} {fixed_value:12.4g}      {model_value:12.4g} {unit}",
            (x0 + 22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 31
    cv2.putText(
        canvas,
        f"estimated strength index={float(model_policy.get('estimated_strength_index', 0.0)):.3f}",
        (x0 + 22, y0 + h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (202, 208, 206),
        1,
        cv2.LINE_AA,
    )


def draw_material_line(canvas: np.ndarray, gt: dict[str, float], pred: dict[str, float], origin: tuple[int, int]) -> None:
    text = (
        f"GT/pred: rho {gt['rho']:.0f}/{pred['rho']:.0f}, "
        f"phi {gt['phi_deg']:.1f}/{pred['phi_deg']:.1f}, "
        f"delta {gt['delta_deg']:.1f}/{pred['delta_deg']:.1f}, "
        f"cohesion {gt['cohesion_kpa']:.1f}/{pred['cohesion_kpa']:.1f}"
    )
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 224, 222), 1, cv2.LINE_AA)


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
