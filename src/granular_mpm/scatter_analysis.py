from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .metrics import write_json


TARGET_COLUMNS = ["phi_deg", "cohesion_kpa"]
NUISANCE_COLUMNS = ["speed_scale", "depth_scale", "drag_distance_scale", "angle_offset", "y_offset", "z_offset"]
FORCE_METRICS = [
    "max_raw_force",
    "mean_active_raw_force",
    "impulse_raw_force",
    "force_energy",
    "contact_fraction",
    "contact_onset_time",
]


def analyze_sweep_scatter(sweep_root: Path, out_dir: Path | None = None, contact_threshold: float = 1.0) -> dict[str, Any]:
    sweep_root = sweep_root.resolve()
    out_dir = (out_dir or (sweep_root / "analysis")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = read_csv_dicts(sweep_root / "samples.csv")
    summary_rows = []
    for sample in sample_rows:
        sequence_root = Path(str(sample.get("sequence_root", "")))
        wrench_log = sequence_root / "runs" / "blade_demo" / "wrench_log.csv"
        metrics = wrench_log_metrics(wrench_log, contact_threshold=contact_threshold)
        summary_rows.append({**sample, **metrics})

    write_summary_csv(out_dir / "force_summary.csv", summary_rows)
    correlations = correlation_table(summary_rows, TARGET_COLUMNS + NUISANCE_COLUMNS, FORCE_METRICS)
    write_summary_csv(out_dir / "correlations.csv", correlations)

    plot_paths = make_scatter_plots(summary_rows, out_dir)
    diagnostics = diagnose(summary_rows, correlations, contact_threshold=contact_threshold)
    report = {
        "sweep_root": sweep_root.as_posix(),
        "out_dir": out_dir.as_posix(),
        "sample_count": len(summary_rows),
        "contact_threshold": contact_threshold,
        "summary_csv": (out_dir / "force_summary.csv").as_posix(),
        "correlations_csv": (out_dir / "correlations.csv").as_posix(),
        "plots": plot_paths,
        "diagnostics": diagnostics,
    }
    write_json(out_dir / "scatter_report.json", report)
    write_markdown_report(out_dir / "scatter_report.md", report, summary_rows, correlations)
    return report


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [coerce_row(row) for row in csv.DictReader(f)]


def coerce_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            out[key] = value
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def wrench_log_metrics(path: Path, contact_threshold: float = 1.0) -> dict[str, Any]:
    if not path.exists():
        return {"wrench_log": path.as_posix(), "wrench_exists": False}
    rows = read_csv_dicts(path)
    if not rows:
        return {"wrench_log": path.as_posix(), "wrench_exists": True, "frame_count": 0}

    time = np.asarray([float(row.get("time", i)) for i, row in enumerate(rows)], dtype=np.float32)
    force = np.asarray(
        [[float(row.get("raw_fx", 0.0)), float(row.get("raw_fy", 0.0)), float(row.get("raw_fz", 0.0))] for row in rows],
        dtype=np.float32,
    )
    torque = np.asarray(
        [[float(row.get("raw_tx", 0.0)), float(row.get("raw_ty", 0.0)), float(row.get("raw_tz", 0.0))] for row in rows],
        dtype=np.float32,
    )
    tool = np.asarray(
        [[float(row.get("tool_x", 0.0)), float(row.get("tool_y", 0.0)), float(row.get("tool_z", 0.0))] for row in rows],
        dtype=np.float32,
    )
    force_norm = np.linalg.norm(force, axis=1)
    torque_norm = np.linalg.norm(torque, axis=1)
    dt = timestep_weights(time)
    active = force_norm > float(contact_threshold)
    path_length = float(np.linalg.norm(np.diff(tool, axis=0), axis=1).sum()) if tool.shape[0] > 1 else 0.0
    duration = float(time[-1] - time[0]) if time.size > 1 else 0.0
    active_force = force_norm[active]

    return {
        "wrench_log": path.as_posix(),
        "wrench_exists": True,
        "frame_count": int(len(rows)),
        "duration": duration,
        "max_raw_force": float(force_norm.max()) if force_norm.size else 0.0,
        "mean_raw_force": float(force_norm.mean()) if force_norm.size else 0.0,
        "mean_active_raw_force": float(active_force.mean()) if active_force.size else 0.0,
        "max_raw_torque": float(torque_norm.max()) if torque_norm.size else 0.0,
        "impulse_raw_force": float(np.sum(force_norm * dt)) if force_norm.size else 0.0,
        "force_energy": float(np.sum(force_norm * force_norm * dt)) if force_norm.size else 0.0,
        "contact_fraction": float(np.count_nonzero(active) / max(1, force_norm.size)),
        "contact_onset_time": float(time[np.argmax(active)]) if np.any(active) else None,
        "tool_path_length": path_length,
        "tool_mean_speed": path_length / max(duration, 1.0e-6),
        "tool_z_travel": float(tool[:, 2].max() - tool[:, 2].min()) if tool.size else 0.0,
        "trajectory_speed_scale_log": float(rows[0].get("trajectory_speed_scale", math.nan)),
        "trajectory_depth_scale_log": float(rows[0].get("trajectory_depth_scale", math.nan)),
        "trajectory_angle_offset_log": float(rows[0].get("trajectory_angle_offset", math.nan)),
        "trajectory_drag_distance_scale_log": float(rows[0].get("trajectory_drag_distance_scale", math.nan)),
    }


def timestep_weights(time: np.ndarray) -> np.ndarray:
    if time.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if time.size == 1:
        return np.ones((1,), dtype=np.float32)
    dt = np.diff(time, prepend=time[0])
    dt[0] = dt[1]
    return np.maximum(dt, 0.0).astype(np.float32)


def correlation_table(rows: list[dict[str, Any]], x_cols: list[str], y_cols: list[str]) -> list[dict[str, Any]]:
    out = []
    for x_col in x_cols:
        for y_col in y_cols:
            x = column_array(rows, x_col)
            y = column_array(rows, y_col)
            out.append(
                {
                    "x": x_col,
                    "y": y_col,
                    "pearson": pearson(x, y),
                    "spearman": pearson(rankdata(x), rankdata(y)),
                    "count": int(np.count_nonzero(np.isfinite(x) & np.isfinite(y))),
                }
            )
    return out


def column_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, math.nan)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(math.nan)
    return np.asarray(values, dtype=np.float32)


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return None
    xx = x[mask]
    yy = y[mask]
    if float(xx.std()) < 1.0e-9 or float(yy.std()) < 1.0e-9:
        return None
    return float(np.corrcoef(xx, yy)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    ranks = np.full_like(arr, np.nan, dtype=np.float32)
    mask = np.isfinite(arr)
    order = np.argsort(arr[mask], kind="mergesort")
    sorted_idx = np.where(mask)[0][order]
    ranks[sorted_idx] = np.arange(sorted_idx.size, dtype=np.float32)
    return ranks


def make_scatter_plots(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    specs = [
        ("phi_deg", "max_raw_force"),
        ("cohesion_kpa", "max_raw_force"),
        ("phi_deg", "impulse_raw_force"),
        ("cohesion_kpa", "impulse_raw_force"),
        ("speed_scale", "max_raw_force"),
        ("depth_scale", "max_raw_force"),
        ("angle_offset", "max_raw_force"),
        ("contact_fraction", "max_raw_force"),
    ]
    paths = {}
    for x_col, y_col in specs:
        path = out_dir / f"scatter_{x_col}_vs_{y_col}.png"
        draw_scatter(path, column_array(rows, x_col), column_array(rows, y_col), x_col, y_col)
        paths[f"{x_col}_vs_{y_col}"] = path.as_posix()
    return paths


def draw_scatter(path: Path, x: np.ndarray, y: np.ndarray, x_label: str, y_label: str) -> None:
    width, height = 780, 520
    ml, mr, mt, mb = 86, 36, 58, 72
    img = np.full((height, width, 3), 250, dtype=np.uint8)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    cv2.putText(img, f"{x_label} vs {y_label}", (ml, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (30, 30, 30), 2, cv2.LINE_AA)
    x0, x1 = ml, width - mr
    y0, y1 = height - mb, mt
    cv2.line(img, (x0, y0), (x1, y0), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.line(img, (x0, y0), (x0, y1), (80, 80, 80), 1, cv2.LINE_AA)
    if x.size == 0:
        cv2.putText(img, "no finite samples", (ml + 60, mt + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 180), 2, cv2.LINE_AA)
        cv2.imwrite(path.as_posix(), img)
        return

    xmin, xmax = padded_range(x)
    ymin, ymax = padded_range(y)
    for idx, (xv, yv) in enumerate(zip(x, y)):
        px = int(x0 + (float(xv) - xmin) / max(xmax - xmin, 1.0e-9) * (x1 - x0))
        py = int(y0 - (float(yv) - ymin) / max(ymax - ymin, 1.0e-9) * (y0 - y1))
        cv2.circle(img, (px, py), 6, (28, 110, 210), -1, cv2.LINE_AA)
        cv2.putText(img, str(idx), (px + 7, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)

    corr = pearson(x, y)
    if corr is not None and x.size >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        endpoints = np.asarray([xmin, xmax], dtype=np.float32)
        line_y = slope * endpoints + intercept
        pts = []
        for xv, yv in zip(endpoints, line_y):
            px = int(x0 + (float(xv) - xmin) / max(xmax - xmin, 1.0e-9) * (x1 - x0))
            py = int(y0 - (float(yv) - ymin) / max(ymax - ymin, 1.0e-9) * (y0 - y1))
            pts.append((px, py))
        cv2.line(img, pts[0], pts[1], (215, 90, 40), 2, cv2.LINE_AA)
        cv2.putText(img, f"r={corr:.3f}", (x1 - 116, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (215, 90, 40), 2, cv2.LINE_AA)

    cv2.putText(img, f"{xmin:.3g}", (x0 - 12, y0 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(img, f"{xmax:.3g}", (x1 - 56, y0 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(img, f"{ymin:.3g}", (10, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(img, f"{ymax:.3g}", (10, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(img, x_label, (width // 2 - 80, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(img, y_label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (50, 50, 50), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path.as_posix(), img)


def padded_range(values: np.ndarray) -> tuple[float, float]:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if abs(hi - lo) < 1.0e-9:
        pad = max(abs(lo) * 0.05, 1.0)
    else:
        pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def diagnose(rows: list[dict[str, Any]], correlations: list[dict[str, Any]], contact_threshold: float) -> dict[str, Any]:
    max_force = column_array(rows, "max_raw_force")
    contact_fraction = column_array(rows, "contact_fraction")
    phi = column_array(rows, "phi_deg")
    cohesion = column_array(rows, "cohesion_kpa")
    issues = []
    recommendations = []

    finite_force = max_force[np.isfinite(max_force)]
    median_max_force = float(np.median(finite_force)) if finite_force.size else 0.0
    median_contact = float(np.nanmedian(contact_fraction)) if contact_fraction.size else 0.0
    if len(rows) < 16:
        issues.append("sample_count_too_small")
        recommendations.append("Use at least 32 samples for a first read and 128+ for model comparison.")
    if median_max_force <= contact_threshold or median_contact < 0.05:
        issues.append("insufficient_contact")
        recommendations.append("Run without --quick, increase frames/substeps, or make depth_scale/z_offset more intrusive.")
    if np.nanmax(phi) - np.nanmin(phi) < 10.0 if phi.size else True:
        issues.append("phi_range_too_narrow")
        recommendations.append("Keep phi_deg range near 25-45 deg for the first sweep.")
    if np.nanmax(cohesion) - np.nanmin(cohesion) < 5.0 if cohesion.size else True:
        issues.append("cohesion_range_too_narrow")
        recommendations.append("Keep cohesion_kpa range near 0-15 kPa for the first sweep.")

    target_strength = max_abs_corr(correlations, TARGET_COLUMNS, ["max_raw_force", "impulse_raw_force", "force_energy"])
    nuisance_strength = max_abs_corr(correlations, NUISANCE_COLUMNS, ["max_raw_force", "impulse_raw_force", "force_energy"])
    if target_strength is not None and target_strength < 0.25 and median_max_force > contact_threshold:
        issues.append("weak_target_force_relation")
        recommendations.append("Increase material contrast or add repeated actions per material before trusting learned inference.")
    if target_strength is not None and nuisance_strength is not None and nuisance_strength > target_strength + 0.15:
        issues.append("nuisance_dominates_force")
        recommendations.append("Use a paired design: repeat 2-4 actions for each material and include action variables in the decoder input/metadata.")

    return {
        "issues": issues,
        "recommendations": recommendations,
        "median_max_raw_force": median_max_force,
        "median_contact_fraction": median_contact,
        "target_force_corr_max_abs": target_strength,
        "nuisance_force_corr_max_abs": nuisance_strength,
    }


def max_abs_corr(correlations: list[dict[str, Any]], x_cols: list[str], y_cols: list[str]) -> float | None:
    vals = []
    for row in correlations:
        if row.get("x") in x_cols and row.get("y") in y_cols and row.get("pearson") is not None:
            vals.append(abs(float(row["pearson"])))
    return max(vals) if vals else None


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_markdown_report(path: Path, report: dict[str, Any], rows: list[dict[str, Any]], correlations: list[dict[str, Any]]) -> None:
    diag = report["diagnostics"]
    top_corr = sorted(
        [row for row in correlations if row.get("pearson") is not None],
        key=lambda row: abs(float(row["pearson"])),
        reverse=True,
    )[:10]
    lines = [
        "# Sweep Scatter Analysis",
        "",
        f"- sweep_root: `{report['sweep_root']}`",
        f"- sample_count: {len(rows)}",
        f"- median_max_raw_force: {diag['median_max_raw_force']:.4g}",
        f"- median_contact_fraction: {diag['median_contact_fraction']:.4g}",
        f"- target_force_corr_max_abs: {diag['target_force_corr_max_abs']}",
        f"- nuisance_force_corr_max_abs: {diag['nuisance_force_corr_max_abs']}",
        "",
        "## Issues",
    ]
    lines.extend([f"- {issue}" for issue in diag["issues"]] or ["- none"])
    lines.append("")
    lines.append("## Recommendations")
    lines.extend([f"- {rec}" for rec in diag["recommendations"]] or ["- Continue with larger full sweep."])
    lines.append("")
    lines.append("## Top Correlations")
    for row in top_corr:
        lines.append(f"- {row['x']} vs {row['y']}: pearson={float(row['pearson']):.3f}, spearman={row['spearman']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
