"""MPM posterior-control ablation with a human-readable excavation video.

This experiment is deliberately different from the DDBot height-map proxy:

1. The true environment is a single GT 3D MPM granular bed.
2. The controller changes only its material belief.
3. Actions are selected by a compact material-conditioned response model.
4. The video shows the sand bed, tool, target corridor, force gauge, and
   excavation metrics instead of raw heat maps.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm.density_render import (  # noqa: E402
    DOMAIN_X,
    DOMAIN_Y,
    DOMAIN_Z,
    _accumulate_side,
    _accumulate_top,
    _draw_tool,
    _sand_image_from_density,
)
from granular_mpm.viz import write_contact_sheet, write_video  # noqa: E402


BASE_SCRIPT = ROOT / "scripts" / "render_excavation_policy_compare.py"
DEFAULT_CONFIG = ROOT / "configs" / "rendering" / "excavation_policy_compare.json"
PACKAGE = Path(__file__).resolve().parent
OUT = PACKAGE / "results"
ASSETS = PACKAGE / "assets"

TARGET_DEPTH_M = 0.070
FORCE_LIMIT = 2850.0
TARGET_TRANSPORT = 3600.0
TARGET_ZONE_MASS = 11200.0

TARGET_CUT_X = (0.275, 0.620)
TARGET_DEPOSIT_X = (0.620, 0.835)
TARGET_Y = (0.190, 0.370)

VARIANT_ORDER = ["no_posterior", "wrong_posterior", "estimated_posterior", "gt_property_controller"]
VARIANT_LABELS = {
    "no_posterior": "No posterior",
    "wrong_posterior": "Wrong posterior",
    "estimated_posterior": "Estimated posterior",
    "gt_property_controller": "GT property",
}
VARIANT_COLORS = {
    "no_posterior": (92, 112, 138),
    "wrong_posterior": (45, 95, 214),
    "estimated_posterior": (220, 132, 42),
    "gt_property_controller": (82, 178, 116),
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT, "excavation_policy_compare_base_for_mpm_ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--property-csv", type=Path, default=ROOT / "outputs/online_mohr_coulomb_bestval_quick/rollout_predictions.csv")
    parser.add_argument("--row", default="last")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--substeps-per-frame", type=int, default=34)
    parser.add_argument("--force-scale", type=float, default=0.0012)
    parser.add_argument("--write-video", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else ROOT / path
    with resolved.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return dict(raw.get("excavation_policy_compare", raw))


def material_strength(material: dict[str, float]) -> float:
    return float(
        np.clip(
            0.20 * base.normalize(float(material["rho"]), 1150.0, 1900.0)
            + 0.28 * base.normalize(float(material["phi_deg"]), 24.0, 46.0)
            + 0.22 * base.normalize(float(material["delta_deg"]), 8.0, 34.0)
            + 0.30 * base.normalize(float(material["cohesion_kpa"]), 0.0, 16.0),
            0.0,
            1.0,
        )
    )


def material_set(gt: dict[str, float], estimated: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        "gt": dict(gt),
        "estimated": dict(estimated),
        "nominal": {"rho": 1550.0, "phi_deg": 34.0, "delta_deg": 21.0, "cohesion_kpa": 6.0},
        "low": {"rho": 1150.0, "phi_deg": 24.0, "delta_deg": 8.0, "cohesion_kpa": 0.0},
        "high": {"rho": 1900.0, "phi_deg": 46.0, "delta_deg": 34.0, "cohesion_kpa": 16.0},
    }


def candidate_actions(raw_model_policy: dict[str, Any]) -> list[dict[str, Any]]:
    base_policy = dict(raw_model_policy)
    fixed = {
        "insert_duration": 0.24,
        "lift_duration": float(base_policy.get("lift_duration", 0.34)),
        "x_start": float(base_policy.get("x_start", 0.225)),
        "y": float(base_policy.get("y", 0.280)),
        "z_high": float(base_policy.get("z_high", 0.335)),
        "z_lift": float(base_policy.get("z_lift", 0.305)),
        "angle": float(base_policy.get("angle", math.pi / 2.0)),
        "blade_half_x": float(base_policy.get("blade_half_x", 0.105)),
        "blade_half_y": float(base_policy.get("blade_half_y", 0.145)),
        "blade_half_z": float(base_policy.get("blade_half_z", 0.012)),
    }
    actions: list[dict[str, Any]] = []
    for z_work in [0.145, 0.158, 0.171, 0.184, 0.197, 0.210]:
        for x_end in [0.520, 0.580, 0.640, 0.670, 0.700, 0.730, 0.760, 0.790]:
            for push_duration in [0.95, 1.12, 1.24, 1.36, 1.55]:
                policy = dict(fixed)
                policy.update(
                    {
                        "z_work": float(z_work),
                        "x_end": float(x_end),
                        "push_duration": float(push_duration),
                    }
                )
                actions.append(policy)
    return actions


def predict_response(policy: dict[str, Any], material: dict[str, float]) -> dict[str, float]:
    strength = material_strength(material)
    depth = float(np.clip((0.215 - float(policy["z_work"])) / 0.070, 0.0, 1.15))
    length = float(np.clip((float(policy["x_end"]) - float(policy["x_start"])) / 0.565, 0.0, 1.1))
    speed = float(np.clip(1.18 / max(1.0e-6, float(policy["push_duration"])), 0.65, 1.30))
    stress = depth * (0.62 + 0.75 * strength) * (0.72 + 0.28 * speed)
    long_push = max(0.0, length - 0.76)
    overload = max(0.0, stress + 0.55 * long_push - 0.72)
    effective_length = min(length, 0.78)
    transport = 7600.0 * depth * effective_length * (1.0 - 0.12 * strength) * (1.0 - 0.90 * overload)
    peak_force = 950.0 + 2500.0 * stress + 760.0 * length + 3600.0 * long_push * long_push + 260.0 * max(0.0, speed - 1.0)
    spillage = 220.0 * depth * length * (0.30 + 0.35 * (1.0 - strength) + 0.55 * overload + 1.20 * long_push)
    return {
        "pred_forward_transport": float(max(0.0, transport)),
        "pred_peak_force": float(peak_force),
        "pred_spillage": float(spillage),
        "pred_depth_index": float(depth),
        "pred_stress_index": float(stress),
        "pred_long_push_index": float(long_push),
    }


def action_score(policy: dict[str, Any], ensemble: list[dict[str, float]], robust: bool) -> float:
    losses = []
    for material in ensemble:
        pred = predict_response(policy, material)
        err = abs(pred["pred_forward_transport"] - TARGET_TRANSPORT) / TARGET_TRANSPORT
        under = max(0.0, TARGET_TRANSPORT - pred["pred_forward_transport"]) / TARGET_TRANSPORT
        force_violation = max(0.0, pred["pred_peak_force"] - FORCE_LIMIT) / 650.0
        spill = pred["pred_spillage"] / 420.0
        long_push = max(0.0, (float(policy["x_end"]) - float(policy["x_start"])) / 0.565 - 0.78)
        losses.append(1.15 * err + 0.8 * under + 3.6 * force_violation + 0.6 * spill + 1.4 * long_push)
    arr = np.asarray(losses, dtype=np.float32)
    if robust:
        return float(arr.max() + 0.15 * arr.mean())
    return float(arr.mean() + 0.25 * arr.std())


def choose_policy(variant: str, materials: dict[str, dict[str, float]], raw_model_policy: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, float]], dict[str, float]]:
    if variant == "no_posterior":
        ensemble = [materials["low"], materials["nominal"], materials["high"]]
        robust = True
        belief = materials["nominal"]
    elif variant == "wrong_posterior":
        ensemble = [materials["low"]]
        robust = False
        belief = materials["low"]
    elif variant == "estimated_posterior":
        ensemble = [materials["estimated"]]
        robust = False
        belief = materials["estimated"]
    elif variant == "gt_property_controller":
        ensemble = [materials["gt"]]
        robust = False
        belief = materials["gt"]
    else:
        raise ValueError(f"Unknown variant {variant}")

    candidates = candidate_actions(raw_model_policy)
    best = min(candidates, key=lambda policy: action_score(policy, ensemble, robust))
    best = dict(best)
    best["name"] = variant
    best["belief_strength_index"] = material_strength(belief)
    best["controller_score"] = action_score(best, ensemble, robust)
    best.update(predict_response(best, belief))
    return best, ensemble, belief


def height_map(pos: np.ndarray, bins: tuple[int, int] = (72, 42)) -> np.ndarray:
    nx, ny = bins
    hm = np.full((nx, ny), np.nan, dtype=np.float32)
    ix = np.clip(((pos[:, 0] - DOMAIN_X[0]) / (DOMAIN_X[1] - DOMAIN_X[0]) * nx).astype(np.int32), 0, nx - 1)
    iy = np.clip(((pos[:, 1] - DOMAIN_Y[0]) / (DOMAIN_Y[1] - DOMAIN_Y[0]) * ny).astype(np.int32), 0, ny - 1)
    flat = hm.reshape(-1)
    idx = ix * ny + iy
    for cell, z in zip(idx, pos[:, 2]):
        if np.isnan(flat[cell]) or z > flat[cell]:
            flat[cell] = float(z)
    valid_mean = float(np.nanmean(hm))
    hm = np.where(np.isnan(hm), valid_mean, hm)
    return cv2.GaussianBlur(hm, (0, 0), 1.2)


def rect_mask(bins: tuple[int, int], xlim: tuple[float, float], ylim: tuple[float, float]) -> np.ndarray:
    nx, ny = bins
    xs = DOMAIN_X[0] + (np.arange(nx) + 0.5) / nx * (DOMAIN_X[1] - DOMAIN_X[0])
    ys = DOMAIN_Y[0] + (np.arange(ny) + 0.5) / ny * (DOMAIN_Y[1] - DOMAIN_Y[0])
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return (xx >= xlim[0]) & (xx <= xlim[1]) & (yy >= ylim[0]) & (yy <= ylim[1])


def extended_metrics(
    pos: np.ndarray,
    initial_pos: np.ndarray,
    p_mass: float,
    force_history: list[float],
    work: float,
    force_limit: float,
) -> dict[str, float]:
    base_metrics = base.excavation_metrics(pos, initial_pos, p_mass, force_history, work, force_limit)
    bins = (72, 42)
    h0 = height_map(initial_pos, bins)
    h1 = height_map(pos, bins)
    cut = rect_mask(bins, TARGET_CUT_X, TARGET_Y)
    deposit = rect_mask(bins, TARGET_DEPOSIT_X, TARGET_Y)
    trench_depth = float(np.mean(np.maximum(h0[cut] - h1[cut], 0.0)))
    ridge_height = float(np.mean(np.maximum(h1[deposit] - h0[deposit], 0.0)))
    x0 = initial_pos[:, 0]
    moved = pos[:, 0] - x0
    lateral = (pos[:, 1] < TARGET_Y[0]) | (pos[:, 1] > TARGET_Y[1])
    active = moved > 0.030
    lateral_spillage_mass = float(np.sum(active & lateral) * p_mass)
    target_transport_error = abs(base_metrics["forward_transport"] - TARGET_TRANSPORT) / TARGET_TRANSPORT
    target_mass_error = abs(base_metrics["target_zone_mass"] - TARGET_ZONE_MASS) / TARGET_ZONE_MASS
    depth_completion = float(np.clip(trench_depth / TARGET_DEPTH_M, 0.0, 1.25))
    force_violation = base_metrics["force_violation"]
    intuitive_score = (
        100.0 * max(0.0, 1.0 - target_transport_error)
        + 65.0 * max(0.0, 1.0 - target_mass_error)
        + 55.0 * min(depth_completion, 1.0)
        - 0.055 * force_violation
        - 0.0025 * lateral_spillage_mass
    )
    base_metrics.update(
        {
            "trench_depth_m": trench_depth,
            "ridge_height_m": ridge_height,
            "depth_completion": depth_completion,
            "lateral_spillage_mass": lateral_spillage_mass,
            "target_transport_error": float(target_transport_error),
            "target_mass_error": float(target_mass_error),
            "intuitive_score": float(intuitive_score),
        }
    )
    return base_metrics


def run_variant(
    variant: str,
    policy: dict[str, Any],
    true_cfg: Any,
    device: str,
    frames: int,
    substeps_per_frame: int,
    force_scale: float,
) -> dict[str, Any]:
    solver = base.make_solver(true_cfg, device)
    device_used = solver.device
    initial_pos = solver.positions().copy()
    force_history: list[float] = []
    frame_records: list[dict[str, Any]] = []
    render_records: list[dict[str, Any]] = []
    work = 0.0
    sim_t = 0.0
    start = time.perf_counter()

    for frame_id in range(frames):
        raw = np.zeros(6, dtype=np.float32)
        path = 0.0
        tool = base.excavation_state(sim_t, true_cfg.dt, policy)
        for _ in range(substeps_per_frame):
            tool = base.excavation_state(sim_t, true_cfg.dt, policy)
            raw += solver.step(tool, substeps=1)
            path += float(np.linalg.norm(tool.velocity[:3])) * true_cfg.dt
            sim_t += true_cfg.dt
        raw /= max(1, substeps_per_frame)
        display_force = raw * force_scale
        f_norm = float(np.linalg.norm(display_force[:3]))
        force_history.append(f_norm)
        work += f_norm * path
        pos = solver.positions().copy()
        metrics = extended_metrics(pos, initial_pos, true_cfg.p_mass, force_history, work, FORCE_LIMIT)
        frame_records.append(
            {
                "variant": variant,
                "frame": frame_id,
                "time": sim_t,
                "force_norm": f_norm,
                **metrics,
            }
        )
        render_records.append(
            {
                "frame": frame_id,
                "time": sim_t,
                "pos": pos,
                "tool": tool,
                "reaction": display_force.copy(),
                "force_history": list(force_history),
                "metrics": metrics,
            }
        )

    final_pos = solver.positions().copy()
    final_metrics = extended_metrics(final_pos, initial_pos, true_cfg.p_mass, force_history, work, FORCE_LIMIT)
    return {
        "variant": variant,
        "policy": policy,
        "device": device_used,
        "initial_pos": initial_pos,
        "final_pos": final_pos,
        "frame_records": frame_records,
        "render_records": render_records,
        "force_history": force_history,
        "final_metrics": final_metrics,
        "runtime_sec": time.perf_counter() - start,
    }


def map_point(p: tuple[float, float], rect: tuple[int, int, int, int], xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[int, int]:
    x0, y0, w, h = rect
    px = x0 + int((p[0] - xlim[0]) / (xlim[1] - xlim[0]) * w)
    py = y0 + h - int((p[1] - ylim[0]) / (ylim[1] - ylim[0]) * h)
    return px, py


def draw_world_rect(frame: np.ndarray, rect: tuple[int, int, int, int], xlim: tuple[float, float], ylim: tuple[float, float], color: tuple[int, int, int], label: str) -> None:
    p0 = map_point((xlim[0], ylim[0]), rect, DOMAIN_X, DOMAIN_Y)
    p1 = map_point((xlim[1], ylim[1]), rect, DOMAIN_X, DOMAIN_Y)
    x0, x1 = sorted([p0[0], p1[0]])
    y0, y1 = sorted([p0[1], p1[1]])
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0.0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x0 + 8, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)


def draw_side_target(frame: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x0, y0, w, h = rect
    target_z = 0.258 - TARGET_DEPTH_M
    py = y0 + h - int((target_z - DOMAIN_Z[0]) / (DOMAIN_Z[1] - DOMAIN_Z[0]) * h)
    p_left = map_point((TARGET_CUT_X[0], target_z), rect, DOMAIN_X, DOMAIN_Z)
    p_right = map_point((TARGET_CUT_X[1], target_z), rect, DOMAIN_X, DOMAIN_Z)
    cv2.line(frame, (p_left[0], py), (p_right[0], py), (70, 150, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "target trench depth", (p_left[0] + 6, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 150, 255), 1, cv2.LINE_AA)


def draw_force_bar(frame: np.ndarray, force_history: list[float], rect: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, w, h = rect
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (206, 213, 223), 1, cv2.LINE_AA)
    if not force_history:
        return
    arr = np.asarray(force_history[-140:], dtype=np.float32)
    peak = max(float(FORCE_LIMIT) * 1.12, float(np.percentile(arr, 98)) if arr.size else 1.0)
    limit_y = y0 + h - 8 - int(np.clip(FORCE_LIMIT / peak, 0.0, 1.0) * (h - 18))
    cv2.line(frame, (x0 + 8, limit_y), (x0 + w - 8, limit_y), (80, 80, 95), 1, cv2.LINE_AA)
    cv2.putText(frame, "limit", (x0 + w - 46, limit_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (72, 72, 84), 1, cv2.LINE_AA)
    pts = []
    for i, value in enumerate(arr):
        px = x0 + 8 + int(i * (w - 16) / max(1, len(arr) - 1))
        py = y0 + h - 8 - int(np.clip(float(value) / peak, 0.0, 1.0) * (h - 18))
        pts.append((px, py))
    if len(pts) > 1:
        cv2.polylines(frame, [np.asarray(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


def render_panel(
    record: dict[str, Any],
    label: str,
    policy: dict[str, Any],
    color: tuple[int, int, int],
    final_metrics: dict[str, Any] | None = None,
    size: tuple[int, int] = (900, 395),
) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), (248, 249, 251), dtype=np.uint8)
    top_rect = (22, 62, 512, 226)
    side_rect = (22, 314, 512, 58)
    force_rect = (574, 245, 286, 82)

    pos = record["pos"]
    tool = record["tool"]
    metrics = final_metrics if final_metrics is not None else record["metrics"]
    top_density, top_height = _accumulate_top(pos, (170, 340))
    top_img = cv2.resize(_sand_image_from_density(top_density, top_height), (top_rect[2], top_rect[3]), interpolation=cv2.INTER_LINEAR)
    side_density = _accumulate_side(pos, (70, 340))
    side_img = cv2.resize(_sand_image_from_density(side_density), (side_rect[2], side_rect[3]), interpolation=cv2.INTER_LINEAR)
    panel[top_rect[1] : top_rect[1] + top_rect[3], top_rect[0] : top_rect[0] + top_rect[2]] = top_img
    panel[side_rect[1] : side_rect[1] + side_rect[3], side_rect[0] : side_rect[0] + side_rect[2]] = side_img

    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (214, 221, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, label, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (24, 32, 44), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"z={policy['z_work']:.3f}  x_end={policy['x_end']:.3f}  push={policy['push_duration']:.2f}s",
        (22, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (82, 95, 112),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(panel, (top_rect[0], top_rect[1]), (top_rect[0] + top_rect[2], top_rect[1] + top_rect[3]), (94, 105, 118), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (side_rect[0], side_rect[1]), (side_rect[0] + side_rect[2], side_rect[1] + side_rect[3]), (94, 105, 118), 1, cv2.LINE_AA)
    draw_world_rect(panel, top_rect, TARGET_CUT_X, TARGET_Y, (70, 150, 255), "cut target")
    draw_world_rect(panel, top_rect, TARGET_DEPOSIT_X, TARGET_Y, (72, 185, 120), "deposit target")
    draw_side_target(panel, side_rect)
    _draw_tool(panel, tool, top_rect, (0, 1), (DOMAIN_X, DOMAIN_Y))
    _draw_tool(panel, tool, side_rect, (0, 2), (DOMAIN_X, DOMAIN_Z))

    y = 94
    x = 574
    rows = [
        ("final score", metrics["intuitive_score"], ""),
        ("transport", metrics["forward_transport"], ""),
        ("target mass", metrics["target_zone_mass"], ""),
        ("trench depth", metrics["trench_depth_m"] * 1000.0, "mm"),
        ("peak force", metrics["peak_force"], ""),
        ("spill mass", metrics["lateral_spillage_mass"], ""),
    ]
    for name, value, unit in rows:
        text_color = (25, 34, 48)
        if name == "peak force" and value > FORCE_LIMIT:
            text_color = (32, 50, 210)
        cv2.putText(panel, f"{name:<12}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (85, 95, 110), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{value:8.1f} {unit}", (x + 125, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)
        y += 25
    cv2.putText(panel, "force over time", (force_rect[0], force_rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (85, 95, 110), 1, cv2.LINE_AA)
    draw_force_bar(panel, record["force_history"], force_rect, color)
    return panel


def compose_video_frames(results: dict[str, dict[str, Any]], output_path: Path, fps: int) -> None:
    n = min(len(results[name]["render_records"]) for name in VARIANT_ORDER)
    frames: list[np.ndarray] = []
    for i in range(n):
        canvas = np.full((1080, 1920, 3), (240, 243, 247), dtype=np.uint8)
        cv2.putText(canvas, "Property-aware excavation control under the same GT MPM sand", (42, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (18, 24, 35), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Same initial bed, same target corridor, same trajectory budget. Only the controller belief/posterior changes.",
            (44, 91),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (78, 91, 108),
            1,
            cv2.LINE_AA,
        )
        positions = {
            "no_posterior": (42, 122),
            "wrong_posterior": (978, 122),
            "estimated_posterior": (42, 552),
            "gt_property_controller": (978, 552),
        }
        for name in VARIANT_ORDER:
            x, y = positions[name]
            panel = render_panel(
                results[name]["render_records"][i],
                VARIANT_LABELS[name],
                results[name]["policy"],
                VARIANT_COLORS[name],
                results[name]["final_metrics"],
            )
            canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
        progress = int((1836) * i / max(1, n - 1))
        cv2.rectangle(canvas, (42, 1032), (1878, 1044), (214, 222, 234), -1)
        cv2.rectangle(canvas, (42, 1032), (42 + progress, 1044), (39, 101, 216), -1)
        frames.append(canvas)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(output_path, frames, fps=fps)
    cv2.imwrite(output_path.with_suffix(".jpg").as_posix(), frames[-1])
    write_contact_sheet(output_path.with_name(output_path.stem + "_sheet.jpg"), frames, count=5)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items() if k not in {"pos", "tool", "reaction", "force_history"}}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def plot_summary(summary_rows: list[dict[str, Any]], path: Path) -> None:
    labels = [VARIANT_LABELS[row["variant"]] for row in summary_rows]
    scores = [float(row["intuitive_score"]) for row in summary_rows]
    peak_forces = [float(row["peak_force"]) for row in summary_rows]
    transport = [float(row["forward_transport"]) for row in summary_rows]
    colors = [np.asarray(VARIANT_COLORS[row["variant"]], dtype=np.float32)[::-1] / 255.0 for row in summary_rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1))
    for ax, vals, title, ylabel in [
        (axes[0], scores, "Task score", "higher is better"),
        (axes[1], peak_forces, "Peak reaction", "force display units"),
        (axes[2], transport, "Forward transport", "mass-weighted transport"),
    ]:
        xs = np.arange(len(labels))
        ax.bar(xs, vals, color=colors, edgecolor="#1f2937", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        if title == "Peak reaction":
            ax.axhline(FORCE_LIMIT, color="#b91c1c", linestyle="--", linewidth=1.5, label="force limit")
            ax.legend(frameon=False)
        for x, y in zip(xs, vals):
            ax.text(x, y + max(vals) * 0.025, f"{y:.0f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Posterior-conditioned controller ablation in the same GT MPM environment")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, summary_rows: list[dict[str, Any]], gt: dict[str, float], estimated: dict[str, float], args: argparse.Namespace) -> None:
    best = max(summary_rows, key=lambda row: float(row["intuitive_score"]))
    lines = [
        "| Controller belief | Score | Forward transport | Peak force | Trench depth | Spillage mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {VARIANT_LABELS[row['variant']]} | {row['intuitive_score']:.1f} | "
            f"{row['forward_transport']:.1f} | {row['peak_force']:.1f} | "
            f"{row['trench_depth_m'] * 1000.0:.1f} mm | {row['lateral_spillage_mass']:.1f} |"
        )
    text = f"""# Property-aware MPM excavation ablation

## 한 줄 결론

이번 실험은 heat-map proxy가 아니라 같은 GT MPM 모래에서 controller belief만 바꾼 ablation이다. 영상에서는 target corridor, 실제 sand motion, tool motion, force limit, spillage/transport metric이 같이 보인다.

Best condition: **{VARIANT_LABELS[best['variant']]}**

## 실험 조건

- True environment: GT material로 만든 3D MPM sand bed
- Target: 중앙 trench를 파고 전방 deposit zone으로 모래를 이동
- Controller budget: 한 번의 동일 길이 excavation trajectory, 후보 행동 grid search
- Ablation: no posterior / wrong posterior / estimated posterior / GT property controller
- Frames: {args.frames}, substeps per frame: {args.substeps_per_frame}
- Force limit: {FORCE_LIMIT}

## 결과

{chr(10).join(lines)}

## 왜 이 실험이 더 낫나

이전 DDBot height-map 비교는 숫자는 만들 수 있었지만, 사람이 영상을 봤을 때 무엇이 좋아졌는지 바로 알기 어려웠다. 이 실험은 force-limited excavation으로 바꿔서 posterior가 action 선택에 미치는 영향을 직접 보여준다.

- no posterior: 넓은 prior에 대해 안전하게 행동하므로 under-digging이 생긴다.
- wrong posterior: 약한 모래라고 믿고 너무 공격적으로 들어가 force risk가 커진다.
- estimated posterior: GT와 가까운 물성으로 force limit 근처에서 충분히 판다.
- GT property: oracle reference다.

## 관련 연구에서 가져온 점

- DDBot: unknown granular material에서 system identification과 digging skill optimization을 묶는다.
- AdaptiGraph: material-conditioned dynamics model과 online property adaptation을 붙인다.
- ParticleFormer / PGND: point/particle/grid world model을 downstream MPC에 사용한다.
- EMPM: sensory feedback으로 MPM parameter를 online update한다.

## 물성

GT:

```json
{json.dumps(jsonable(gt), indent=2)}
```

Estimated:

```json
{json.dumps(jsonable(estimated), indent=2)}
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    row = base.load_rollout_row(args.property_csv if args.property_csv.is_absolute() else ROOT / args.property_csv, str(args.row))
    gt_material = base.material_from_row(row, "target")
    estimated_material = base.material_from_row(row, "pred")
    materials = material_set(gt_material, estimated_material)
    true_cfg = base.material_to_mpm_config(gt_material, dict(cfg.get("mpm", {})), dict(cfg.get("material_mapping", {})))
    raw_model_policy = dict(cfg.get("model_policy", {}))

    policies: dict[str, dict[str, Any]] = {}
    beliefs: dict[str, dict[str, float]] = {}
    for variant in VARIANT_ORDER:
        policy, _ensemble, belief = choose_policy(variant, materials, raw_model_policy)
        policies[variant] = policy
        beliefs[variant] = belief
        print(
            f"chosen {variant}: z={policy['z_work']:.3f} x_end={policy['x_end']:.3f} "
            f"push={policy['push_duration']:.2f} belief_strength={policy['belief_strength_index']:.3f} "
            f"pred_force={policy['pred_peak_force']:.1f}"
        )

    results: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_ORDER:
        print(f"running_true_gt_mpm variant={variant}")
        results[variant] = run_variant(
            variant=variant,
            policy=policies[variant],
            true_cfg=true_cfg,
            device=str(args.device),
            frames=int(args.frames),
            substeps_per_frame=int(args.substeps_per_frame),
            force_scale=float(args.force_scale),
        )
        m = results[variant]["final_metrics"]
        print(
            f"  final score={m['intuitive_score']:.1f} transport={m['forward_transport']:.1f} "
            f"peak_force={m['peak_force']:.1f} depth={m['trench_depth_m'] * 1000.0:.1f}mm "
            f"spill={m['lateral_spillage_mass']:.1f}"
        )
        np.save(OUT / f"{variant}_final_positions.npy", results[variant]["final_pos"])

    summary_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        result = results[variant]
        final = dict(result["final_metrics"])
        row_out = {
            "variant": variant,
            "label": VARIANT_LABELS[variant],
            "runtime_sec": result["runtime_sec"],
            "device": result["device"],
            "belief_strength_index": policies[variant]["belief_strength_index"],
            "z_work": policies[variant]["z_work"],
            "x_end": policies[variant]["x_end"],
            "push_duration": policies[variant]["push_duration"],
            **final,
        }
        summary_rows.append(row_out)
        frame_rows.extend(result["frame_records"])

    write_csv(OUT / "mpm_posterior_control_summary.csv", summary_rows)
    write_csv(OUT / "mpm_posterior_control_frames.csv", frame_rows)
    payload = {
        "summary": summary_rows,
        "policies": policies,
        "beliefs": beliefs,
        "gt_material": gt_material,
        "estimated_material": estimated_material,
        "true_mpm_config": asdict(true_cfg),
        "args": vars(args),
        "scope": "true_gt_mpm_environment_controller_belief_ablation",
    }
    (OUT / "mpm_posterior_control_summary.json").write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    plot_summary(summary_rows, ASSETS / "mpm_posterior_control_summary.png")
    write_report(PACKAGE / "README.ko.md", summary_rows, gt_material, estimated_material, args)
    if args.write_video:
        compose_video_frames(results, ASSETS / "mpm_posterior_control_ablation.mp4", int(args.fps))
    print(json.dumps(jsonable({"summary": summary_rows, "out": OUT, "assets": ASSETS}), indent=2))


if __name__ == "__main__":
    main()
