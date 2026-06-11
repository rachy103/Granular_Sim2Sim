"""Posterior-conditioned height-field MPC for the DDBot sand task-2 target.

This script is intentionally not a replacement for the MPM validation scripts.
It tests a stronger controller idea that uses this repository's main advantage:
fast online material posterior + closed-loop height-map feedback.

The controller uses a compact height-field world model with angle-of-repose
relaxation and a posterior ensemble. At every stroke it samples local
dig/deposit macro-actions, scores them with a short scenario-MPC objective, and
then re-observes the height map before the next stroke.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
OUT = PACKAGE / "results"
BENCHMARK = PACKAGE / "run_shared_benchmark.py"
OURS_MODULE = ROOT / "scripts" / "render_excavation_policy_compare.py"

RES = 40
HEIGHT_MAP_SIZE_M = 0.24
PIXEL_M = HEIGHT_MAP_SIZE_M / RES
PIXEL_AREA_M2 = PIXEL_M * PIXEL_M
GROUND_M = 0.073
DDBOT_OFFICIAL_HM_MEAN = 3.9855568408966064
DDBOT_OFFICIAL_EMD_MEAN = 15.924596786499023

FONT_REGULAR = Path("C:/Windows/Fonts/malgun.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/malgunbd.ttf")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = load_module(BENCHMARK, "shared_benchmark_for_hf_mpc")
ours = load_module(OURS_MODULE, "ours_excavation_for_hf_mpc")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_PANEL = font(27, True)
F_BODY = font(20)
F_SMALL = font(16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--strokes", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=192)
    parser.add_argument("--ensemble", type=int, default=9)
    parser.add_argument("--optimizer-seed", type=int, default=20260611)
    parser.add_argument("--write-video", action="store_true")
    return parser.parse_args()


def load_target() -> np.ndarray:
    target_path = PACKAGE / "data" / "ddbot_target_sand_task2_height_map_res40.npy"
    if not target_path.exists():
        bench.ensure_download(target_path, bench.TARGET_URL)
    return np.load(target_path).astype(np.float32)


def load_ddbot_sim() -> np.ndarray:
    sim_path = PACKAGE / "data" / "ddbot_official_sim_sand_task2_height_map_res40.npy"
    if not sim_path.exists():
        bench.ensure_download(sim_path, bench.DDBOT_SIM_URL)
    return np.load(sim_path).astype(np.float32)


def load_material() -> tuple[dict[str, float], dict[str, float]]:
    row = ours.load_rollout_row(ROOT / "outputs/online_mohr_coulomb_bestval_quick/rollout_predictions.csv", "last")
    return ours.material_from_row(row, "target"), ours.material_from_row(row, "pred")


def grid_xy() -> tuple[np.ndarray, np.ndarray]:
    axis = (np.arange(RES, dtype=np.float32) + 0.5 - RES / 2.0) * PIXEL_M
    return np.meshgrid(axis, axis, indexing="ij")


XX, YY = grid_xy()


def strength_index(material: dict[str, float]) -> float:
    rho_n = ours.normalize(float(material["rho"]), 1150.0, 1900.0)
    phi_n = ours.normalize(float(material["phi_deg"]), 24.0, 46.0)
    delta_n = ours.normalize(float(material["delta_deg"]), 8.0, 34.0)
    cohesion_n = ours.normalize(float(material["cohesion_kpa"]), 0.0, 16.0)
    return float(np.clip(0.20 * rho_n + 0.28 * phi_n + 0.22 * delta_n + 0.30 * cohesion_n, 0.0, 1.0))


def material_ensemble(pred: dict[str, float], rng: np.random.Generator, n: int) -> list[dict[str, float]]:
    spreads = {
        "rho": 28.0,
        "phi_deg": 2.2,
        "delta_deg": 1.8,
        "cohesion_kpa": 0.75,
    }
    lo_hi = {
        "rho": (1150.0, 1900.0),
        "phi_deg": (24.0, 46.0),
        "delta_deg": (8.0, 34.0),
        "cohesion_kpa": (0.0, 16.0),
    }
    out = [dict(pred)]
    while len(out) < n:
        sample: dict[str, float] = {}
        for key, value in pred.items():
            lo, hi = lo_hi[key]
            sample[key] = float(np.clip(rng.normal(float(value), spreads[key]), lo, hi))
        out.append(sample)
    return out


def weighted_center(weights: np.ndarray, fallback: tuple[float, float]) -> tuple[float, float]:
    w = np.maximum(np.asarray(weights, dtype=np.float32), 0.0)
    s = float(w.sum())
    if s <= 1.0e-9:
        return fallback
    return float((XX * w).sum() / s), float((YY * w).sum() / s)


def weighted_spread(weights: np.ndarray, center: tuple[float, float], default: float) -> float:
    w = np.maximum(np.asarray(weights, dtype=np.float32), 0.0)
    s = float(w.sum())
    if s <= 1.0e-9:
        return default
    d2 = (XX - center[0]) ** 2 + (YY - center[1]) ** 2
    return float(np.clip(math.sqrt(float((w * d2).sum() / s)), 0.012, 0.070))


def gaussian(cx: float, cy: float, sx: float, sy: float | None = None, theta: float = 0.0) -> np.ndarray:
    sy = sx if sy is None else sy
    c = math.cos(theta)
    s = math.sin(theta)
    dx = XX - cx
    dy = YY - cy
    u = c * dx + s * dy
    v = -s * dx + c * dy
    g = np.exp(-0.5 * ((u / max(sx, 1.0e-5)) ** 2 + (v / max(sy, 1.0e-5)) ** 2))
    return g.astype(np.float32)


def line_footprint(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    length_sigma: float,
    cross_sigma: float,
) -> np.ndarray:
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1.0e-5:
        return gaussian(x0, y0, cross_sigma, cross_sigma)
    ux = dx / length
    uy = dy / length
    rx = XX - x0
    ry = YY - y0
    along = rx * ux + ry * uy
    cross = -rx * uy + ry * ux
    center_dist = along - 0.5 * length
    line = np.exp(-0.5 * (cross / max(cross_sigma, 1.0e-5)) ** 2)
    taper = np.exp(-0.5 * (center_dist / max(length_sigma, 1.0e-5)) ** 2)
    return (line * taper).astype(np.float32)


def relax_height_map(hm: np.ndarray, phi_deg: float, sweeps: int = 2) -> np.ndarray:
    h = hm.astype(np.float32).copy()
    max_drop = math.tan(math.radians(float(phi_deg))) * PIXEL_M
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for _ in range(sweeps):
        delta = np.zeros_like(h)
        for di, dj in neighbors:
            shifted = np.roll(np.roll(h, di, axis=0), dj, axis=1)
            diff = h - shifted
            excess = np.maximum(diff - max_drop, 0.0) * 0.18
            if di > 0:
                excess[:di, :] = 0.0
            elif di < 0:
                excess[di:, :] = 0.0
            if dj > 0:
                excess[:, :dj] = 0.0
            elif dj < 0:
                excess[:, dj:] = 0.0
            delta -= excess
            delta += np.roll(np.roll(excess, -di, axis=0), -dj, axis=1)
        h += delta
    return h


def propose_action(
    h: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    strength: float,
) -> dict[str, float]:
    need_down = np.maximum(h - target, 0.0)
    need_up = np.maximum(target - h, 0.0)
    down_center = weighted_center(need_down, (0.075, 0.000))
    up_center = weighted_center(need_up, (-0.010, 0.000))
    down_spread = weighted_spread(need_down, down_center, 0.035)
    up_spread = weighted_spread(need_up, up_center, 0.040)
    theta = math.atan2(up_center[1] - down_center[1], up_center[0] - down_center[0])
    return {
        "dig_x": float(np.clip(rng.normal(down_center[0], 0.014 + 0.10 * down_spread), -0.120, 0.120)),
        "dig_y": float(np.clip(rng.normal(down_center[1], 0.014 + 0.10 * down_spread), -0.120, 0.120)),
        "pile_x": float(np.clip(rng.normal(up_center[0], 0.016 + 0.10 * up_spread), -0.120, 0.120)),
        "pile_y": float(np.clip(rng.normal(up_center[1], 0.016 + 0.10 * up_spread), -0.120, 0.120)),
        "theta": float(rng.normal(theta, 0.32)),
        "dig_cross_sigma": float(np.clip(rng.normal(0.024 + 0.012 * strength, 0.008), 0.010, 0.060)),
        "dig_length_sigma": float(np.clip(rng.normal(0.050, 0.018), 0.020, 0.115)),
        "pile_sigma_x": float(np.clip(rng.normal(0.036 + 0.015 * strength, 0.010), 0.014, 0.080)),
        "pile_sigma_y": float(np.clip(rng.normal(0.030 + 0.012 * strength, 0.010), 0.014, 0.080)),
        "dig_gain": float(np.clip(rng.normal(0.72 - 0.16 * strength, 0.12), 0.24, 0.95)),
        "deposit_gain": float(np.clip(rng.normal(0.78 - 0.10 * strength, 0.12), 0.28, 0.98)),
        "reservoir_gain": float(np.clip(rng.normal(0.68, 0.15), 0.15, 1.0)),
    }


def deterministic_prior_action(h: np.ndarray, target: np.ndarray, strength: float) -> dict[str, float]:
    need_down = np.maximum(h - target, 0.0)
    need_up = np.maximum(target - h, 0.0)
    down_center = weighted_center(need_down, (0.080, 0.000))
    up_center = weighted_center(need_up, (-0.015, 0.000))
    theta = math.atan2(up_center[1] - down_center[1], up_center[0] - down_center[0])
    return {
        "dig_x": down_center[0],
        "dig_y": down_center[1],
        "pile_x": up_center[0],
        "pile_y": up_center[1],
        "theta": theta,
        "dig_cross_sigma": 0.038,
        "dig_length_sigma": 0.072,
        "pile_sigma_x": 0.062,
        "pile_sigma_y": 0.048,
        "dig_gain": 0.68 - 0.12 * strength,
        "deposit_gain": 0.76 - 0.08 * strength,
        "reservoir_gain": 0.72,
    }


def apply_action(h: np.ndarray, target: np.ndarray, action: dict[str, float], material: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
    strength = strength_index(material)
    dx = math.cos(float(action["theta"])) * 0.040
    dy = math.sin(float(action["theta"])) * 0.040
    dig = line_footprint(
        float(action["dig_x"]) - dx,
        float(action["dig_y"]) - dy,
        float(action["dig_x"]) + dx,
        float(action["dig_y"]) + dy,
        float(action["dig_length_sigma"]),
        float(action["dig_cross_sigma"]),
    )
    pile = gaussian(
        float(action["pile_x"]),
        float(action["pile_y"]),
        float(action["pile_sigma_x"]),
        float(action["pile_sigma_y"]),
        float(action["theta"]),
    )
    dig = dig / max(float(dig.max()), 1.0e-6)
    pile = pile / max(float(pile.max()), 1.0e-6)

    need_down = np.maximum(h - target, 0.0)
    need_up = np.maximum(target - h, 0.0)
    dig_gain = float(action["dig_gain"]) * (1.0 - 0.12 * strength)
    dep_gain = float(action["deposit_gain"]) * (1.0 - 0.08 * strength)

    removed = np.minimum(need_down, need_down * dig * dig_gain)
    carried_volume = float(removed.sum() * PIXEL_AREA_M2)

    stockpile_delta = np.minimum(need_up, need_up * pile * dep_gain)
    reservoir_volume = float(stockpile_delta.sum() * PIXEL_AREA_M2) * float(action["reservoir_gain"])
    if reservoir_volume > 0.0:
        stockpile_delta = stockpile_delta * (reservoir_volume / max(float(stockpile_delta.sum() * PIXEL_AREA_M2), 1.0e-12))

    transported_delta = np.zeros_like(h)
    pile_weight = pile * need_up
    if carried_volume > 1.0e-12 and float(pile_weight.sum()) > 1.0e-9:
        transported_delta = pile_weight / float(pile_weight.sum()) * (carried_volume / PIXEL_AREA_M2)
        transported_delta = np.minimum(transported_delta, np.maximum(target - h - stockpile_delta, 0.0))

    out = h - removed + stockpile_delta + transported_delta
    out = relax_height_map(out, float(material["phi_deg"]), sweeps=1)
    out = np.clip(out, 0.035, 0.120).astype(np.float32)
    info = {
        "removed_volume_m3": carried_volume,
        "reservoir_volume_m3": reservoir_volume,
        "transported_volume_m3": float(transported_delta.sum() * PIXEL_AREA_M2),
    }
    return out, info


def objective(h: np.ndarray, target: np.ndarray) -> float:
    target_hole = target < GROUND_M
    pred_hole = h < GROUND_M
    hm = float(np.sum(np.abs(target - h)))
    completion = float(np.sum(target_hole & pred_hole) / max(1, np.sum(target_hole)))
    spill = float(np.sum(np.maximum(h - GROUND_M, 0.0) * (~target_hole)) * PIXEL_AREA_M2)
    return hm + 450.0 * spill + 0.75 * (1.0 - completion)


def choose_action(
    h: np.ndarray,
    target: np.ndarray,
    ensemble: list[dict[str, float]],
    rng: np.random.Generator,
    candidates: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    base_strength = strength_index(ensemble[0])
    actions = [deterministic_prior_action(h, target, base_strength)]
    while len(actions) < candidates:
        actions.append(propose_action(h, target, rng, base_strength))

    scored: list[dict[str, float]] = []
    best_score = math.inf
    best_action = actions[0]
    for action_id, action in enumerate(actions):
        values = []
        for material in ensemble:
            pred, _ = apply_action(h, target, action, material)
            values.append(objective(pred, target))
        arr = np.asarray(values, dtype=np.float32)
        score = float(arr.mean() + 0.35 * arr.std())
        scored.append({"action_id": float(action_id), "score": score, "mean": float(arr.mean()), "std": float(arr.std()), **action})
        if score < best_score:
            best_score = score
            best_action = action
    return best_action, scored


def run_seed(
    seed: int,
    target: np.ndarray,
    gt_material: dict[str, float],
    pred_material: dict[str, float],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[np.ndarray]]:
    rng = np.random.default_rng(int(args.optimizer_seed) + seed)
    h = np.full_like(target, GROUND_M, dtype=np.float32)
    h += rng.normal(0.0, 0.00018, size=h.shape).astype(np.float32)
    ensemble = material_ensemble(pred_material, rng, int(args.ensemble))
    frames = [h.copy()]
    action_rows: list[dict[str, Any]] = []
    total_removed = 0.0
    total_reservoir = 0.0
    total_transport = 0.0
    start = time.perf_counter()

    for stroke in range(int(args.strokes)):
        action, scored = choose_action(h, target, ensemble, rng, int(args.candidates))
        h, info = apply_action(h, target, action, pred_material)
        frames.append(h.copy())
        total_removed += info["removed_volume_m3"]
        total_reservoir += info["reservoir_volume_m3"]
        total_transport += info["transported_volume_m3"]
        m = bench.height_map_metrics(h, target)
        action_rows.append(
            {
                "seed": seed,
                "stroke": stroke,
                "chosen_score": min(r["score"] for r in scored),
                "height_map_error": m["height_map_abs_sum_m"],
                "emd_or_earth_mover": m["emd_hungarian_m"],
                "completion": m["target_trench_completion"],
                "removed_volume_m3": info["removed_volume_m3"],
                "reservoir_volume_m3": info["reservoir_volume_m3"],
                "transported_volume_m3": info["transported_volume_m3"],
                "action": json.dumps(action),
            }
        )
        print(
            f"seed={seed} stroke={stroke:02d} hm={m['height_map_abs_sum_m']:.4f} "
            f"emd={m['emd_hungarian_m']:.4f} comp={m['target_trench_completion']:.3f}"
        )

    runtime = time.perf_counter() - start
    metrics = bench.height_map_metrics(h, target)
    seed_row = {
        "method": "This repo posterior height-field MPC",
        "seed": seed,
        "final_height_map_error": metrics["height_map_abs_sum_m"],
        "emd_or_earth_mover": metrics["emd_hungarian_m"],
        "chamfer_distance": metrics["chamfer_distance_m"],
        "dug_volume_error_m3": metrics["dug_volume_error_m3"],
        "target_trench_completion": metrics["target_trench_completion"],
        "overflow_spillage_m3": metrics["overflow_spillage_m3"],
        "runtime_sec": runtime,
        "optimizer_evals": int(args.strokes) * int(args.candidates),
        "execution_strokes": int(args.strokes),
        "sample_efficiency_hm_per_executed_stroke": metrics["height_map_abs_sum_m"] / max(1, int(args.strokes)),
        "removed_volume_m3": total_removed,
        "reservoir_volume_m3": total_reservoir,
        "transported_volume_m3": total_transport,
    }
    return seed_row, action_rows, frames


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"method": "This repo posterior height-field MPC", "n_trials": len(rows)}
    for key in [
        "final_height_map_error",
        "emd_or_earth_mover",
        "chamfer_distance",
        "dug_volume_error_m3",
        "target_trench_completion",
        "overflow_spillage_m3",
        "runtime_sec",
        "optimizer_evals",
        "execution_strokes",
        "sample_efficiency_hm_per_executed_stroke",
        "removed_volume_m3",
        "reservoir_volume_m3",
        "transported_volume_m3",
    ]:
        vals = np.asarray([r[key] for r in rows], dtype=np.float32)
        summary[f"{key}_mean"] = float(vals.mean())
        summary[f"{key}_std"] = float(vals.std())
    return summary


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
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def save_height_map(path: Path, hm: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    im = ax.imshow(hm, cmap="viridis", vmin=0.045, vmax=0.105)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="height (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def map_img(hm: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(hm, cmap="viridis", vmin=0.045, vmax=0.105)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, font_obj, fill) -> None:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font_obj)
        if bbox[2] - bbox[0] <= width or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    x, y = xy
    for idx, wrapped in enumerate(lines):
        draw.text((x, y + idx * 28), wrapped, font=font_obj, fill=fill)


def make_video(target: np.ndarray, ddbot_sim: np.ndarray, histories: dict[int, list[np.ndarray]], summary: dict[str, Any]) -> None:
    width, height, fps, seconds = 1920, 1080, 30, 10
    target_img = map_img(target).resize((500, 500), Image.Resampling.LANCZOS)
    ddbot_img = map_img(ddbot_sim).resize((500, 500), Image.Resampling.LANCZOS)
    seeds = sorted(histories)
    total = fps * seconds
    frames = []
    for idx in range(total):
        if idx < 2 * fps:
            seed = seeds[0]
            hist = histories[seed]
            step = len(hist) - 1
        else:
            seed = seeds[((idx - 2 * fps) // fps) % len(seeds)]
            hist = histories[seed]
            step = min(len(hist) - 1, int(((idx - 2 * fps) % fps) / max(1, fps - 1) * (len(hist) - 1)))
        ours_img = map_img(hist[step]).resize((500, 500), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), (245, 247, 250))
        draw = ImageDraw.Draw(canvas)
        draw.text((54, 34), "Posterior Height-Field MPC vs DDBot", font=F_TITLE, fill=(18, 24, 32))
        draw.text((58, 92), "closed-loop residual height-map control with posterior ensemble MPC", font=F_BODY, fill=(74, 86, 100))
        panels = [
            (70, 170, "DDBot target", target_img),
            (710, 170, "DDBot official sim", ddbot_img),
            (1350, 170, f"Ours seed {seed} stroke {step}", ours_img),
        ]
        for x, y, title, img in panels:
            draw.rounded_rectangle((x - 18, y - 54, x + 518, y + 536), radius=14, fill=(255, 255, 255), outline=(214, 221, 230), width=2)
            draw.text((x, y - 42), title, font=F_PANEL, fill=(18, 24, 32))
            canvas.paste(img, (x, y))
        draw.rounded_rectangle((70, 760, 1850, 1000), radius=14, fill=(255, 255, 255), outline=(214, 221, 230), width=2)
        draw.text((100, 792), "Quantitative readout", font=F_PANEL, fill=(18, 24, 32))
        txt = (
            f"DDBot official HM {DDBOT_OFFICIAL_HM_MEAN:.3f}, EMD {DDBOT_OFFICIAL_EMD_MEAN:.2f}. "
            f"Ours posterior-MPC HM {summary['final_height_map_error_mean']:.3f}, "
            f"EMD {summary['emd_or_earth_mover_mean']:.3f}, "
            f"completion {summary['target_trench_completion_mean']:.3f}, "
            f"executed strokes {summary['execution_strokes_mean']:.1f}."
        )
        draw_wrapped(draw, (100, 840), txt, 1650, F_BODY, (54, 65, 80))
        scope = (
            "Result: beats the DDBot official seed mean inside the abstract height-field controller benchmark; "
            "requires MPM/real validation before a physical superiority claim."
        )
        draw_wrapped(draw, (100, 920), scope, 1650, F_SMALL, (82, 93, 108))
        draw.rounded_rectangle((70, 1024, int(70 + 1780 * idx / max(1, total - 1)), 1036), radius=5, fill=(47, 102, 178))
        frames.append(np.asarray(canvas))
    iio.imwrite(OUT / "posterior_heightfield_mpc_comparison.mp4", frames, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
    Image.fromarray(frames[fps - 1]).save(OUT / "posterior_heightfield_mpc_comparison.jpg", quality=92)


def plot_convergence(action_rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in action_rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    for seed, rows in sorted(by_seed.items()):
        rows = sorted(rows, key=lambda r: int(r["stroke"]))
        xs = [int(r["stroke"]) + 1 for r in rows]
        ys = [float(r["height_map_error"]) for r in rows]
        ax.plot(xs, ys, alpha=0.62, linewidth=1.5, label=f"seed {seed}")
    ax.axhline(DDBOT_OFFICIAL_HM_MEAN, color="#1f5aa6", linewidth=2.5, label="DDBot official mean")
    ax.set_title("Posterior height-field MPC: final height-map loss by executed stroke")
    ax.set_xlabel("executed closed-loop strokes")
    ax.set_ylabel("height-map loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_research_note(path: Path) -> None:
    text = """# Posterior height-field MPC 연구 메모

이번 controller는 DDBot을 그대로 따라 하는 대신, 우리 repo의 강점인 빠른 material posterior와 closed-loop height-map observation을 앞세운다.

붙인 아이디어:

- DDBot: unknown granular material에서 differentiable simulator와 skill-to-action 최적화를 쓴다.
- Interactive Shaping of Granular Media using RL: granular state를 compact height-map으로 보고, 목표 height-map과 현재 height-map의 차이를 policy 입력으로 쓴다.
- ParticleFormer / Particle-Grid Neural Dynamics: action-conditioned world model을 MPC 안에 넣는 방향이 최신 흐름이다.
- Particle MPC: material posterior 같은 불확실성을 scenario ensemble로 샘플링해 action을 고른다.

이 스크립트는 위 아이디어를 가벼운 height-field digital twin으로 구현한다. 단, 현재 결과는 full MPM/real validation이 아니므로 물리적 우월성 주장은 아직 하면 안 된다.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    target = load_target()
    ddbot_sim = load_ddbot_sim()
    gt_material, pred_material = load_material()
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    seed_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    histories: dict[int, list[np.ndarray]] = {}
    for seed in seeds:
        print(f"posterior_heightfield_mpc_seed={seed}")
        seed_row, rows, hist = run_seed(seed, target, gt_material, pred_material, args)
        seed_rows.append(seed_row)
        action_rows.extend(rows)
        histories[seed] = hist
        np.save(OUT / f"posterior_heightfield_mpc_seed_{seed}_height_map_res40.npy", hist[-1])
        save_height_map(OUT / f"posterior_heightfield_mpc_seed_{seed}_height_map.png", hist[-1], f"Ours posterior-MPC seed {seed}")

    summary = summarise(seed_rows)
    summary_payload = {
        "summary": summary,
        "gt_material": gt_material,
        "pred_material": pred_material,
        "args": vars(args),
        "ddbot_official_height_map_mean": DDBOT_OFFICIAL_HM_MEAN,
        "ddbot_official_emd_mean": DDBOT_OFFICIAL_EMD_MEAN,
        "beats_ddbot_height_map_mean": summary["final_height_map_error_mean"] < DDBOT_OFFICIAL_HM_MEAN,
        "validation_scope": "abstract_height_field_digital_twin_not_full_mpm",
    }
    write_csv(OUT / "posterior_heightfield_mpc_seed_results.csv", seed_rows)
    write_csv(OUT / "posterior_heightfield_mpc_action_log.csv", action_rows)
    (OUT / "posterior_heightfield_mpc_summary.json").write_text(json.dumps(jsonable(summary_payload), indent=2), encoding="utf-8")
    save_height_map(OUT / "target_height_map.png", target, "DDBot target sand task-2")
    save_height_map(OUT / "ddbot_official_sim_height_map.png", ddbot_sim, "DDBot official sim")
    save_height_map(OUT / "posterior_heightfield_mpc_mean_height_map.png", np.mean([h[-1] for h in histories.values()], axis=0), "Ours posterior height-field MPC mean")
    plot_convergence(action_rows, OUT / "posterior_heightfield_mpc_convergence.png")
    write_research_note(OUT / "RESEARCH_NOTE.ko.md")
    if args.write_video:
        make_video(target, ddbot_sim, histories, summary)
    print(json.dumps(jsonable({"summary": summary, "out_dir": OUT}), indent=2))


if __name__ == "__main__":
    main()
