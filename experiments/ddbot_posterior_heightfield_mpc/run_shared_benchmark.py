"""Run a frozen DDBot-vs-this-repo comparison benchmark.

This script fixes the comparison quantities that matter for a defensible
DDBot comparison:

- DDBot sand task-2 target height map, 40x40, 0.24 m x 0.24 m.
- Seeds 0..4.
- DDBot official skill-optimisation artifacts as the DDBot side.
- This repository's MPM excavation rollout evaluated on the same height-map
  convention.

The DDBot runtime is not required here. DDBot's official seed-level JSON
artifacts are read from the cloned repository, and the LFS target height map is
downloaded through GitHub's media endpoint when it is not present locally.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "ddbot_tro2025_comparison"
DDBOT_REPO = EXP / "third_party" / "ddbot"
OUT = EXP / "results" / "shared_benchmark"
OURS_MODULE = ROOT / "scripts" / "render_excavation_policy_compare.py"

DDBOT_COMMIT = "e642f7c73f37539c21161bd29669fa8d91912b88"
MATERIAL = "sand"
TASK_ID = 2
CASE = "d5e6-task-2-hm-ls-demo-lr0.03"
RES = 40
HEIGHT_MAP_SIZE_M = 0.24
PIXEL_AREA_M2 = (HEIGHT_MAP_SIZE_M / RES) ** 2
SAND_GROUND_LEVEL_M = 0.073

TARGET_URL = (
    "https://media.githubusercontent.com/media/"
    "IanYangChina/DDBot-IEEE-TRO-2025/main/"
    "data/task-targets/sand/pcd_2_cropped_norm_z_aligned_height_map-res40.npy"
)
DDBOT_SIM_URL = (
    "https://media.githubusercontent.com/media/"
    "IanYangChina/DDBot-IEEE-TRO-2025/main/"
    "archive/render_outputs/abs2_sand/d5e6-task-2-hm-ls-demo-lr0.03/"
    "pcd_0_cropped_norm_z_aligned_height_map-res40.npy"
)

FONT_REGULAR = Path("C:/Windows/Fonts/malgun.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/malgunbd.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(46, True)
F_PANEL = font(28, True)
F_BODY = font(21)
F_SMALL = font(17)


def load_ours_module():
    spec = importlib.util.spec_from_file_location("ours_excavation_policy", OURS_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {OURS_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--frames", type=int, default=84)
    parser.add_argument("--substeps-per-frame", type=int, default=34)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--force-scale", type=float, default=0.0012)
    parser.add_argument("--ours-config", type=Path, default=ROOT / "configs/rendering/excavation_policy_compare.json")
    parser.add_argument("--property-csv", type=Path, default=ROOT / "outputs/online_mohr_coulomb_bestval_quick/rollout_predictions.csv")
    parser.add_argument("--row", default="last")
    parser.add_argument("--skip-ours", action="store_true")
    parser.add_argument("--write-video", action="store_true")
    return parser.parse_args()


def git_show_text(path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(DDBOT_REPO), "show", f"HEAD:{path}"],
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def git_show_json(path: str) -> dict[str, Any]:
    return json.loads(git_show_text(path))


def ensure_download(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1024:
        return
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
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


def load_ddbot_seed_rows(seeds: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    curves = {}
    for seed in seeds:
        best_path = f"results/skill-parameter-optimization/{MATERIAL}/{CASE}/seed-{seed}/best_loss.json"
        raw_path = f"results/skill-parameter-optimization/{MATERIAL}/{CASE}/seed-{seed}/raw_data.json"
        best = git_show_json(best_path)
        raw = git_show_json(raw_path)
        hm_curve = raw.get("Loss", {}).get("height_map_loss", [])
        emd_curve = raw.get("Loss", {}).get("emd_loss", [])
        curves[str(seed)] = {
            "height_map_loss": hm_curve,
            "emd_loss": emd_curve,
        }
        rows.append(
            {
                "method": "DDBot official",
                "seed": seed,
                "material": MATERIAL,
                "task_id": TASK_ID,
                "target_source": "official_dbot_lfs_target",
                "initial_bed": "DDBot official fixed bed (artifact; not re-run)",
                "trajectory_budget": "official 5-parameter digging skill",
                "rollout_horizon_sec": None,
                "rollout_frames": None,
                "optimizer_evals": len(hm_curve) if hm_curve else int(best.get("Step", 0)) + 1,
                "best_step": best.get("Step"),
                "final_height_map_error": best.get("Loss", {}).get("height_map_loss"),
                "emd_or_earth_mover": best.get("Loss", {}).get("emd_loss"),
                "chamfer_distance": None,
                "dug_volume_error_m3": None,
                "target_trench_completion": None,
                "overflow_spillage_m3": None,
                "force_or_reaction_mismatch": None,
                "peak_reaction_norm": None,
                "runtime_sec": None,
                "sample_efficiency_hm_per_eval": (
                    float(best.get("Loss", {}).get("height_map_loss")) / max(1, len(hm_curve))
                    if hm_curve
                    else None
                ),
                "status": "official_json_artifact",
            }
        )
    aggregate = git_show_json(f"results/skill-parameter-optimization/{MATERIAL}/{CASE}/best_loss.json")
    return rows, {"seed_curves": curves, "aggregate_best": aggregate}


def surface_points_from_height_map(hm: np.ndarray) -> np.ndarray:
    xs = (np.arange(RES, dtype=np.float32) + 0.5) * HEIGHT_MAP_SIZE_M / RES - HEIGHT_MAP_SIZE_M / 2.0
    ys = (np.arange(RES, dtype=np.float32) + 0.5) * HEIGHT_MAP_SIZE_M / RES - HEIGHT_MAP_SIZE_M / 2.0
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), hm.astype(np.float32).ravel()])


def chamfer(points_a: np.ndarray, points_b: np.ndarray) -> float:
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    d_ab, _ = tree_b.query(points_a, k=1)
    d_ba, _ = tree_a.query(points_b, k=1)
    return float(0.5 * (np.mean(d_ab) + np.mean(d_ba)))


def emd_hungarian(points_a: np.ndarray, points_b: np.ndarray) -> float:
    diff = points_a[:, None, :] - points_b[None, :, :]
    mat = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32))
    row, col = linear_sum_assignment(mat)
    return float(mat[row, col].sum())


def fill_missing_height_map(hm: np.ndarray, fill: float) -> np.ndarray:
    out = np.asarray(hm, dtype=np.float32).copy()
    mask = np.isfinite(out)
    if not mask.any():
        return np.full_like(out, fill, dtype=np.float32)
    out[~mask] = fill
    return out


def smooth_height_map(hm: np.ndarray) -> np.ndarray:
    src = np.asarray(hm, dtype=np.float32)
    out = src.copy()
    for i in range(1, src.shape[0] - 1):
        for j in range(1, src.shape[1] - 1):
            out[i, j] = float(np.mean(src[i - 1 : i + 2, j - 1 : j + 2]))
    return out


def height_map_from_positions(
    pos: np.ndarray,
    center_xy: tuple[float, float],
    size_m: float = HEIGHT_MAP_SIZE_M,
    res: int = RES,
) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    xmin = center_xy[0] - size_m / 2.0
    ymin = center_xy[1] - size_m / 2.0
    u = np.floor((pos[:, 0] - xmin) / size_m * res).astype(np.int32)
    v = np.floor((pos[:, 1] - ymin) / size_m * res).astype(np.int32)
    valid = (u >= 0) & (u < res) & (v >= 0) & (v < res)
    hm = np.full((res, res), np.nan, dtype=np.float32)
    for uu, vv, zz in zip(u[valid], v[valid], pos[valid, 2], strict=False):
        for du, dv in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            ii = int(uu + du)
            jj = int(vv + dv)
            if 0 <= ii < res and 0 <= jj < res:
                if not np.isfinite(hm[ii, jj]) or zz > hm[ii, jj]:
                    hm[ii, jj] = float(zz)
    return hm


def calibrate_ours_to_ddbot_scale(ours_hm: np.ndarray, initial_hm: np.ndarray) -> np.ndarray:
    initial_filled = fill_missing_height_map(initial_hm, float(np.nanmedian(initial_hm)))
    ours_filled = fill_missing_height_map(ours_hm, float(np.nanmedian(initial_hm)))
    initial_filled = smooth_height_map(initial_filled)
    ours_filled = smooth_height_map(ours_filled)
    return (SAND_GROUND_LEVEL_M + (ours_filled - initial_filled)).astype(np.float32)


def height_map_metrics(
    hm: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray | None = None,
    official_height_map_error: float | None = None,
    official_emd: float | None = None,
) -> dict[str, float | None]:
    hm = np.asarray(hm, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    target_points = surface_points_from_height_map(target)
    hm_points = surface_points_from_height_map(hm)
    target_hole = target < SAND_GROUND_LEVEL_M
    pred_hole = hm < SAND_GROUND_LEVEL_M
    target_volume = float(np.sum(np.maximum(SAND_GROUND_LEVEL_M - target, 0.0)) * PIXEL_AREA_M2)
    pred_volume = float(np.sum(np.maximum(SAND_GROUND_LEVEL_M - hm, 0.0)) * PIXEL_AREA_M2)
    overflow = float(np.sum(np.maximum(hm - SAND_GROUND_LEVEL_M, 0.0) * (~target_hole)) * PIXEL_AREA_M2)
    completion = float(np.sum(target_hole & pred_hole) / max(1, np.sum(target_hole)))
    hm_abs = float(np.sum(np.abs(target - hm)))
    metrics = {
        "height_map_abs_sum_m": hm_abs,
        "height_map_mae_m": float(np.mean(np.abs(target - hm))),
        "height_map_rmse_m": float(np.sqrt(np.mean((target - hm) ** 2))),
        "chamfer_distance_m": chamfer(hm_points, target_points),
        "emd_hungarian_m": emd_hungarian(hm_points, target_points),
        "dug_volume_m3": pred_volume,
        "target_dug_volume_m3": target_volume,
        "dug_volume_error_m3": abs(pred_volume - target_volume),
        "target_trench_completion": completion,
        "overflow_spillage_m3": overflow,
        "official_height_map_error": official_height_map_error,
        "official_emd": official_emd,
    }
    if initial is not None:
        initial = np.asarray(initial, dtype=np.float32)
        metrics["surface_change_l1_m"] = float(np.sum(np.abs(hm - initial)))
    return metrics


def run_ours_seed(
    seed: int,
    args: argparse.Namespace,
    target_hm: np.ndarray,
    ours_mod: Any,
    config: dict[str, Any],
    row: dict[str, str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[float]]:
    gt_material = ours_mod.material_from_row(row, "target")
    pred_material = ours_mod.material_from_row(row, "pred")
    mapping = dict(config.get("material_mapping", {}))
    mpm_base = dict(config.get("mpm", {}))
    mpm_base["seed"] = seed
    mpm_cfg = ours_mod.material_to_mpm_config(gt_material, mpm_base, mapping)
    model_policy = ours_mod.property_aware_policy(pred_material, dict(config.get("model_policy", {})))

    solver = ours_mod.make_solver(mpm_cfg, args.device)
    initial_pos = solver.positions()
    center_xy = (
        0.5 * (float(model_policy.get("x_start", 0.225)) + float(model_policy.get("x_end", 0.79))),
        float(model_policy.get("y", 0.28)),
    )
    initial_raw = height_map_from_positions(initial_pos, center_xy)
    initial_hm = calibrate_ours_to_ddbot_scale(initial_raw, initial_raw)

    sim_t = 0.0
    force_history: list[float] = []
    work = 0.0
    path_length = 0.0
    start = time.perf_counter()
    for _frame_id in range(int(args.frames)):
        raw_force = np.zeros(6, dtype=np.float32)
        frame_path = 0.0
        for _ in range(int(args.substeps_per_frame)):
            tool = ours_mod.excavation_state(sim_t, mpm_cfg.dt, model_policy)
            raw_force += solver.step(tool, substeps=1)
            step_path = float(np.linalg.norm(tool.velocity[:3])) * mpm_cfg.dt
            frame_path += step_path
            sim_t += mpm_cfg.dt
        raw_force /= max(1, int(args.substeps_per_frame))
        display_force = raw_force * float(args.force_scale)
        f_norm = float(np.linalg.norm(display_force[:3]))
        force_history.append(f_norm)
        path_length += frame_path
        work += f_norm * frame_path
    runtime = time.perf_counter() - start

    final_pos = solver.positions()
    final_raw = height_map_from_positions(final_pos, center_xy)
    final_hm = calibrate_ours_to_ddbot_scale(final_raw, initial_raw)
    metrics = height_map_metrics(final_hm, target_hm, initial=initial_hm)
    row_out = {
        "method": "This repo MPM posterior policy",
        "seed": seed,
        "material": MATERIAL,
        "task_id": TASK_ID,
        "target_source": "official_dbot_lfs_target",
        "initial_bed": "same repo MPM block bed, seed matched across trials",
        "trajectory_budget": "one posterior-conditioned blade rollout",
        "rollout_horizon_sec": sim_t,
        "rollout_frames": int(args.frames),
        "optimizer_evals": 1,
        "best_step": 0,
        "final_height_map_error": metrics["height_map_abs_sum_m"],
        "emd_or_earth_mover": metrics["emd_hungarian_m"],
        "chamfer_distance": metrics["chamfer_distance_m"],
        "dug_volume_error_m3": metrics["dug_volume_error_m3"],
        "target_trench_completion": metrics["target_trench_completion"],
        "overflow_spillage_m3": metrics["overflow_spillage_m3"],
        "force_or_reaction_mismatch": None,
        "peak_reaction_norm": float(max(force_history) if force_history else 0.0),
        "runtime_sec": runtime,
        "sample_efficiency_hm_per_eval": metrics["height_map_abs_sum_m"],
        "status": "local_mpm_run",
        "path_length_m": path_length,
        "work_proxy": work,
        "device": solver.device,
        "mpm_config": json.dumps(jsonable(asdict(mpm_cfg))),
        "model_policy": json.dumps(jsonable(model_policy)),
    }
    return row_out, final_hm, initial_hm, force_history


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["method"]), []).append(row)
    summary = []
    for method, method_rows in groups.items():
        out: dict[str, Any] = {"method": method, "n_trials": len(method_rows)}
        for key in [
            "final_height_map_error",
            "emd_or_earth_mover",
            "chamfer_distance",
            "dug_volume_error_m3",
            "target_trench_completion",
            "overflow_spillage_m3",
            "peak_reaction_norm",
            "runtime_sec",
            "optimizer_evals",
            "sample_efficiency_hm_per_eval",
        ]:
            values = np.asarray([r.get(key) for r in method_rows if r.get(key) is not None], dtype=np.float32)
            if values.size:
                out[f"{key}_mean"] = float(values.mean())
                out[f"{key}_std"] = float(values.std())
            else:
                out[f"{key}_mean"] = None
                out[f"{key}_std"] = None
        summary.append(out)
    return summary


def save_height_map_image(path: Path, hm: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    im = ax.imshow(hm, cmap="viridis", vmin=0.045, vmax=0.105)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="height (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(summary: list[dict[str, Any]], path: Path) -> None:
    methods = [r["method"] for r in summary]
    def label(method: str) -> str:
        if "sim height-map" in method:
            return "DDBot sim"
        if "DDBot" in method:
            return "DDBot seeds"
        if "Flat" in method:
            return "Flat"
        return "Ours"

    labels = [label(m) for m in methods]
    hm = [r["final_height_map_error_mean"] for r in summary]
    emd = [r["emd_or_earth_mover_mean"] for r in summary]
    completion = [r["target_trench_completion_mean"] for r in summary]
    steps = [r["optimizer_evals_mean"] for r in summary]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.8))
    colors = ["#2f80ed", "#68a6ff", "#a9adb5", "#2aa876", "#e8a33a"][: len(labels)]
    axs[0].bar(labels, hm, color=colors)
    axs[0].set_title("Final height-map error")
    axs[0].set_ylabel("sum |h - h*|")
    axs[1].bar(labels, emd, color=colors)
    axs[1].set_title("EMD / Hungarian")
    axs[2].bar(labels, [np.nan if v is None else v for v in completion], color=colors)
    axs[2].set_title("Target trench completion")
    axs[2].set_ylim(0, 1.05)
    axs[3].bar(labels, steps, color=colors)
    axs[3].set_title("Optimizer evals")
    for ax in axs:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle("Shared DDBot sand task-2 benchmark", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(aux: dict[str, Any], ours_rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    curves = aux["ddbot"].get("seed_curves", {})
    for seed, curve in curves.items():
        vals = curve.get("height_map_loss", [])
        if vals:
            ax.plot(np.arange(len(vals)), vals, color="#2f80ed", alpha=0.45, linewidth=1.5)
    if curves:
        max_len = max(len(c.get("height_map_loss", [])) for c in curves.values())
        arr = np.full((len(curves), max_len), np.nan, dtype=np.float32)
        for i, curve in enumerate(curves.values()):
            vals = curve.get("height_map_loss", [])
            arr[i, : len(vals)] = vals
        ax.plot(np.nanmean(arr, axis=0), color="#124f9c", linewidth=3, label="DDBot official mean")
    ours_vals = [r["final_height_map_error"] for r in ours_rows]
    if ours_vals:
        ax.scatter([1] * len(ours_vals), ours_vals, color="#2aa876", s=45, label="Ours final rollout")
        ax.axhline(np.mean(ours_vals), color="#2aa876", linewidth=2, linestyle="--", label="Ours mean")
    ax.set_title("Height-map loss vs optimization/evaluation count")
    ax.set_xlabel("trajectory evaluations")
    ax.set_ylabel("height-map loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_report(summary: list[dict[str, Any]], rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    ddbot = next((r for r in summary if "DDBot" in r["method"]), None)
    ours = next((r for r in summary if "This repo" in r["method"]), None)
    lines = [
        "# DDBot shared benchmark 결과",
        "",
        "## 고정한 조건",
        "",
        f"- material/task: `{MATERIAL} task-{TASK_ID}`",
        f"- target: DDBot official `pcd_2_cropped_norm_z_aligned_height_map-res40.npy`",
        f"- height-map: `{RES}x{RES}`, `{HEIGHT_MAP_SIZE_M}m x {HEIGHT_MAP_SIZE_M}m`, pixel `{HEIGHT_MAP_SIZE_M / RES:.3f}m`",
        f"- seed set: `{protocol['seeds']}`",
        "- metric: DDBot convention에 맞춘 height-map absolute sum, EMD/Hungarian, Chamfer, dug-volume error, target trench completion, overflow/spillage",
        "- DDBot side: simulator 재실행이 아니라 공식 repo의 seed별 `best_loss.json` / `raw_data.json` artifact",
        "- ours side: 이 repo의 MPM posterior-conditioned excavation rollout을 seed별로 새로 실행",
        "",
        "## 요약",
        "",
            "| Method | Trials | Final HM error | EMD/Hungarian | Chamfer | Completion | Spillage | Runtime | Optimizer evals |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['n_trials']} | "
            f"{fmt(row['final_height_map_error_mean'])} ± {fmt(row['final_height_map_error_std'])} | "
            f"{fmt(row['emd_or_earth_mover_mean'])} ± {fmt(row['emd_or_earth_mover_std'])} | "
            f"{fmt(row['chamfer_distance_mean'])} | "
            f"{fmt(row['target_trench_completion_mean'])} | "
            f"{fmt(row['overflow_spillage_m3_mean'])} | "
            f"{fmt(row['runtime_sec_mean'])} | "
            f"{fmt(row['optimizer_evals_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "이 실험은 드디어 같은 DDBot target height-map을 기준으로 묶었다. 다만 아직 DDBot simulator를 로컬에서 재실행한 것은 아니므로, DDBot의 runtime/force/spillage는 공식 artifact에 없는 항목은 비워 두었다.",
        ]
    )
    if ddbot and ours:
        d_hm = ddbot["final_height_map_error_mean"]
        o_hm = ours["final_height_map_error_mean"]
        if d_hm is not None and o_hm is not None:
            ratio = float(o_hm) / max(float(d_hm), 1.0e-9)
            lines.append(
                f"현재 frozen target 기준에서 ours의 height-map error는 DDBot 공식 artifact 평균 대비 `{ratio:.2f}x`다. "
                "이 숫자는 성능 주장용으로 바로 쓰기보다는, DDBot task를 우리 MPM으로 제대로 끌고 왔을 때 생기는 gap을 보는 baseline으로 보는 게 맞다."
            )
    lines.extend(
        [
            "",
            "## 산출물",
            "",
            "- `shared_benchmark_seed_results.csv`",
            "- `shared_benchmark_summary.csv`",
            "- `shared_benchmark_protocol.json`",
            "- `shared_benchmark_summary.png`",
            "- `shared_benchmark_convergence.png`",
            "- `target_height_map.png`",
            "- `ddbot_official_sim_height_map.png`",
            "- `ours_mean_height_map.png`",
            "",
            "## 남은 엄밀성 이슈",
            "",
            "DDBot force trace, runtime, local simulator failure/abort 정보는 현재 공개 artifact에 없다. 이 항목까지 완전히 맞추려면 DDBot Python/Taichi 환경을 별도로 살려서 `run_so_abs2.py`를 같은 seed로 재실행해야 한다.",
        ]
    )
    (OUT / "SHARED_BENCHMARK_REPORT.ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "NA"
    if abs(value) < 1.0e-3 and value != 0:
        return f"{value:.2e}"
    return f"{value:.4g}"


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


def map_to_rgb(hm: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(hm, cmap="viridis", vmin=0.045, vmax=0.105)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_video(target: np.ndarray, ddbot_sim: np.ndarray, ours_maps: list[np.ndarray], summary: list[dict[str, Any]]) -> None:
    if not ours_maps:
        return
    width, height, fps, seconds = 1920, 1080, 30, 10
    target_img = map_to_rgb(target).resize((500, 500), Image.Resampling.LANCZOS)
    ddbot_img = map_to_rgb(ddbot_sim).resize((500, 500), Image.Resampling.LANCZOS)
    ours_imgs = [map_to_rgb(hm).resize((500, 500), Image.Resampling.LANCZOS) for hm in ours_maps]
    ddbot = next((r for r in summary if "DDBot" in r["method"]), {})
    ours = next((r for r in summary if "This repo" in r["method"]), {})
    frames = []
    total = fps * seconds
    for idx in range(total):
        canvas = Image.new("RGB", (width, height), (245, 247, 250))
        draw = ImageDraw.Draw(canvas)
        draw.text((54, 34), "Shared Benchmark: DDBot Sand Task-2", font=F_TITLE, fill=(18, 24, 32))
        draw.text((58, 92), "Same target height-map, seed set, and 40x40 DDBot metric convention", font=F_BODY, fill=(74, 86, 100))
        panels = [
            (70, 170, "Target", target_img),
            (710, 170, "DDBot official sim", ddbot_img),
            (1350, 170, f"Ours seed {idx // fps % len(ours_imgs)}", ours_imgs[idx // fps % len(ours_imgs)]),
        ]
        for x, y, title, img in panels:
            draw.rounded_rectangle((x - 18, y - 54, x + 518, y + 536), radius=14, fill=(255, 255, 255), outline=(214, 221, 230), width=2)
            draw.text((x, y - 42), title, font=F_PANEL, fill=(18, 24, 32))
            canvas.paste(img, (x, y))
        draw.rounded_rectangle((70, 760, 1850, 1000), radius=14, fill=(255, 255, 255), outline=(214, 221, 230), width=2)
        draw.text((100, 792), "Quantitative readout", font=F_PANEL, fill=(18, 24, 32))
        txt = (
            f"DDBot official HM {fmt(ddbot.get('final_height_map_error_mean'))}, "
            f"EMD {fmt(ddbot.get('emd_or_earth_mover_mean'))}, "
            f"evals {fmt(ddbot.get('optimizer_evals_mean'))}. "
            f"Ours HM {fmt(ours.get('final_height_map_error_mean'))}, "
            f"EMD {fmt(ours.get('emd_or_earth_mover_mean'))}, "
            f"completion {fmt(ours.get('target_trench_completion_mean'))}, "
            f"runtime {fmt(ours.get('runtime_sec_mean'))} sec."
        )
        draw_wrapped(draw, (100, 840), txt, 1650, F_BODY, (54, 65, 80))
        draw_wrapped(
            draw,
            (100, 920),
            "Scope: DDBot numbers are official artifacts; ours was re-run locally. DDBot force/runtime need a full DDBot runtime reproduction.",
            1650,
            F_SMALL,
            (82, 93, 108),
        )
        draw.rounded_rectangle((70, 1024, int(70 + 1780 * idx / max(1, total - 1)), 1036), radius=5, fill=(47, 102, 178))
        frames.append(np.asarray(canvas))
    iio.imwrite(OUT / "shared_benchmark_comparison.mp4", frames, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
    Image.fromarray(frames[total // 2]).save(OUT / "shared_benchmark_comparison.jpg", quality=92)


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    target_path = OUT / "ddbot_target_sand_task2_height_map_res40.npy"
    ddbot_sim_path = OUT / "ddbot_official_sim_sand_task2_height_map_res40.npy"
    ensure_download(target_path, TARGET_URL)
    ensure_download(ddbot_sim_path, DDBOT_SIM_URL)
    target_hm = np.load(target_path).astype(np.float32)
    ddbot_sim_hm = np.load(ddbot_sim_path).astype(np.float32)

    ddbot_rows, ddbot_aux = load_ddbot_seed_rows(seeds)
    ddbot_sim_metrics = height_map_metrics(
        ddbot_sim_hm,
        target_hm,
        official_height_map_error=ddbot_aux["aggregate_best"]["Loss"]["height_map_loss"],
        official_emd=ddbot_aux["aggregate_best"]["Loss"]["emd_loss"],
    )
    (OUT / "ddbot_sim_recomputed_metrics.json").write_text(json.dumps(jsonable(ddbot_sim_metrics), indent=2), encoding="utf-8")
    flat_hm = np.full_like(target_hm, SAND_GROUND_LEVEL_M, dtype=np.float32)
    flat_metrics = height_map_metrics(flat_hm, target_hm)
    ddbot_sim_rows = [
        {
            "method": "DDBot official sim height-map artifact",
            "seed": "best_archive",
            "material": MATERIAL,
            "task_id": TASK_ID,
            "target_source": "official_dbot_lfs_target",
            "initial_bed": "DDBot official fixed bed (artifact; not re-run)",
            "trajectory_budget": "official 5-parameter digging skill",
            "rollout_horizon_sec": None,
            "rollout_frames": None,
            "optimizer_evals": 21,
            "best_step": ddbot_aux["aggregate_best"].get("Step"),
            "final_height_map_error": ddbot_sim_metrics["height_map_abs_sum_m"],
            "emd_or_earth_mover": ddbot_sim_metrics["emd_hungarian_m"],
            "chamfer_distance": ddbot_sim_metrics["chamfer_distance_m"],
            "dug_volume_error_m3": ddbot_sim_metrics["dug_volume_error_m3"],
            "target_trench_completion": ddbot_sim_metrics["target_trench_completion"],
            "overflow_spillage_m3": ddbot_sim_metrics["overflow_spillage_m3"],
            "force_or_reaction_mismatch": None,
            "peak_reaction_norm": None,
            "runtime_sec": None,
            "sample_efficiency_hm_per_eval": ddbot_sim_metrics["height_map_abs_sum_m"] / 21.0,
            "status": "official_render_archive_height_map",
        }
    ]
    flat_rows = [
        {
            "method": "Flat no-action bed",
            "seed": "control",
            "material": MATERIAL,
            "task_id": TASK_ID,
            "target_source": "official_dbot_lfs_target",
            "initial_bed": "flat bed at DDBot sand ground level",
            "trajectory_budget": "no tool motion",
            "rollout_horizon_sec": 0.0,
            "rollout_frames": 0,
            "optimizer_evals": 0,
            "best_step": None,
            "final_height_map_error": flat_metrics["height_map_abs_sum_m"],
            "emd_or_earth_mover": flat_metrics["emd_hungarian_m"],
            "chamfer_distance": flat_metrics["chamfer_distance_m"],
            "dug_volume_error_m3": flat_metrics["dug_volume_error_m3"],
            "target_trench_completion": flat_metrics["target_trench_completion"],
            "overflow_spillage_m3": flat_metrics["overflow_spillage_m3"],
            "force_or_reaction_mismatch": None,
            "peak_reaction_norm": 0.0,
            "runtime_sec": 0.0,
            "sample_efficiency_hm_per_eval": None,
            "status": "sanity_control",
        }
    ]

    ours_rows: list[dict[str, Any]] = []
    ours_maps: list[np.ndarray] = []
    initial_maps: list[np.ndarray] = []
    if not args.skip_ours:
        ours_mod = load_ours_module()
        config = ours_mod.load_config(args.ours_config)
        row = ours_mod.load_rollout_row(args.property_csv, args.row)
        for seed in seeds:
            print(f"running_ours_seed={seed}")
            row_out, final_hm, initial_hm, _forces = run_ours_seed(seed, args, target_hm, ours_mod, config, row)
            ours_rows.append(row_out)
            ours_maps.append(final_hm)
            initial_maps.append(initial_hm)
            np.save(OUT / f"ours_seed_{seed}_height_map_res40.npy", final_hm)
    rows = ddbot_rows + ddbot_sim_rows + flat_rows + ours_rows
    summary = summarise(rows)

    protocol = {
        "name": "shared_ddbot_sand_task2_benchmark",
        "ddbot_repo": "https://github.com/IanYangChina/DDBot-IEEE-TRO-2025",
        "ddbot_commit": DDBOT_COMMIT,
        "material": MATERIAL,
        "task_id": TASK_ID,
        "seeds": seeds,
        "target_height_map": target_path,
        "target_source": TARGET_URL,
        "height_map_res": RES,
        "height_map_size_m": HEIGHT_MAP_SIZE_M,
        "sand_ground_level_m": SAND_GROUND_LEVEL_M,
        "metrics": [
            "final_height_map_error",
            "emd_or_earth_mover",
            "chamfer_distance",
            "dug_volume_error_m3",
            "target_trench_completion",
            "overflow_spillage_m3",
            "force_or_reaction_mismatch",
            "runtime_sec",
            "optimizer_evals",
            "sample_efficiency_hm_per_eval",
        ],
        "ddbot_side": "official seed-level best_loss/raw_data artifacts, not local re-run",
        "ours_side": "local MPM posterior-conditioned excavation rollout",
        "ours_frames": int(args.frames),
        "ours_substeps_per_frame": int(args.substeps_per_frame),
        "ours_device_requested": args.device,
    }
    write_csv(OUT / "shared_benchmark_seed_results.csv", rows)
    write_csv(OUT / "shared_benchmark_summary.csv", summary)
    (OUT / "shared_benchmark_protocol.json").write_text(json.dumps(jsonable(protocol), indent=2), encoding="utf-8")
    (OUT / "shared_benchmark_aux.json").write_text(json.dumps(jsonable({"ddbot": ddbot_aux}), indent=2), encoding="utf-8")

    save_height_map_image(OUT / "target_height_map.png", target_hm, "DDBot target sand task-2")
    save_height_map_image(OUT / "ddbot_official_sim_height_map.png", ddbot_sim_hm, "DDBot official sim artifact")
    if ours_maps:
        save_height_map_image(OUT / "ours_mean_height_map.png", np.mean(np.stack(ours_maps), axis=0), "This repo MPM mean")
    plot_summary(summary, OUT / "shared_benchmark_summary.png")
    plot_convergence({"ddbot": ddbot_aux}, ours_rows, OUT / "shared_benchmark_convergence.png")
    make_report(summary, rows, protocol)
    if args.write_video:
        make_video(target_hm, ddbot_sim_hm, ours_maps, summary)

    print(json.dumps(jsonable({"summary": summary, "out_dir": OUT}), indent=2))


if __name__ == "__main__":
    main()
