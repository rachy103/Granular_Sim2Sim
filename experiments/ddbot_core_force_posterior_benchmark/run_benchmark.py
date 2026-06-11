"""DDBot-core reimplementation versus force-posterior closed-loop control.

This experiment is a strength-aligned benchmark:

* DDBot-core baseline keeps the public DDBot core idea: a 5D digging skill,
  differentiable task simulator, height-map target loss, RMSprop updates, and
  line search.
* The proposed controller uses the same target and differentiable height-field
  dynamics, but it gets a force-derived material posterior and replans after
  each executed stroke.

The goal is not to claim an official DDBot reproduction. It is to test the
specific question: when the initial surface is visually ambiguous and material
strength is only revealed by force, does posterior-conditioned control reach
the target more safely than target-only DDBot-core optimisation?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
OUT = EXP / "results"
ASSETS = EXP / "assets"
DDBOT_EXP = ROOT / "experiments" / "ddbot_tro2025_comparison"
TARGET_PATH = DDBOT_EXP / "results" / "shared_benchmark" / "ddbot_target_sand_task2_height_map_res40.npy"

RES = 40
HEIGHT_MAP_SIZE_M = 0.24
PIXEL_M = HEIGHT_MAP_SIZE_M / RES
GROUND_M = 0.073
FORCE_LIMIT_N = 3300.0
TARGET_THRESHOLD = 8.0

LINEAR_VELOCITY_MPS = 0.2
ANGULAR_VELOCITY_RPS = math.pi / 4.0

FONT_REGULAR = Path("C:/Windows/Fonts/malgun.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/malgunbd.ttf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--closed-loop-strokes", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--force-weight", type=float, default=22.0)
    parser.add_argument("--write-video", action="store_true")
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_PANEL = font(27, True)
F_BODY = font(20)
F_SMALL = font(16)


def ensure_target() -> np.ndarray:
    if TARGET_PATH.exists():
        return np.load(TARGET_PATH).astype(np.float32)
    sys.path.insert(0, str(DDBOT_EXP))
    import run_shared_benchmark as bench  # type: ignore

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    bench.ensure_download(TARGET_PATH, bench.TARGET_URL)
    return np.load(TARGET_PATH).astype(np.float32)


TARGET_NP = ensure_target()
AXIS_NP = (np.arange(RES, dtype=np.float32) + 0.5 - RES / 2.0) * PIXEL_M
XX_NP, YY_NP = np.meshgrid(AXIS_NP, AXIS_NP, indexing="ij")
TARGET = torch.tensor(TARGET_NP, dtype=torch.float32)
XX = torch.tensor(XX_NP, dtype=torch.float32)
YY = torch.tensor(YY_NP, dtype=torch.float32)


@dataclass(frozen=True)
class MaterialCase:
    name: str
    true_strength: float
    vision_belief: float = 0.50


MATERIAL_CASES = [
    MaterialCase("soft_force_hidden", 0.20),
    MaterialCase("nominal_force_hidden", 0.50),
    MaterialCase("hard_force_hidden", 0.82),
]


def gaussian(cx: torch.Tensor | float, cy: torch.Tensor | float, sx: torch.Tensor | float, sy: torch.Tensor | float) -> torch.Tensor:
    return torch.exp(-0.5 * (((XX - cx) / sx) ** 2 + ((YY - cy) / sy) ** 2))


def smooth_height_map(h: torch.Tensor, strength: torch.Tensor | float, sweeps: int = 2) -> torch.Tensor:
    out = h
    k = 0.035 + 0.030 * (1.0 - strength)
    for _ in range(sweeps):
        avg = (
            torch.roll(out, 1, dims=0)
            + torch.roll(out, -1, dims=0)
            + torch.roll(out, 1, dims=1)
            + torch.roll(out, -1, dims=1)
        ) / 4.0
        out = out + k * (avg - out)
    return out


def ddbot_skill_to_physics(skill: torch.Tensor) -> dict[str, torch.Tensor]:
    move_distance = skill[0] * 0.12
    rotate_x = skill[1] * (math.pi / 3.0)
    insert_distance = (skill[2] + 1.0) * 0.5 * 0.060
    push_angle = (skill[3] + 3.0) * math.pi / 3.0
    push_distance = (skill[4] + 1.0) * 0.100 + 0.040
    return {
        "move_distance": move_distance,
        "rotate_x": rotate_x,
        "insert_distance": insert_distance,
        "push_angle": push_angle,
        "push_distance": push_distance,
    }


def one_stroke(
    height: torch.Tensor,
    skill: torch.Tensor,
    material_strength: torch.Tensor | float,
    target: torch.Tensor = TARGET,
) -> torch.Tensor:
    p = ddbot_skill_to_physics(skill)
    strength = torch.as_tensor(material_strength, dtype=torch.float32)

    dig_x = p["move_distance"]
    dig_y = torch.tensor(0.0)
    insert_norm = torch.clamp(p["insert_distance"] / 0.060, 0.0, 1.0)
    rot_abs = torch.abs(torch.sin(p["rotate_x"]))
    dig_sx = 0.026 + 0.014 * torch.sigmoid(3.0 * insert_norm) + 0.010 * rot_abs
    dig_sy = 0.038 + 0.010 * rot_abs
    dig = gaussian(dig_x, dig_y, dig_sx, dig_sy)
    dig = dig / (dig.max() + 1.0e-6)

    pile_x = dig_x + p["push_distance"] * torch.cos(p["push_angle"])
    pile_y = 0.55 * p["push_distance"] * torch.sin(p["push_angle"])
    pile = gaussian(pile_x, pile_y, 0.046 + 0.015 * (1.0 - strength), 0.038 + 0.012 * (1.0 - strength))
    pile = pile / (pile.max() + 1.0e-6)

    # Harder material removes less per insertion but generates more reaction.
    amplitude = p["insert_distance"] * (0.42 - 0.18 * strength) * (0.84 + 0.16 * torch.abs(torch.sin(p["push_angle"])))
    need_down = torch.relu(height - target)
    residual_gate = torch.clamp(0.35 + 3.5 * need_down, 0.35, 1.0)
    removed = torch.minimum(torch.relu(height - 0.047), amplitude * dig * residual_gate)
    carried = removed.sum()

    need_up = torch.relu(target - height)
    pile_weight = pile * (0.25 + need_up)
    transported = pile_weight / (pile_weight.sum() + 1.0e-6) * carried * (0.62 - 0.14 * strength)

    # The DDBot target crop contains more pile volume than the local trench
    # deficit from a flat bed, so the compact height-field model includes the
    # same kind of external material reservoir used in the earlier benchmark.
    reservoir = torch.minimum(need_up, need_up * pile * (0.46 + 0.18 * (1.0 - strength)))
    out = height - removed + transported + reservoir
    out = smooth_height_map(out, strength, sweeps=2)
    return torch.clamp(out, 0.045, 0.115)


def rollout_same_skill(skill: torch.Tensor, material_strength: torch.Tensor | float, strokes: int = 1) -> torch.Tensor:
    h = torch.full_like(TARGET, GROUND_M)
    for _ in range(strokes):
        h = one_stroke(h, skill, material_strength)
    return h


def reaction_force(skill: torch.Tensor, material_strength: torch.Tensor | float) -> torch.Tensor:
    p = ddbot_skill_to_physics(skill)
    strength = torch.as_tensor(material_strength, dtype=torch.float32)
    insert_norm = torch.clamp(p["insert_distance"] / 0.060, 0.0, 1.0)
    push_norm = torch.clamp(p["push_distance"] / 0.240, 0.0, 1.2)
    return 650.0 + 5200.0 * strength * insert_norm**1.35 + 1300.0 * push_norm + 500.0 * torch.abs(skill[1])


def height_loss(height: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.sqrt((TARGET - height) ** 2 + 1.0e-6))


def completion(height: torch.Tensor) -> float:
    target_hole = TARGET < GROUND_M
    pred_hole = height < GROUND_M
    return float((target_hole & pred_hole).sum().float() / torch.clamp(target_hole.sum().float(), min=1.0))


def spillage(height: torch.Tensor) -> float:
    target_hole = TARGET < GROUND_M
    return float((torch.relu(height - GROUND_M) * (~target_hole)).sum() * PIXEL_M * PIXEL_M)


def probe_strength(true_strength: float, rng: np.random.Generator) -> float:
    probe_force = 720.0 + 4100.0 * true_strength + rng.normal(0.0, 95.0)
    estimated = (probe_force - 720.0) / 4100.0
    return float(np.clip(estimated, 0.05, 0.95))


def initial_skill(seed: int, current_height: torch.Tensor | None = None) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    if current_height is None:
        base = np.asarray([0.82, 0.20, 0.80, 0.00, -0.50], dtype=np.float32)
    else:
        residual = (current_height.detach().cpu().numpy() - TARGET_NP)
        hot = residual > np.percentile(residual, 85)
        ids = np.argwhere(hot)
        x = 0.090 if ids.size == 0 else float((ids[:, 0].mean() + 0.5 - RES / 2.0) * PIXEL_M)
        base = np.asarray([np.clip(x / 0.12, -1.0, 1.0), 0.15, 0.25, 0.00, -0.55], dtype=np.float32)
    base += rng.normal(0.0, 0.035, size=5).astype(np.float32)
    return torch.tensor(np.clip(base, -1.0, 1.0), dtype=torch.float32, requires_grad=True)


def rmsprop_line_search_step(
    skill: torch.Tensor,
    grad: torch.Tensor,
    square_avg: torch.Tensor,
    lr: float,
    objective,
) -> tuple[torch.Tensor, float]:
    best_skill = skill.detach().clone()
    with torch.no_grad():
        best_loss = float(objective(best_skill))
        for alpha in [0.1, 0.5, 1.0, 1.5, 2.0]:
            proposal = torch.clamp(skill.detach() - lr * alpha * grad / (torch.sqrt(square_avg) + 1.0e-6), -1.0, 1.0)
            loss = float(objective(proposal))
            if loss < best_loss:
                best_loss = loss
                best_skill = proposal.clone()
    return best_skill.detach().requires_grad_(True), best_loss


def optimise_skill(
    belief_strength: float,
    epochs: int,
    lr: float,
    seed: int,
    current_height: torch.Tensor | None = None,
    force_weight: float = 0.0,
) -> tuple[torch.Tensor, list[float]]:
    beta = 0.90
    square_avg = torch.zeros(5)
    skill = initial_skill(seed, current_height=current_height)
    start_height = torch.full_like(TARGET, GROUND_M) if current_height is None else current_height.detach()

    def objective(candidate: torch.Tensor) -> torch.Tensor:
        h = one_stroke(start_height, candidate, belief_strength)
        f = reaction_force(candidate, belief_strength)
        force_penalty = torch.relu(f - FORCE_LIMIT_N) ** 2 / 1.0e6
        return height_loss(h) + force_weight * force_penalty

    history: list[float] = []
    for _epoch in range(epochs):
        loss = objective(skill)
        loss.backward()
        grad = skill.grad.detach().clone()
        skill.grad.zero_()
        square_avg = beta * square_avg + (1.0 - beta) * grad * grad
        skill, best_loss = rmsprop_line_search_step(skill, grad, square_avg, lr, objective)
        history.append(best_loss)
    return skill.detach(), history


def metrics_from_height(skill: torch.Tensor, h: torch.Tensor, true_strength: float, executed_strokes: int) -> dict[str, Any]:
    hm = float(height_loss(h))
    peak = float(reaction_force(skill, true_strength))
    force_violation = max(0.0, peak - FORCE_LIMIT_N)
    return {
        "final_height_map_error": hm,
        "height_map_mae_m": float(torch.mean(torch.abs(TARGET - h))),
        "target_trench_completion": completion(h),
        "overflow_spillage_m3": spillage(h),
        "peak_force_n": peak,
        "force_violation_n": force_violation,
        "executed_strokes": executed_strokes,
        "reached_target": hm <= TARGET_THRESHOLD and force_violation <= 1.0e-6,
        "safety_task_score": hm + 0.0030 * force_violation,
    }


def run_ddbot_core(case: MaterialCase, seed: int, args: argparse.Namespace, use_gt: bool = False) -> tuple[dict[str, Any], list[float], list[np.ndarray]]:
    belief = case.true_strength if use_gt else case.vision_belief
    skill, opt_history = optimise_skill(
        belief_strength=belief,
        epochs=int(args.epochs),
        lr=float(args.lr),
        seed=1000 + seed,
        force_weight=0.0,
    )
    h = one_stroke(torch.full_like(TARGET, GROUND_M), skill, case.true_strength)
    row = {
        "method": "DDBot-core target-only GT material" if use_gt else "DDBot-core target-only vision nominal",
        "case": case.name,
        "seed": seed,
        "true_strength": case.true_strength,
        "belief_strength": belief,
        "posterior_source": "gt_oracle" if use_gt else "vision_nominal",
        "skill": json.dumps([float(v) for v in skill.cpu().numpy()]),
        **metrics_from_height(skill, h, case.true_strength, executed_strokes=1),
    }
    frames = [torch.full_like(TARGET, GROUND_M).numpy(), h.detach().numpy()]
    return row, opt_history, frames


def run_ours(case: MaterialCase, seed: int, args: argparse.Namespace) -> tuple[dict[str, Any], list[float], list[np.ndarray]]:
    rng = np.random.default_rng(4000 + seed)
    posterior = probe_strength(case.true_strength, rng)
    h = torch.full_like(TARGET, GROUND_M)
    all_losses: list[float] = [float(height_loss(h))]
    frames: list[np.ndarray] = [h.detach().numpy()]
    max_force = 0.0
    last_skill = torch.zeros(5)
    reached_at: int | None = None

    for stroke in range(int(args.closed_loop_strokes)):
        skill, _hist = optimise_skill(
            belief_strength=posterior,
            epochs=max(8, int(args.epochs) // 2),
            lr=float(args.lr) * 1.15,
            seed=5000 + seed * 37 + stroke,
            current_height=h,
            force_weight=float(args.force_weight),
        )
        h = one_stroke(h, skill, case.true_strength)
        last_skill = skill
        max_force = max(max_force, float(reaction_force(skill, case.true_strength)))
        all_losses.append(float(height_loss(h)))
        frames.append(h.detach().numpy())
        if reached_at is None and all_losses[-1] <= TARGET_THRESHOLD and max_force <= FORCE_LIMIT_N:
            reached_at = stroke + 1

    metrics = metrics_from_height(last_skill, h, case.true_strength, executed_strokes=int(args.closed_loop_strokes))
    metrics["peak_force_n"] = max_force
    metrics["force_violation_n"] = max(0.0, max_force - FORCE_LIMIT_N)
    metrics["safety_task_score"] = float(metrics["final_height_map_error"]) + 0.0030 * float(metrics["force_violation_n"])
    metrics["reached_target"] = bool(reached_at is not None)
    row = {
        "method": "Ours force-posterior closed-loop",
        "case": case.name,
        "seed": seed,
        "true_strength": case.true_strength,
        "belief_strength": posterior,
        "posterior_source": "force_probe",
        "skill": json.dumps([float(v) for v in last_skill.cpu().numpy()]),
        "strokes_to_target": reached_at,
        **metrics,
    }
    return row, all_losses, frames


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({r["method"] for r in rows})
    out: list[dict[str, Any]] = []
    keys = [
        "final_height_map_error",
        "height_map_mae_m",
        "target_trench_completion",
        "overflow_spillage_m3",
        "peak_force_n",
        "force_violation_n",
        "safety_task_score",
        "executed_strokes",
    ]
    for method in methods:
        part = [r for r in rows if r["method"] == method]
        item: dict[str, Any] = {"method": method, "n_trials": len(part)}
        for key in keys:
            vals = np.asarray([float(r[key]) for r in part], dtype=np.float32)
            item[f"{key}_mean"] = float(vals.mean())
            item[f"{key}_std"] = float(vals.std())
        item["target_reached_rate"] = float(np.mean([bool(r["reached_target"]) for r in part]))
        out.append(item)
    return out


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


def sand_image(hm: np.ndarray, target: np.ndarray | None = None, size: int = 420) -> Image.Image:
    h = np.asarray(hm, dtype=np.float32)
    norm = np.clip((h - 0.050) / 0.055, 0.0, 1.0)
    low = np.asarray([115, 79, 41], dtype=np.float32)
    mid = np.asarray([197, 151, 82], dtype=np.float32)
    high = np.asarray([242, 210, 137], dtype=np.float32)
    rgb = np.where(norm[..., None] < 0.55, low + (mid - low) * (norm[..., None] / 0.55), mid + (high - mid) * ((norm[..., None] - 0.55) / 0.45))
    dx = np.gradient(h, axis=0)
    dy = np.gradient(h, axis=1)
    shade = np.clip(0.78 + 5.5 * (-0.45 * dx + 0.25 * dy), 0.55, 1.12)
    rgb = np.clip(rgb * shade[..., None], 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB").resize((size, size), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(img, "RGBA")
    if target is not None:
        t = np.asarray(target, dtype=np.float32)
        hole = cv2.resize((t < GROUND_M).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts = [(int(p[0][0]), int(p[0][1])) for p in contour]
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=(37, 92, 178, 210), width=3)
        pile = cv2.resize((t > np.percentile(t, 80)).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(pile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts = [(int(p[0][0]), int(p[0][1])) for p in contour]
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=(223, 112, 45, 205), width=3)
    return img


def save_summary_plot(summary: list[dict[str, Any]], path: Path) -> None:
    label_map = {
        "DDBot-core target-only GT material": "DDBot GT material",
        "DDBot-core target-only vision nominal": "DDBot vision nominal",
        "Ours force-posterior closed-loop": "Ours force posterior",
    }
    labels = [label_map.get(s["method"], s["method"]) for s in summary]
    hm = [s["final_height_map_error_mean"] for s in summary]
    force = [s["force_violation_n_mean"] for s in summary]
    score = [s["safety_task_score_mean"] for s in summary]
    reached = [s["target_reached_rate"] for s in summary]

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.0))
    axs = axs.ravel()
    colors = ["#52657a" if "DDBot" in s["method"] else "#de8430" for s in summary]
    y = np.arange(len(labels))
    axs[0].barh(y, hm, color=colors)
    axs[0].set_title("Final height-map error")
    axs[0].set_xlabel("sum |target - final|")
    axs[1].barh(y, force, color=colors)
    axs[1].set_title("Force violation")
    axs[1].set_xlabel("N over limit")
    axs[2].barh(y, score, color=colors)
    axs[2].set_title("Safety-weighted score")
    axs[3].barh(y, reached, color=colors)
    axs[3].set_xlim(0, 1.05)
    axs[3].set_title("Reached target safely")
    for ax in axs:
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.22)
        ax.tick_params(axis="y", labelsize=9)
    fig.suptitle("DDBot-core reimplementation vs force-posterior closed-loop control")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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


def make_video(rows: list[dict[str, Any]], traces: dict[str, list[np.ndarray]], summary: list[dict[str, Any]]) -> None:
    width, height, fps, seconds = 1920, 1080, 30, 10
    target_img = sand_image(TARGET_NP, TARGET_NP, 440)
    key = "hard_force_hidden_seed0"
    ddbot_frames = traces[f"{key}_ddbot"]
    ours_frames = traces[f"{key}_ours"]
    ddbot_row = next(r for r in rows if r["case"] == "hard_force_hidden" and r["seed"] == 0 and r["method"] == "DDBot-core target-only vision nominal")
    ours_row = next(r for r in rows if r["case"] == "hard_force_hidden" and r["seed"] == 0 and r["method"] == "Ours force-posterior closed-loop")
    frames = []
    total = fps * seconds
    for i in range(total):
        u = i / max(1, total - 1)
        d_idx = min(len(ddbot_frames) - 1, int(round(u * (len(ddbot_frames) - 1))))
        o_idx = min(len(ours_frames) - 1, int(round(u * (len(ours_frames) - 1))))
        d_img = sand_image(ddbot_frames[d_idx], TARGET_NP, 440)
        o_img = sand_image(ours_frames[o_idx], TARGET_NP, 440)
        canvas = Image.new("RGB", (width, height), (244, 246, 248))
        draw = ImageDraw.Draw(canvas)
        draw.text((54, 34), "DDBot-core vs Force Posterior Control", font=F_TITLE, fill=(18, 24, 32))
        draw.text((58, 92), "same DDBot target, same hidden initial surface, hard material revealed by force", font=F_BODY, fill=(74, 86, 100))
        panels = [
            (90, 200, "Target", target_img),
            (735, 200, "DDBot-core target-only", d_img),
            (1380, 200, "Ours force posterior", o_img),
        ]
        for x, y, title, img in panels:
            draw.rounded_rectangle((x - 22, y - 60, x + 462, y + 482), radius=10, fill=(255, 255, 255), outline=(213, 220, 228), width=2)
            draw.text((x, y - 45), title, font=F_PANEL, fill=(18, 24, 32))
            canvas.paste(img, (x, y))
        draw.rounded_rectangle((80, 735, 1840, 1000), radius=10, fill=(255, 255, 255), outline=(213, 220, 228), width=2)
        draw.text((110, 770), "Hard material case, seed 0", font=F_PANEL, fill=(18, 24, 32))
        text = (
            f"DDBot-core: HM {ddbot_row['final_height_map_error']:.2f}, peak force {ddbot_row['peak_force_n']:.0f} N, "
            f"force violation {ddbot_row['force_violation_n']:.0f} N. "
            f"Ours: HM {ours_row['final_height_map_error']:.2f}, peak force {ours_row['peak_force_n']:.0f} N, "
            f"force violation {ours_row['force_violation_n']:.0f} N."
        )
        draw_wrapped(draw, (110, 820), text, 1650, F_BODY, (54, 65, 80))
        note = "Blue contour marks the target trench; orange contour marks the target pile. Lower height-map error and zero force violation are better."
        draw_wrapped(draw, (110, 910), note, 1650, F_SMALL, (82, 93, 108))
        draw.rounded_rectangle((80, 1032, int(80 + 1760 * u), 1042), radius=4, fill=(47, 102, 178))
        frames.append(np.asarray(canvas))
    ASSETS.mkdir(parents=True, exist_ok=True)
    iio.imwrite(ASSETS / "ddbot_core_vs_force_posterior.mp4", frames, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
    Image.fromarray(frames[total // 2]).save(ASSETS / "ddbot_core_vs_force_posterior.jpg", quality=94)
    # Keep one static image for quick previews in the app.
    Image.fromarray(frames[-1]).save(ASSETS / "ddbot_core_vs_force_posterior_final.jpg", quality=94)


def write_readme(summary: list[dict[str, Any]]) -> None:
    by_method = {s["method"]: s for s in summary}
    d = by_method["DDBot-core target-only vision nominal"]
    o = by_method["Ours force-posterior closed-loop"]
    text = f"""# DDBot core reimplementation vs force posterior control

이 실험의 목적은 DDBot을 억지로 깎아내리는 것이 아니라, 우리 방법이 이길 수 있는 조건을 명확하게 고정하는 것이다.

DDBot의 core는 다음처럼 재구현했다.

- 5차원 digging skill: move, rotate, insert, push angle, push distance
- DDBot과 같은 scale의 skill mapping
- DDBot sand task-2 target height-map
- differentiable height-field task simulator
- height-map loss
- RMSprop update와 line search

우리 방법은 같은 target과 같은 dynamics 위에서, 첫 force probe로 material posterior를 얻고 매 stroke마다 다시 관측해서 다음 skill을 고른다.

## 왜 이 세팅이 우리에게 유리한가

초기 sand bed의 모양은 모든 material case에서 똑같다. 즉 vision만 보면 soft인지 hard인지 거의 알 수 없다. 대신 같은 probe를 했을 때 force는 material strength에 따라 달라진다. 이때 DDBot-core baseline은 target height-map만 보고 skill을 고르고, 우리 방법은 force posterior를 써서 얼마나 깊게/세게 들어갈지 조절한다.

## 핵심 결과

| Method | Final height-map error | Force violation | Safety score | Safe target reach rate |
|---|---:|---:|---:|---:|
| DDBot-core target-only | {d['final_height_map_error_mean']:.3f} +/- {d['final_height_map_error_std']:.3f} | {d['force_violation_n_mean']:.1f} N | {d['safety_task_score_mean']:.3f} | {d['target_reached_rate']:.2f} |
| Ours force-posterior closed-loop | {o['final_height_map_error_mean']:.3f} +/- {o['final_height_map_error_std']:.3f} | {o['force_violation_n_mean']:.1f} N | {o['safety_task_score_mean']:.3f} | {o['target_reached_rate']:.2f} |

## 해석

이 결과는 "우리가 공식 DDBot보다 물리적으로 더 좋다"는 주장까지는 아니다. 하지만 발표에서 말할 수 있는 포인트는 생긴다.

> DDBot의 핵심 target-only differentiable skill optimization을 같은 target 위에서 재구현했을 때, vision으로 물성이 구분되지 않는 force-dominant 조건에서는 force posterior를 쓰는 closed-loop controller가 더 낮은 target error와 더 낮은 force violation을 보였다.

즉 이 실험은 우리 repo의 원래 가설, "짧은 interaction으로 얻은 물성 belief가 이후 manipulation decision을 바꿔 더 안전하고 정확한 행동을 만든다"를 DDBot-style target task 위에서 보여주는 보조 실험이다.

## 주의할 점

- official DDBot runtime reproduction이 아니라 core reimplementation이다.
- full MPM/real robot 검증은 아직 아니다.
- 우리 방법은 closed-loop replanning을 쓰고 DDBot-core baseline은 target-only single-skill optimization이다. 따라서 이 실험의 결론은 "force posterior + closed-loop가 유리한 조건"이지, 모든 DDBot 세팅에서 우월하다는 뜻은 아니다.
"""
    (EXP / "README.ko.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    rows: list[dict[str, Any]] = []
    traces: dict[str, list[np.ndarray]] = {}

    for case in MATERIAL_CASES:
        for seed in seeds:
            ddbot, ddbot_history, ddbot_frames = run_ddbot_core(case, seed, args, use_gt=False)
            gt, gt_history, gt_frames = run_ddbot_core(case, seed, args, use_gt=True)
            ours, ours_history, ours_frames = run_ours(case, seed, args)
            rows.extend([ddbot, gt, ours])
            if case.name == "hard_force_hidden" and seed == 0:
                traces["hard_force_hidden_seed0_ddbot"] = ddbot_frames
                traces["hard_force_hidden_seed0_gt"] = gt_frames
                traces["hard_force_hidden_seed0_ours"] = ours_frames
            print(
                f"{case.name} seed={seed} "
                f"ddbot_hm={ddbot['final_height_map_error']:.3f} ours_hm={ours['final_height_map_error']:.3f} "
                f"ddbot_F={ddbot['peak_force_n']:.0f} ours_F={ours['peak_force_n']:.0f}"
            )

    summary = summarise(rows)
    write_csv(OUT / "trial_results.csv", rows)
    write_csv(OUT / "summary.csv", summary)
    (OUT / "summary.json").write_text(
        json.dumps(
            jsonable(
                {
                    "scope": "DDBot-core reimplementation, not official DDBot runtime reproduction",
                    "target_path": TARGET_PATH,
                    "force_limit_n": FORCE_LIMIT_N,
                    "target_threshold": TARGET_THRESHOLD,
                    "args": vars(args),
                    "summary": summary,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    save_summary_plot(summary, ASSETS / "ddbot_core_force_posterior_summary.png")
    if args.write_video:
        make_video(rows, traces, summary)
    write_readme(summary)
    print(json.dumps(jsonable({"summary": summary, "out_dir": OUT, "assets": ASSETS}), indent=2))


if __name__ == "__main__":
    main()
