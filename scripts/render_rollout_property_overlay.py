from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

import cv2  # noqa: E402
import numpy as np  # noqa: E402


TARGETS = ["rho", "phi_deg", "delta_deg", "cohesion_kpa"]
COLORS = {
    "rho": (45, 125, 210),
    "phi_deg": (70, 155, 65),
    "delta_deg": (215, 130, 40),
    "cohesion_kpa": (165, 80, 170),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "outputs/mujoco_3d_mpm_cosim/mujoco_franka_3d_mpm_interaction.mp4",
    )
    parser.add_argument(
        "--rollout-csv",
        type=Path,
        default=ROOT / "outputs/online_mohr_coulomb_bestval_quick/rollout_predictions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/online_mohr_coulomb_bestval_quick/mujoco_property_estimation_overlay.mp4",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=ROOT / "outputs/online_mohr_coulomb_bestval_quick/mujoco_property_estimation_overlay_preview.png",
    )
    parser.add_argument("--graph-width", type=int, default=700)
    parser.add_argument(
        "--title",
        default="MuJoCo rendering + online Mohr-Coulomb property estimation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_with_paths(
        video=resolve(args.video),
        rollout_csv=resolve(args.rollout_csv),
        output=resolve(args.output),
        preview=resolve(args.preview),
        graph_width=int(args.graph_width),
        title=str(args.title),
    )


def main_with_paths(
    video: Path,
    rollout_csv: Path,
    output: Path,
    preview: Path,
    graph_width: int = 700,
    title: str = "MuJoCo rendering + online Mohr-Coulomb property estimation",
) -> None:
    video = resolve(video)
    rollout_csv = resolve(rollout_csv)
    output = resolve(output)
    preview = resolve(preview)

    rows = load_rollout(rollout_csv)
    cap = cv2.VideoCapture(video.as_posix())
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width + graph_width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output}")

    preview_frame: np.ndarray | None = None
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rollout_id = map_index(frame_id, max(frame_count, 1), len(rows))
        graph = draw_property_graph(rows, rollout_id, graph_width, height)
        composed = np.concatenate([frame, graph], axis=1)
        draw_header(composed, frame_id, frame_count, rollout_id, len(rows), title)
        writer.write(composed)
        if frame_id == max(0, frame_count // 2):
            preview_frame = composed.copy()
        frame_id += 1

    cap.release()
    writer.release()
    if preview_frame is None:
        raise RuntimeError("No frames were written")
    cv2.imwrite(preview.as_posix(), preview_frame)
    print(f"overlay_video={output}")
    print(f"overlay_preview={preview}")
    print(f"frames_written={frame_id}")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_rollout(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items() if value != ""})
    if not rows:
        raise RuntimeError(f"Empty rollout CSV: {path}")
    return rows


def map_index(video_frame: int, video_frames: int, rollout_frames: int) -> int:
    if video_frames <= 1 or rollout_frames <= 1:
        return 0
    alpha = video_frame / float(video_frames - 1)
    return int(np.clip(round(alpha * (rollout_frames - 1)), 0, rollout_frames - 1))


def draw_header(
    image: np.ndarray,
    frame_id: int,
    frame_count: int,
    rollout_id: int,
    rollout_count: int,
    title: str,
) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (245, 245, 245), -1)
    cv2.putText(
        image,
        title,
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    status = f"video frame {frame_id + 1}/{frame_count} | belief update {rollout_id + 1}/{rollout_count}"
    cv2.putText(
        image,
        status,
        (image.shape[1] - 560, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )


def draw_property_graph(rows: list[dict[str, float]], current: int, width: int, height: int) -> np.ndarray:
    graph = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.rectangle(graph, (0, 0), (width - 1, height - 1), (220, 220, 220), 1)
    cv2.putText(
        graph,
        "Property posterior rollout",
        (24, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        graph,
        "solid: estimate   dashed: GT   band: +/- sigma",
        (24, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (85, 85, 85),
        1,
        cv2.LINE_AA,
    )
    top = 124
    panel_h = (height - top - 24) // len(TARGETS)
    for target_id, name in enumerate(TARGETS):
        y_top = top + target_id * panel_h
        y_bottom = y_top + panel_h - 18
        draw_target_panel(graph, rows, current, name, 22, width - 24, y_top, y_bottom)
    return graph


def draw_target_panel(
    image: np.ndarray,
    rows: list[dict[str, float]],
    current: int,
    name: str,
    x0: int,
    x1: int,
    y_top: int,
    y_bottom: int,
) -> None:
    pred = np.asarray([row[f"pred_{name}"] for row in rows], dtype=np.float32)
    sigma = np.asarray([row[f"sigma_{name}"] for row in rows], dtype=np.float32)
    target = float(rows[0][f"target_{name}"])
    lo = float(min(np.min(pred - sigma), target))
    hi = float(max(np.max(pred + sigma), target))
    pad = max((hi - lo) * 0.12, 1.0e-3)
    lo -= pad
    hi += pad

    axis_y = y_bottom - 24
    plot_top = y_top + 28
    plot_bottom = axis_y
    cv2.line(image, (x0, axis_y), (x1, axis_y), (95, 95, 95), 1, cv2.LINE_AA)
    cv2.line(image, (x0, plot_top), (x0, plot_bottom), (95, 95, 95), 1, cv2.LINE_AA)

    target_y = plot_y(target, lo, hi, plot_top, plot_bottom)
    draw_dashed_line(image, (x0, target_y), (x1, target_y), (65, 65, 65), dash=12)

    xs: list[int] = []
    center_pts: list[list[int]] = []
    upper_pts: list[list[int]] = []
    lower_pts: list[list[int]] = []
    for idx in range(current + 1):
        x = int(x0 + idx / max(1, len(rows) - 1) * (x1 - x0))
        xs.append(x)
        center_pts.append([x, plot_y(float(pred[idx]), lo, hi, plot_top, plot_bottom)])
        upper_pts.append([x, plot_y(float(pred[idx] + sigma[idx]), lo, hi, plot_top, plot_bottom)])
        lower_pts.append([x, plot_y(float(pred[idx] - sigma[idx]), lo, hi, plot_top, plot_bottom)])

    color = COLORS[name]
    if len(center_pts) > 1:
        band = np.asarray(upper_pts + list(reversed(lower_pts)), dtype=np.int32)
        soft_color = tuple(int(0.65 * c + 0.35 * 250) for c in color)
        cv2.fillPoly(image, [band], soft_color)
        cv2.polylines(image, [np.asarray(center_pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    else:
        cv2.circle(image, tuple(center_pts[0]), 4, color, -1, cv2.LINE_AA)

    current_pred = float(pred[current])
    current_sigma = float(sigma[current])
    current_err = abs(current_pred - target)
    title = f"{name}: pred={current_pred:.3g}  GT={target:.3g}  err={current_err:.3g}  sigma={current_sigma:.3g}"
    cv2.putText(image, title, (x0, y_top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(image, f"{hi:.3g}", (x0 + 4, plot_top + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(image, f"{lo:.3g}", (x0 + 4, plot_bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1, cv2.LINE_AA)


def plot_y(value: float, lo: float, hi: float, top: int, bottom: int) -> int:
    return int(bottom - (value - lo) / max(hi - lo, 1.0e-6) * (bottom - top))


def draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    dash: int = 8,
) -> None:
    x0, y0 = start
    x1, y1 = end
    length = int(np.hypot(x1 - x0, y1 - y0))
    if length == 0:
        return
    for offset in range(0, length, dash * 2):
        t0 = offset / length
        t1 = min(offset + dash, length) / length
        p0 = (int(x0 + (x1 - x0) * t0), int(y0 + (y1 - y0) * t0))
        p1 = (int(x0 + (x1 - x0) * t1), int(y0 + (y1 - y0) * t1))
        cv2.line(image, p0, p1, color, 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
