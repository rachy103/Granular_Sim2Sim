"""Run Newton MPM sand examples and make a lightweight particle preview.

Newton's own USD output is the useful artifact here: it preserves the animated
particle cloud for a real renderer/viewer.  The MP4 preview below exists so the
run can be inspected on machines without usdview/usdrecord installed.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example",
        choices=["mpm_granular", "mpm_grain_rendering", "mpm_twoway_coupling"],
        default="mpm_granular",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/newton_mpm_spike"))
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel-size", type=float, default=0.06)
    parser.add_argument("--points-per-particle", type=int, default=4)
    parser.add_argument("--collider", choices=["cube", "wedge", "concave", "none"], default="wedge")
    parser.add_argument("--friction", type=float, default=0.55)
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--points-prim", default="auto")
    parser.add_argument("--max-preview-points", type=int, default=220_000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def run_newton(args: argparse.Namespace, usd_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "newton.examples",
        args.example,
        "--viewer",
        "usd",
        "--output-path",
        str(usd_path),
        "--num-frames",
        str(args.num_frames),
        "--device",
        args.device,
        "--headless",
        "--quiet",
        "--voxel-size",
        str(args.voxel_size),
    ]
    if args.example == "mpm_grain_rendering":
        cmd.extend(["--points-per-particle", str(args.points_per_particle)])
    elif args.example == "mpm_granular":
        cmd.extend(["--collider", args.collider, "--friction", str(args.friction)])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def choose_points_prim(stage, requested: str):
    if requested != "auto":
        prim = stage.GetPrimAtPath(requested)
        if not prim:
            raise ValueError(f"USD prim not found: {requested}")
        return prim

    for path in ["/root/grains", "/root/sand", "/root/model/particles"]:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.GetAttribute("points"):
            return prim
    raise ValueError("Could not find a Points prim. Try --points-prim /path/to/points")


def look_at_basis(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-8
    return right, up, forward


def sample_points(points_attr, time_code: float, max_points: int) -> np.ndarray:
    pts = np.asarray(points_attr.Get(time_code), dtype=np.float32)
    if pts.shape[0] > max_points:
        stride = max(1, math.ceil(pts.shape[0] / max_points))
        pts = pts[::stride]
    return pts


def make_camera(points_attr, time_samples: list[float], max_points: int) -> tuple[np.ndarray, np.ndarray, float]:
    probe_times = [time_samples[0], time_samples[len(time_samples) // 2], time_samples[-1]]
    probes = [sample_points(points_attr, t, max_points // 3) for t in probe_times]
    pts = np.concatenate(probes, axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = (lo + hi) * 0.5
    extent = hi - lo
    radius = float(max(extent.max(), 1e-3))
    eye = center + radius * np.array([1.45, -2.10, 1.05], dtype=np.float32)
    scale = radius * 1.20
    return eye.astype(np.float32), center.astype(np.float32), scale


def render_points(
    points: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    scale: float,
    width: int,
    height: int,
    label: str,
) -> np.ndarray:
    right, up, forward = look_at_basis(eye, target)
    rel = points - eye
    x = rel @ right
    y = rel @ up
    z = rel @ forward

    px = (width * 0.5 + x / scale * width * 0.72).astype(np.int32)
    py = (height * 0.55 - y / scale * height * 0.72).astype(np.int32)
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height) & (z > 0)
    px = px[valid]
    py = py[valid]
    if px.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    depth = z[valid]
    depth = (depth - depth.min()) / (np.ptp(depth) + 1e-6)
    height_world = points[valid, 2]
    height_world = (height_world - height_world.min()) / (np.ptp(height_world) + 1e-6)

    density = np.zeros((height, width), dtype=np.float32)
    shade = np.zeros((height, width), dtype=np.float32)
    np.add.at(density, (py, px), 1.0)
    np.add.at(shade, (py, px), 0.60 + 0.30 * height_world + 0.10 * (1.0 - depth))

    density = cv2.GaussianBlur(density, (0, 0), 1.05)
    shade = cv2.GaussianBlur(shade, (0, 0), 1.05)
    avg_shade = shade / (density + 1e-6)

    alpha = 1.0 - np.exp(-density * 0.95)
    alpha = np.clip(alpha, 0.0, 0.96)

    bg = np.zeros((height, width, 3), dtype=np.float32)
    bg[:] = np.array([28.0, 32.0, 36.0], dtype=np.float32)
    sand = np.array([99.0, 156.0, 194.0], dtype=np.float32)
    sand_img = sand[None, None, :] * np.clip(avg_shade[:, :, None], 0.52, 1.15)
    img = bg * (1.0 - alpha[:, :, None]) + sand_img * alpha[:, :, None]

    cv2.putText(img, label, (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (238, 238, 232), 2, cv2.LINE_AA)
    cv2.putText(img, "Newton USD point preview", (28, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 196, 200), 1, cv2.LINE_AA)
    return np.clip(img, 0, 255).astype(np.uint8)


def render_preview(usd_path: Path, mp4_path: Path, args: argparse.Namespace) -> None:
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    prim = choose_points_prim(stage, args.points_prim)
    points_attr = prim.GetAttribute("points")
    time_samples = points_attr.GetTimeSamples()
    if not time_samples:
        time_samples = list(np.linspace(stage.GetStartTimeCode(), stage.GetEndTimeCode(), args.num_frames))

    eye, target, scale = make_camera(points_attr, time_samples, args.max_preview_points)
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {mp4_path}")

    preview_png = mp4_path.with_suffix(".preview.png")
    contact_sheet_frames: list[np.ndarray] = []
    for i, t in enumerate(time_samples):
        pts = sample_points(points_attr, t, args.max_preview_points)
        label = f"{args.example}  frame {i + 1:03d}/{len(time_samples):03d}  points {len(pts):,}"
        frame = render_points(pts, eye, target, scale, args.width, args.height, label)
        if i == 0:
            cv2.imwrite(str(preview_png), frame)
        if i in {0, len(time_samples) // 3, 2 * len(time_samples) // 3, len(time_samples) - 1}:
            contact_sheet_frames.append(cv2.resize(frame, (args.width // 2, args.height // 2)))
        writer.write(frame)
    writer.release()

    if contact_sheet_frames:
        rows = []
        for row in range(0, len(contact_sheet_frames), 2):
            pair = contact_sheet_frames[row : row + 2]
            if len(pair) == 1:
                pair.append(np.zeros_like(pair[0]))
            rows.append(np.concatenate(pair, axis=1))
        cv2.imwrite(str(mp4_path.with_suffix(".sheet.png")), np.concatenate(rows, axis=0))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    usd_path = args.output_dir / f"{args.example}.usd"
    mp4_path = args.output_dir / f"{args.example}_preview.mp4"

    if not args.skip_sim:
        run_newton(args, usd_path)
    if not usd_path.exists():
        raise FileNotFoundError(usd_path)
    render_preview(usd_path, mp4_path, args)
    print(f"USD: {usd_path}")
    print(f"MP4: {mp4_path}")


if __name__ == "__main__":
    main()
