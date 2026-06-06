from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--density-video",
        type=Path,
        default=ROOT / "outputs/3d_mpm_density_render/sand3d_density_render.mp4",
        help="MPM density-render video used as the background layer.",
    )
    parser.add_argument(
        "--robot-video",
        type=Path,
        default=ROOT / "outputs/mujoco_newton_mpm_bridge/mujoco_robot_pass.mp4",
        help="MuJoCo-rendered Franka source. The script extracts the white robot links.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/density_franka_composite/franka_on_density.mp4",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=ROOT / "outputs/density_franka_composite/franka_on_density_preview.png",
    )
    parser.add_argument("--robot-alpha", type=float, default=0.96)
    parser.add_argument("--mask-threshold", type=int, default=172)
    parser.add_argument("--min-component-area", type=int, default=260)
    parser.add_argument("--max-component-area-frac", type=float, default=0.16)
    parser.add_argument("--feather", type=int, default=9)
    parser.add_argument(
        "--hide-density-tool",
        action="store_true",
        help="Cover the density-rendered blade outlines before adding Franka.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    density_video = resolve(args.density_video)
    robot_video = resolve(args.robot_video)
    output = resolve(args.output)
    preview = resolve(args.preview)

    density_cap = cv2.VideoCapture(density_video.as_posix())
    robot_cap = cv2.VideoCapture(robot_video.as_posix())
    if not density_cap.isOpened():
        raise RuntimeError(f"Could not open density video: {density_video}")
    if not robot_cap.isOpened():
        raise RuntimeError(f"Could not open robot video: {robot_video}")

    density_frames = int(density_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    robot_frames = int(robot_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(density_cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(density_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(density_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output}")

    preview_frame: np.ndarray | None = None
    for frame_id in range(density_frames):
        ok, base = density_cap.read()
        if not ok:
            break
        robot_id = map_frame(frame_id, max(density_frames, 1), max(robot_frames, 1))
        robot_cap.set(cv2.CAP_PROP_POS_FRAMES, robot_id)
        ok, robot = robot_cap.read()
        if not ok:
            break
        if robot.shape[:2] != base.shape[:2]:
            robot = cv2.resize(robot, (width, height), interpolation=cv2.INTER_AREA)
        if args.hide_density_tool:
            base = remove_density_tool_overlays(base)
        mask = extract_franka_mask(
            robot,
            threshold=args.mask_threshold,
            min_area=args.min_component_area,
            max_area_frac=args.max_component_area_frac,
            feather=args.feather,
        )
        composed = alpha_composite(base, robot, mask, args.robot_alpha)
        writer.write(composed)
        if frame_id == max(0, density_frames // 2):
            preview_frame = composed.copy()

    density_cap.release()
    robot_cap.release()
    writer.release()
    if preview_frame is None:
        raise RuntimeError("No frames were written")
    cv2.imwrite(preview.as_posix(), preview_frame)
    print(f"composite_video={output}")
    print(f"composite_preview={preview}")
    print(f"frames_written={frame_id + 1}")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def map_frame(frame_id: int, base_count: int, source_count: int) -> int:
    if base_count <= 1 or source_count <= 1:
        return 0
    alpha = frame_id / float(base_count - 1)
    return int(np.clip(round(alpha * (source_count - 1)), 0, source_count - 1))


def extract_franka_mask(
    frame: np.ndarray,
    threshold: int,
    min_area: int,
    max_area_frac: float,
    feather: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    b, g, r = cv2.split(frame)
    channel_spread = np.maximum.reduce(
        [
            cv2.absdiff(b, g),
            cv2.absdiff(g, r),
            cv2.absdiff(b, r),
        ]
    )
    candidate = ((value >= threshold) & (saturation <= 86) & (channel_spread <= 76)).astype(np.uint8) * 255
    candidate[:44, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    max_area = int(frame.shape[0] * frame.shape[1] * max_area_frac)
    kept = np.zeros_like(candidate)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            kept[labels == label] = 255

    kept = cv2.dilate(kept, kernel, iterations=1)
    if feather > 0:
        size = feather if feather % 2 == 1 else feather + 1
        kept = cv2.GaussianBlur(kept, (size, size), 0)
    return kept.astype(np.float32) / 255.0


def alpha_composite(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    a = np.clip(mask[:, :, None] * alpha, 0.0, 1.0)
    blended = base.astype(np.float32) * (1.0 - a) + overlay.astype(np.float32) * a
    return np.clip(blended, 0, 255).astype(np.uint8)


def remove_density_tool_overlays(frame: np.ndarray) -> np.ndarray:
    cleaned = frame.copy()
    for rect in [(334, 144, 192, 260), (330, 602, 190, 38)]:
        x, y, w, h = rect
        roi = cleaned[y : y + h, x : x + w]
        if roi.size:
            cleaned[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (31, 31), 0)
    return cleaned


if __name__ == "__main__":
    main()
