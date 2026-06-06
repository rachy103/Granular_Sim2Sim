from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm.mpm3d import ToolState3D
from granular_mpm.viz import tool_corners
from scripts.run_3d_blade_demo import blade_state


TOP_RECT = (34, 108, 820, 382)
SIDE_RECT = (34, 542, 820, 132)
DOMAIN_X = (0.12, 0.92)
DOMAIN_Y = (0.08, 0.48)
DOMAIN_Z = (0.02, 0.50)


@dataclass(frozen=True)
class ProjectedShape:
    name: str
    points: np.ndarray
    fill: tuple[int, int, int]
    edge: tuple[int, int, int]
    thickness: int = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--density-video",
        type=Path,
        default=ROOT / "outputs/3d_mpm_density_render/sand3d_density_render.mp4",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/density_projected_eef_overlay")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--substeps-per-frame", type=int, default=34)
    parser.add_argument("--mpm-dt", type=float, default=8.0e-4)
    parser.add_argument("--robot-alpha", type=float, default=0.96)
    parser.add_argument("--show-center", action="store_true")
    parser.add_argument("--skip-property-overlay", action="store_true")
    parser.add_argument(
        "--property-csv",
        type=Path,
        default=ROOT / "outputs/online_mohr_coulomb_bestval_quick/rollout_predictions.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "density_projected_eef_overlay.mp4"
    preview_path = out_dir / "density_projected_eef_overlay_preview.png"
    sheet_path = out_dir / "density_projected_eef_overlay_sheet.png"
    property_video = out_dir / "density_projected_eef_property_overlay.mp4"
    property_preview = out_dir / "density_projected_eef_property_overlay_preview.png"

    density_cap = cv2.VideoCapture(resolve(args.density_video).as_posix())
    if not density_cap.isOpened():
        raise RuntimeError(f"Could not open density video: {args.density_video}")
    fps = float(density_cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(density_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(density_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = min(int(density_cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(args.frames))
    writer = cv2.VideoWriter(video_path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    preview: np.ndarray | None = None
    sheet_frames: list[np.ndarray] = []
    sample_ids = set(int(v) for v in np.linspace(0, max(0, frame_count - 1), 5).astype(int))

    for frame_id in range(frame_count):
        ok, frame = density_cap.read()
        if not ok:
            break
        tool_time = (frame_id * args.substeps_per_frame + max(0, args.substeps_per_frame - 1)) * args.mpm_dt
        tool = blade_state(tool_time, args.mpm_dt)
        composed = draw_projected_eef(frame, tool, alpha=float(args.robot_alpha), show_center=bool(args.show_center))
        writer.write(composed)
        if frame_id in sample_ids:
            sheet_frames.append(composed.copy())
        if frame_id == frame_count // 2:
            preview = composed.copy()

    density_cap.release()
    writer.release()
    if preview is None:
        raise RuntimeError("No frames were written")
    cv2.imwrite(preview_path.as_posix(), preview)
    write_sheet(sheet_path, sheet_frames)
    print(f"projected_eef_video={video_path}")
    print(f"projected_eef_preview={preview_path}")
    print(f"projected_eef_sheet={sheet_path}")
    print(f"frames_written={len(sheet_frames) and frame_count}")

    if not args.skip_property_overlay:
        from scripts.render_rollout_property_overlay import main_with_paths

        main_with_paths(
            video=video_path,
            rollout_csv=resolve(args.property_csv),
            output=property_video,
            preview=property_preview,
            title="Projected EEF density overlay + online Mohr-Coulomb property estimation",
        )


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def draw_projected_eef(frame: np.ndarray, tool: ToolState3D, alpha: float, show_center: bool) -> np.ndarray:
    overlay = frame.copy()
    shapes = eef_shapes(tool)
    draw_shapes_for_view(overlay, shapes, TOP_RECT, (0, 1), (DOMAIN_X, DOMAIN_Y))
    draw_shapes_for_view(overlay, shapes, SIDE_RECT, (0, 2), (DOMAIN_X, DOMAIN_Z))
    if show_center:
        draw_center_marker(overlay, tool.center)
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)


def eef_shapes(tool: ToolState3D) -> list[ProjectedShape]:
    basis = tool_basis(tool.angle)
    x_axis, y_axis, z_axis = basis[:, 0], basis[:, 1], basis[:, 2]
    center = tool.center.astype(np.float32)
    blade = tool_corners(tool)
    handle_start = center + 0.026 * z_axis
    handle_end = center + 0.150 * z_axis
    palm_center = center + 0.150 * z_axis
    knuckle_center = center + 0.215 * z_axis

    shapes = [
        ProjectedShape(
            name="blade",
            points=blade,
            fill=(24, 28, 30),
            edge=(242, 244, 239),
            thickness=2,
        ),
        ProjectedShape(
            name="handle",
            points=np.stack([handle_start, handle_end], axis=0),
            fill=(24, 28, 30),
            edge=(230, 232, 228),
            thickness=5,
        ),
        ProjectedShape(
            name="left_finger",
            points=box_corners(
                center + 0.050 * z_axis + 0.070 * y_axis,
                basis,
                np.array([0.030, 0.010, 0.018], dtype=np.float32),
            ),
            fill=(246, 247, 244),
            edge=(90, 92, 90),
            thickness=1,
        ),
        ProjectedShape(
            name="right_finger",
            points=box_corners(
                center + 0.050 * z_axis - 0.070 * y_axis,
                basis,
                np.array([0.030, 0.010, 0.018], dtype=np.float32),
            ),
            fill=(246, 247, 244),
            edge=(90, 92, 90),
            thickness=1,
        ),
        ProjectedShape(
            name="palm",
            points=box_corners(palm_center, basis, np.array([0.052, 0.055, 0.040], dtype=np.float32)),
            fill=(238, 240, 236),
            edge=(88, 90, 88),
            thickness=1,
        ),
        ProjectedShape(
            name="knuckle",
            points=box_corners(knuckle_center, basis, np.array([0.038, 0.046, 0.026], dtype=np.float32)),
            fill=(218, 221, 218),
            edge=(80, 82, 82),
            thickness=1,
        ),
    ]
    return shapes


def tool_basis(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    x_axis = np.array([c, 0.0, -s], dtype=np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    z_axis = np.array([s, 0.0, c], dtype=np.float32)
    return np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)


def box_corners(center: np.ndarray, basis: np.ndarray, half: np.ndarray) -> np.ndarray:
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=np.float32)
                corners.append(center + basis @ local)
    return np.asarray(corners, dtype=np.float32)


def draw_shapes_for_view(
    frame: np.ndarray,
    shapes: list[ProjectedShape],
    rect: tuple[int, int, int, int],
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    for shape in shapes:
        if shape.name == "handle":
            draw_projected_handle(frame, shape, rect, axes, limits)
        else:
            draw_projected_hull(frame, shape, rect, axes, limits)


def draw_projected_hull(
    frame: np.ndarray,
    shape: ProjectedShape,
    rect: tuple[int, int, int, int],
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    pts = np.asarray([map_point(point, rect, axes, limits) for point in shape.points], dtype=np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(frame, hull, shape.fill, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [hull], isClosed=True, color=shape.edge, thickness=shape.thickness, lineType=cv2.LINE_AA)
    if shape.name in {"palm", "knuckle"}:
        x, y, w, h = cv2.boundingRect(hull)
        if w > 14 and h > 8:
            cv2.line(frame, (x + 3, y + h // 2), (x + w - 4, y + h // 2), (202, 205, 202), 1, cv2.LINE_AA)


def draw_projected_handle(
    frame: np.ndarray,
    shape: ProjectedShape,
    rect: tuple[int, int, int, int],
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    p0 = map_point(shape.points[0], rect, axes, limits)
    p1 = map_point(shape.points[1], rect, axes, limits)
    cv2.line(frame, p0, p1, shape.fill, shape.thickness + 2, cv2.LINE_AA)
    cv2.line(frame, p0, p1, shape.edge, max(1, shape.thickness - 2), cv2.LINE_AA)
    cv2.circle(frame, p0, max(2, shape.thickness // 2), shape.fill, -1, cv2.LINE_AA)
    cv2.circle(frame, p1, max(2, shape.thickness // 2), shape.fill, -1, cv2.LINE_AA)


def map_point(
    point: np.ndarray,
    rect: tuple[int, int, int, int],
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[int, int]:
    x0, y0, w, h = rect
    px = x0 + int((float(point[axes[0]]) - limits[0][0]) / (limits[0][1] - limits[0][0]) * w)
    py = y0 + h - int((float(point[axes[1]]) - limits[1][0]) / (limits[1][1] - limits[1][0]) * h)
    return px, py


def draw_center_marker(frame: np.ndarray, center: np.ndarray) -> None:
    for rect, axes, limits in [
        (TOP_RECT, (0, 1), (DOMAIN_X, DOMAIN_Y)),
        (SIDE_RECT, (0, 2), (DOMAIN_X, DOMAIN_Z)),
    ]:
        p = map_point(center, rect, axes, limits)
        cv2.circle(frame, p, 5, (52, 190, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 7, (20, 24, 25), 1, cv2.LINE_AA)


def write_sheet(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        return
    thumbs = []
    for idx, frame in enumerate(frames):
        thumb = cv2.resize(frame, (512, 288), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"sample {idx + 1}", (14, 266), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2)
        thumbs.append(thumb)
    cv2.imwrite(path.as_posix(), np.vstack(thumbs))


if __name__ == "__main__":
    main()
