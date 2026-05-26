from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .mpm3d import ToolState3D


DOMAIN_X = (0.12, 0.92)
DOMAIN_Y = (0.08, 0.48)
DOMAIN_Z = (0.02, 0.50)


def _map_2d(
    a: float,
    b: float,
    rect: tuple[int, int, int, int],
    lim_a: tuple[float, float],
    lim_b: tuple[float, float],
) -> tuple[int, int]:
    x0, y0, w, h = rect
    px = x0 + int((a - lim_a[0]) / (lim_a[1] - lim_a[0]) * w)
    py = y0 + h - int((b - lim_b[0]) / (lim_b[1] - lim_b[0]) * h)
    return px, py


def tool_corners(tool: ToolState3D) -> np.ndarray:
    c = float(np.cos(tool.angle))
    s = float(np.sin(tool.angle))
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * tool.half[0], sy * tool.half[1], sz * tool.half[2]], dtype=np.float32)
                world = tool.center + np.array(
                    [c * local[0] + s * local[2], local[1], -s * local[0] + c * local[2]],
                    dtype=np.float32,
                )
                corners.append(world)
    return np.asarray(corners, dtype=np.float32)


def _draw_tool_projection(
    frame: np.ndarray,
    tool: ToolState3D,
    rect: tuple[int, int, int, int],
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
    color: tuple[int, int, int] = (226, 235, 236),
) -> None:
    corners = tool_corners(tool)
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (3, 1),
        (3, 2),
        (3, 7),
        (5, 1),
        (5, 4),
        (5, 7),
        (6, 2),
        (6, 4),
        (6, 7),
    ]
    pts = [_map_2d(p[axes[0]], p[axes[1]], rect, limits[0], limits[1]) for p in corners]
    for a, b in edges:
        cv2.line(frame, pts[a], pts[b], color, 1, lineType=cv2.LINE_AA)


def _draw_points(
    frame: np.ndarray,
    pos: np.ndarray,
    rect: tuple[int, int, int, int],
    axes: tuple[int, int],
    color_axis: int,
    limits: tuple[tuple[float, float], tuple[float, float]],
    radius: int = 1,
) -> None:
    order = np.argsort(pos[:, color_axis])
    low, high = float(pos[:, color_axis].min()), float(pos[:, color_axis].max())
    denom = max(1.0e-6, high - low)
    palette = np.array(
        [
            [52, 92, 132],
            [66, 122, 165],
            [82, 152, 198],
            [106, 185, 226],
            [150, 213, 240],
        ],
        dtype=np.uint8,
    )
    for p in pos[order]:
        u = np.clip((float(p[color_axis]) - low) / denom, 0.0, 1.0)
        c = palette[min(len(palette) - 1, int(u * len(palette)))]
        px, py = _map_2d(float(p[axes[0]]), float(p[axes[1]]), rect, limits[0], limits[1])
        if rect[0] <= px < rect[0] + rect[2] and rect[1] <= py < rect[1] + rect[3]:
            cv2.circle(frame, (px, py), radius, (int(c[0]), int(c[1]), int(c[2])), -1, lineType=cv2.LINE_AA)


def _draw_force_arrow(
    frame: np.ndarray,
    tool: ToolState3D,
    reaction: np.ndarray,
    rect: tuple[int, int, int, int],
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
    scale: float,
) -> None:
    start = _map_2d(tool.center[axes[0]], tool.center[axes[1]], rect, limits[0], limits[1])
    vec = np.asarray(reaction, dtype=np.float32)
    end_world = tool.center.copy()
    end_world[axes[0]] += vec[axes[0]] * scale
    end_world[axes[1]] += vec[axes[1]] * scale
    end = _map_2d(float(end_world[axes[0]]), float(end_world[axes[1]]), rect, limits[0], limits[1])
    cv2.arrowedLine(frame, start, end, (76, 176, 255), 2, tipLength=0.22, line_type=cv2.LINE_AA)


def render_frame_3d(
    pos: np.ndarray,
    tool: ToolState3D,
    reaction: np.ndarray,
    force_history: list[float],
    frame_id: int,
    sim_time: float,
    width: int = 1280,
    height: int = 720,
) -> np.ndarray:
    frame = np.full((height, width, 3), (22, 24, 26), dtype=np.uint8)
    top_rect = (34, 104, 760, 340)
    side_rect = (34, 502, 760, 168)
    front_rect = (850, 104, 370, 340)
    plot_rect = (850, 510, 370, 134)

    for rect, title in [
        (top_rect, "top view: x-y, color = height"),
        (side_rect, "side view: x-z, color = depth"),
        (front_rect, "front view: y-z, color = x"),
    ]:
        x0, y0, w, h = rect
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (50, 55, 58), 1)
        cv2.putText(frame, title, (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (222, 228, 226), 1, cv2.LINE_AA)

    cv2.putText(frame, "3D MLS-MPM granular bed + SDF blade", (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (235, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(frame, f"frame {frame_id:03d}   t={sim_time:5.3f}s", (26, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 216, 216), 1, cv2.LINE_AA)
    cv2.putText(frame, f"reaction |F|={np.linalg.norm(reaction[:3]):6.2f}", (850, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 224, 190), 2, cv2.LINE_AA)

    _draw_points(frame, pos, top_rect, (0, 1), 2, (DOMAIN_X, DOMAIN_Y), radius=1)
    _draw_points(frame, pos, side_rect, (0, 2), 1, (DOMAIN_X, DOMAIN_Z), radius=1)
    _draw_points(frame, pos, front_rect, (1, 2), 0, (DOMAIN_Y, DOMAIN_Z), radius=1)

    _draw_tool_projection(frame, tool, top_rect, (0, 1), (DOMAIN_X, DOMAIN_Y))
    _draw_tool_projection(frame, tool, side_rect, (0, 2), (DOMAIN_X, DOMAIN_Z))
    _draw_tool_projection(frame, tool, front_rect, (1, 2), (DOMAIN_Y, DOMAIN_Z))

    arrow_scale = 0.0025
    _draw_force_arrow(frame, tool, reaction[:3], top_rect, (0, 1), (DOMAIN_X, DOMAIN_Y), arrow_scale)
    _draw_force_arrow(frame, tool, reaction[:3], side_rect, (0, 2), (DOMAIN_X, DOMAIN_Z), arrow_scale)
    _draw_force_arrow(frame, tool, reaction[:3], front_rect, (1, 2), (DOMAIN_Y, DOMAIN_Z), arrow_scale)

    x0, y0, w, h = plot_rect
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (50, 55, 58), 1)
    cv2.putText(frame, "reaction norm", (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (222, 228, 226), 1, cv2.LINE_AA)
    if force_history:
        hist = np.asarray(force_history[-120:], dtype=np.float32)
        peak = max(1.0, float(np.percentile(hist, 95)))
        pts = []
        for i, f in enumerate(hist):
            px = x0 + 8 + int(i * (w - 16) / max(1, len(hist) - 1))
            py = y0 + h - 10 - int(min(float(f) / peak, 1.0) * (h - 22))
            pts.append((px, py))
        if len(pts) > 1:
            cv2.polylines(frame, [np.asarray(pts, dtype=np.int32)], False, (76, 176, 255), 2, lineType=cv2.LINE_AA)

    return frame


def write_video(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    if not frames:
        raise ValueError("No frames to write")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def write_contact_sheet(path: Path, frames: list[np.ndarray], count: int = 5) -> None:
    if not frames:
        raise ValueError("No frames for sheet")
    ids = np.linspace(0, len(frames) - 1, count).astype(int)
    thumbs = []
    for fid in ids:
        thumb = cv2.resize(frames[fid], (384, 216), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"frame {fid:03d}", (14, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (240, 240, 240), 2, cv2.LINE_AA)
        thumbs.append(thumb)
    sheet = np.hstack(thumbs)
    cv2.imwrite(path.as_posix(), sheet)
