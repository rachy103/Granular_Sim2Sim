from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .mpm3d import ToolState3D
from .viz import tool_corners, write_contact_sheet, write_video


DOMAIN_X = (0.12, 0.92)
DOMAIN_Y = (0.08, 0.48)
DOMAIN_Z = (0.02, 0.50)


def _stable_noise(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    noise = (
        0.52
        + 0.20 * np.sin(xx * 0.063 + yy * 0.017)
        + 0.14 * np.sin(xx * 0.021 - yy * 0.071 + 1.7)
        + 0.08 * np.sin(xx * 0.137 + yy * 0.109 + 0.4)
    )
    return np.clip(noise, 0.0, 1.0).astype(np.float32)


def _accumulate_top(pos: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    density = np.zeros((h, w), dtype=np.float32)
    height = np.zeros((h, w), dtype=np.float32)
    ix = np.clip(((pos[:, 0] - DOMAIN_X[0]) / (DOMAIN_X[1] - DOMAIN_X[0]) * w).astype(np.int32), 0, w - 1)
    iy = np.clip(((pos[:, 1] - DOMAIN_Y[0]) / (DOMAIN_Y[1] - DOMAIN_Y[0]) * h).astype(np.int32), 0, h - 1)
    np.add.at(density, (h - 1 - iy, ix), 1.0)
    np.maximum.at(height, (h - 1 - iy, ix), pos[:, 2])
    density = cv2.GaussianBlur(density, (0, 0), 3.0)
    height = cv2.GaussianBlur(height, (0, 0), 4.0)
    return density, height


def _accumulate_side(pos: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    density = np.zeros((h, w), dtype=np.float32)
    ix = np.clip(((pos[:, 0] - DOMAIN_X[0]) / (DOMAIN_X[1] - DOMAIN_X[0]) * w).astype(np.int32), 0, w - 1)
    iz = np.clip(((pos[:, 2] - DOMAIN_Z[0]) / (DOMAIN_Z[1] - DOMAIN_Z[0]) * h).astype(np.int32), 0, h - 1)
    np.add.at(density, (h - 1 - iz, ix), 1.0)
    return cv2.GaussianBlur(density, (0, 0), 2.0)


def _sand_image_from_density(density: np.ndarray, height: np.ndarray | None = None) -> np.ndarray:
    d = density / max(1.0e-6, float(np.percentile(density, 98)))
    d = np.clip(d, 0.0, 1.0)
    noise = _stable_noise(*density.shape)
    if height is None:
        hnorm = d
        relief = np.ones_like(d)
    else:
        hnorm = np.clip((height - 0.045) / 0.260, 0.0, 1.0)
        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=5)
        relief = np.clip(0.92 - 3.0 * gx - 2.0 * gy, 0.72, 1.18)
    r = (96 + 122 * d + 34 * noise + 18 * hnorm) * relief
    g = (70 + 82 * d + 23 * noise + 12 * hnorm) * relief
    b = (31 + 38 * d + 12 * noise) * relief
    img = np.stack([b, g, r], axis=2)
    mask = d > 0.025
    img[~mask] = np.array([28, 31, 32], dtype=np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _map_point(p: np.ndarray, rect: tuple[int, int, int, int], axes: tuple[int, int], limits) -> tuple[int, int]:
    x0, y0, w, h = rect
    x = x0 + int((float(p[axes[0]]) - limits[0][0]) / (limits[0][1] - limits[0][0]) * w)
    y = y0 + h - int((float(p[axes[1]]) - limits[1][0]) / (limits[1][1] - limits[1][0]) * h)
    return x, y


def _draw_tool(frame: np.ndarray, tool: ToolState3D, rect: tuple[int, int, int, int], axes, limits) -> None:
    corners = tool_corners(tool)
    edges = [(0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7), (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7)]
    pts = [_map_point(c, rect, axes, limits) for c in corners]
    for a, b in edges:
        cv2.line(frame, pts[a], pts[b], (235, 238, 235), 2, lineType=cv2.LINE_AA)


def _draw_force_plot(frame: np.ndarray, hist: list[float], rect: tuple[int, int, int, int]) -> None:
    x0, y0, w, h = rect
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (55, 59, 60), 1)
    if not hist:
        return
    arr = np.asarray(hist[-120:], dtype=np.float32)
    peak = max(1.0, float(np.percentile(arr, 95)))
    pts = []
    for i, f in enumerate(arr):
        x = x0 + 8 + int(i * (w - 16) / max(1, len(arr) - 1))
        y = y0 + h - 10 - int(np.clip(float(f) / peak, 0.0, 1.0) * (h - 22))
        pts.append((x, y))
    if len(pts) > 1:
        cv2.polylines(frame, [np.asarray(pts, dtype=np.int32)], False, (66, 174, 255), 2, lineType=cv2.LINE_AA)


def render_density_frame(
    pos: np.ndarray,
    tool: ToolState3D,
    reaction: np.ndarray,
    force_history: list[float],
    frame_id: int,
    sim_time: float,
    width: int = 1280,
    height: int = 720,
) -> np.ndarray:
    frame = np.full((height, width, 3), (22, 24, 25), dtype=np.uint8)
    top_rect = (34, 108, 820, 382)
    side_rect = (34, 542, 820, 132)
    plot_rect = (910, 536, 310, 112)

    top_density, top_height = _accumulate_top(pos, (220, 440))
    top_img = cv2.resize(_sand_image_from_density(top_density, top_height), (top_rect[2], top_rect[3]), interpolation=cv2.INTER_LINEAR)
    side_density = _accumulate_side(pos, (110, 440))
    side_img = cv2.resize(_sand_image_from_density(side_density), (side_rect[2], side_rect[3]), interpolation=cv2.INTER_LINEAR)
    frame[top_rect[1] : top_rect[1] + top_rect[3], top_rect[0] : top_rect[0] + top_rect[2]] = top_img
    frame[side_rect[1] : side_rect[1] + side_rect[3], side_rect[0] : side_rect[0] + side_rect[2]] = side_img

    cv2.rectangle(frame, (top_rect[0], top_rect[1]), (top_rect[0] + top_rect[2], top_rect[1] + top_rect[3]), (64, 68, 68), 1)
    cv2.rectangle(frame, (side_rect[0], side_rect[1]), (side_rect[0] + side_rect[2], side_rect[1] + side_rect[3]), (64, 68, 68), 1)
    _draw_tool(frame, tool, top_rect, (0, 1), (DOMAIN_X, DOMAIN_Y))
    _draw_tool(frame, tool, side_rect, (0, 2), (DOMAIN_X, DOMAIN_Z))

    mag = float(np.linalg.norm(reaction[:3]))
    cv2.putText(frame, "3D MPM sand density render", (28, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.84, (238, 242, 240), 2, cv2.LINE_AA)
    cv2.putText(frame, f"frame {frame_id:03d}   t={sim_time:5.3f}s   reaction |F|={mag:5.2f}", (28, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (238, 226, 190), 2, cv2.LINE_AA)
    cv2.putText(frame, "top density/height field", (top_rect[0], top_rect[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (218, 222, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "side density projection", (side_rect[0], side_rect[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (218, 222, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "reaction norm", (plot_rect[0], plot_rect[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (218, 222, 220), 1, cv2.LINE_AA)
    _draw_force_plot(frame, force_history, plot_rect)

    return frame


def write_density_video(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    write_video(path, frames, fps=fps)


def write_density_sheet(path: Path, frames: list[np.ndarray]) -> None:
    write_contact_sheet(path, frames, count=5)
