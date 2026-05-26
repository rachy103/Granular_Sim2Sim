from __future__ import annotations

import cv2
import mujoco
import numpy as np


def project_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    points: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam_pos = data.cam_xpos[camera_id].copy()
    cam_mat = data.cam_xmat[camera_id].reshape(3, 3).copy()
    local = (points - cam_pos) @ cam_mat
    z = -local[:, 2]
    fovy = np.deg2rad(float(model.cam_fovy[camera_id]))
    fy = 0.5 * height / np.tan(0.5 * fovy)
    fx = fy
    u = fx * local[:, 0] / np.maximum(z, 1.0e-6) + 0.5 * width
    v = 0.5 * height - fy * local[:, 1] / np.maximum(z, 1.0e-6)
    valid = (z > 0.05) & (u >= -8.0) & (u < width + 8.0) & (v >= -8.0) & (v < height + 8.0)
    return np.stack([u, v], axis=1), z, valid


def render_sand_layer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    positions: np.ndarray,
    width: int,
    height: int,
    radius: int = 4,
    blur_sigma: float = 2.0,
    alpha_blur_sigma: float = 1.2,
    alpha_cutoff: float = 0.04,
    alpha_gain: float = 0.60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uv, depth, valid = project_points(model, data, camera_id, positions, width, height)
    uv = uv[valid]
    depth = depth[valid]
    pts = positions[valid]

    density = np.zeros((height, width), dtype=np.float32)
    depth_img = np.full((height, width), np.inf, dtype=np.float32)
    color_sum = np.zeros((height, width, 3), dtype=np.float32)

    z = pts[:, 2]
    z_u = (z - float(z.min())) / max(1.0e-6, float(z.max() - z.min()))
    colors = np.stack(
        [
            132.0 + 64.0 * z_u,
            94.0 + 50.0 * z_u,
            42.0 + 24.0 * z_u,
        ],
        axis=1,
    )

    base_x = np.round(uv[:, 0]).astype(np.int32)
    base_y = np.round(uv[:, 1]).astype(np.int32)
    sigma = max(1.0, radius * 0.45)
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            r2 = float(ox * ox + oy * oy)
            if r2 > radius * radius:
                continue
            weight = float(np.exp(-0.5 * r2 / (sigma * sigma)))
            px = base_x + ox
            py = base_y + oy
            m = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            if not np.any(m):
                continue
            pxm = px[m]
            pym = py[m]
            np.add.at(density, (pym, pxm), weight)
            np.minimum.at(depth_img, (pym, pxm), depth[m])
            for c in range(3):
                np.add.at(color_sum[:, :, c], (pym, pxm), colors[m, c] * weight)

    density_blur = cv2.GaussianBlur(density, (0, 0), blur_sigma)
    color_blur = cv2.GaussianBlur(color_sum, (0, 0), blur_sigma)
    denom = np.maximum(density_blur[:, :, None], 1.0e-5)
    sand = color_blur / denom
    noise = _noise(height, width)
    sand[:, :, 0] += 24.0 * noise
    sand[:, :, 1] += 18.0 * noise
    sand[:, :, 2] += 7.0 * noise

    density_norm = density_blur / max(1.0e-6, float(np.percentile(density_blur, 98.5)))
    alpha = np.clip((density_norm - alpha_cutoff) / alpha_gain, 0.0, 0.92).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), alpha_blur_sigma)
    return np.clip(sand, 0, 255).astype(np.uint8), alpha, depth_img


def render_sand_density_layer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    positions: np.ndarray,
    width: int,
    height: int,
    blur_sigma: float = 4.2,
    alpha_blur_sigma: float = 1.6,
    alpha_cutoff: float = 0.025,
    alpha_gain: float = 0.52,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render MPM material points as a continuous camera-space sand layer.

    This is a preview renderer, not a physically based renderer. It intentionally
    suppresses individual material-point glyphs and favors density/height relief,
    matching the standalone density diagnostic more closely.
    """

    uv, depth, valid = project_points(model, data, camera_id, positions, width, height)
    uv = uv[valid]
    depth = depth[valid]
    pts = positions[valid]
    if pts.size == 0:
        return (
            np.zeros((height, width, 3), dtype=np.uint8),
            np.zeros((height, width), dtype=np.float32),
            np.full((height, width), np.inf, dtype=np.float32),
        )

    density = np.zeros((height, width), dtype=np.float32)
    depth_sum = np.zeros((height, width), dtype=np.float32)
    height_sum = np.zeros((height, width), dtype=np.float32)

    x0 = np.floor(uv[:, 0]).astype(np.int32)
    y0 = np.floor(uv[:, 1]).astype(np.int32)
    fx = uv[:, 0] - x0
    fy = uv[:, 1] - y0

    for ox, wx in ((0, 1.0 - fx), (1, fx)):
        for oy, wy in ((0, 1.0 - fy), (1, fy)):
            px = x0 + ox
            py = y0 + oy
            w = (wx * wy).astype(np.float32)
            m = (px >= 0) & (px < width) & (py >= 0) & (py < height) & (w > 1.0e-5)
            if not np.any(m):
                continue
            pxm = px[m]
            pym = py[m]
            wm = w[m]
            np.add.at(density, (pym, pxm), wm)
            np.add.at(depth_sum, (pym, pxm), depth[m].astype(np.float32) * wm)
            np.add.at(height_sum, (pym, pxm), pts[m, 2].astype(np.float32) * wm)

    density_blur = cv2.GaussianBlur(density, (0, 0), blur_sigma)
    depth_blur = cv2.GaussianBlur(depth_sum, (0, 0), blur_sigma)
    height_blur = cv2.GaussianBlur(height_sum, (0, 0), blur_sigma)
    denom = np.maximum(density_blur, 1.0e-6)
    depth_img = depth_blur / denom
    depth_img[density_blur <= 1.0e-5] = np.inf
    height_img = height_blur / denom

    density_norm = density_blur / max(1.0e-6, float(np.percentile(density_blur, 98.8)))
    density_norm = np.clip(density_norm, 0.0, 1.0)
    hnorm = np.clip((height_img - float(np.percentile(pts[:, 2], 2.0))) / max(1.0e-6, float(np.ptp(pts[:, 2]))), 0.0, 1.0)

    gx = cv2.Sobel(height_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(height_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5)
    relief = np.clip(0.96 - 2.4 * gx - 1.6 * gy, 0.70, 1.22)
    noise = _noise(height, width)

    sand = np.empty((height, width, 3), dtype=np.float32)
    sand[:, :, 0] = (96.0 + 116.0 * density_norm + 36.0 * noise + 18.0 * hnorm) * relief
    sand[:, :, 1] = (70.0 + 78.0 * density_norm + 23.0 * noise + 12.0 * hnorm) * relief
    sand[:, :, 2] = (31.0 + 38.0 * density_norm + 12.0 * noise) * relief

    alpha = np.clip((density_norm - alpha_cutoff) / alpha_gain, 0.0, 0.94).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), alpha_blur_sigma)
    sand[alpha < 0.01] = 0.0
    return np.clip(sand, 0, 255).astype(np.uint8), alpha, depth_img


def composite_sand(
    robot_rgb: np.ndarray,
    robot_depth: np.ndarray,
    sand_rgb: np.ndarray,
    sand_alpha: np.ndarray,
    sand_depth: np.ndarray,
    depth_bias: float = 0.006,
) -> np.ndarray:
    visible = (sand_alpha > 0.01) & np.isfinite(sand_depth) & (sand_depth < robot_depth - depth_bias)
    alpha = np.where(visible, sand_alpha, 0.0)[:, :, None]
    out = robot_rgb.astype(np.float32) * (1.0 - alpha) + sand_rgb.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _noise(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    noise = (
        0.45
        + 0.20 * np.sin(xx * 0.051 + yy * 0.023)
        + 0.12 * np.sin(xx * 0.017 - yy * 0.067 + 1.3)
        + 0.09 * np.sin(xx * 0.109 + yy * 0.093 + 0.8)
    )
    return np.clip(noise, 0.0, 1.0).astype(np.float32)
