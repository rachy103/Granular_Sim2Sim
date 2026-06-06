from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm.mpm3d import ToolState3D
from scripts.render_density_franka_two_view_overlay import (
    expand_to_aspect,
    reset_home,
    write_scene_xml,
    zoom_density_panel,
)
from scripts.run_3d_blade_demo import blade_state


DEFAULT_CONFIG = ROOT / "configs/rendering/density_mujoco_eef_render_fixed.json"
TOP_RECT = (34, 108, 820, 382)
SIDE_RECT = (34, 542, 820, 132)
DOMAIN_X = (0.12, 0.92)
DOMAIN_Y = (0.08, 0.48)
DOMAIN_Z = (0.02, 0.50)

LIGHT_DIR = np.asarray([-0.35, -0.55, 0.78], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)


@dataclass(frozen=True)
class MeshPart:
    name: str
    local_vertices: np.ndarray
    faces: np.ndarray
    base_bgr: np.ndarray
    edge_bgr: tuple[int, int, int]
    outline: bool
    priority: int


@dataclass(frozen=True)
class RenderView:
    name: str
    rect: tuple[int, int, int, int]
    axes: tuple[int, int]
    limits: tuple[tuple[float, float], tuple[float, float]]
    crop: tuple[int, int, int, int]
    depth_axis: int
    depth_sign: float


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, remaining = pre_parser.parse_known_args()
    config = load_render_config(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument(
        "--density-video",
        type=Path,
        default=config_path(config, "density_video", ROOT / "outputs/3d_mpm_density_render/sand3d_density_render.mp4"),
    )
    parser.add_argument("--output-dir", type=Path, default=config_path(config, "output_dir", ROOT / "outputs/density_mujoco_eef_render"))
    parser.add_argument("--frames", type=int, default=int(config.get("frames", 60)))
    parser.add_argument("--substeps-per-frame", type=int, default=int(config.get("substeps_per_frame", 34)))
    parser.add_argument("--mpm-dt", type=float, default=float(config.get("mpm_dt", 8.0e-4)))
    parser.add_argument("--robot-alpha", type=float, default=float(config.get("robot_alpha", 0.98)))
    parser.add_argument("--crop-mode", choices=["eef_path", "full"], default=str(config.get("crop_mode", "eef_path")))
    parser.add_argument("--top-crop-margin", type=int, default=int(config.get("top_crop_margin", 34)))
    parser.add_argument("--side-crop-margin", type=int, default=int(config.get("side_crop_margin", 20)))
    parser.add_argument("--show-center", action="store_true", default=bool(config.get("show_center", False)))
    parser.add_argument("--skip-property-overlay", action="store_true", default=bool(config.get("skip_property_overlay", False)))
    parser.add_argument(
        "--property-csv",
        type=Path,
        default=config_path(
            config,
            "property_csv",
            ROOT / "outputs/online_mohr_coulomb_bestval_quick/rollout_predictions.csv",
        ),
    )
    args = parser.parse_args(remaining)
    args.config = pre_args.config
    return args


def load_render_config(path: Path) -> dict[str, object]:
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        return {}
    with resolved.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return dict(raw.get("density_mujoco_eef_render", raw))


def config_path(config: dict[str, object], key: str, default: Path) -> Path:
    value = config.get(key)
    if value is None:
        return default
    return Path(str(value))


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_xml = out_dir / "mujoco_eef_render_scene.xml"
    video_path = out_dir / "density_mujoco_eef_render.mp4"
    preview_path = out_dir / "density_mujoco_eef_render_preview.png"
    sheet_path = out_dir / "density_mujoco_eef_render_sheet.png"
    metadata_path = out_dir / "density_mujoco_eef_render_metadata.json"
    property_video = out_dir / "density_mujoco_eef_property_overlay.mp4"
    property_preview = out_dir / "density_mujoco_eef_property_overlay_preview.png"

    write_scene_xml(scene_xml)
    model = mujoco.MjModel.from_xml_path(scene_xml.as_posix())
    data = mujoco.MjData(model)
    reset_home(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mpm3d_blade_center")
    if site_id < 0:
        raise RuntimeError("Missing mpm3d_blade_center site")
    parts = extract_eef_parts(model, data, site_id)

    density_cap = cv2.VideoCapture(resolve(args.density_video).as_posix())
    if not density_cap.isOpened():
        raise RuntimeError(f"Could not open density video: {args.density_video}")
    fps = float(density_cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(density_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(density_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = min(int(density_cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(args.frames))
    top_crop, side_crop = render_crops(
        frame_count=frame_count,
        substeps_per_frame=args.substeps_per_frame,
        mpm_dt=args.mpm_dt,
        crop_mode=args.crop_mode,
        top_margin=args.top_crop_margin,
        side_margin=args.side_crop_margin,
        parts=parts,
    )
    views = [
        RenderView("top", TOP_RECT, (0, 1), (DOMAIN_X, DOMAIN_Y), top_crop, 2, 1.0),
        RenderView("side", SIDE_RECT, (0, 2), (DOMAIN_X, DOMAIN_Z), side_crop, 1, -1.0),
    ]

    writer = cv2.VideoWriter(video_path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    preview: np.ndarray | None = None
    sheet_frames: list[np.ndarray] = []
    sample_ids = set(int(v) for v in np.linspace(0, max(0, frame_count - 1), 5).astype(int))
    written = 0
    for frame_id in range(frame_count):
        ok, frame = density_cap.read()
        if not ok:
            break
        tool_time = (frame_id * args.substeps_per_frame + max(0, args.substeps_per_frame - 1)) * args.mpm_dt
        tool = blade_state(tool_time, args.mpm_dt)
        composed = render_frame(
            frame,
            tool,
            parts,
            views,
            alpha=float(args.robot_alpha),
            show_center=bool(args.show_center),
        )
        writer.write(composed)
        written += 1
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
    metadata = {
        "renderer": "density_mujoco_eef_render",
        "script": Path(__file__).resolve().as_posix(),
        "config": resolve(args.config).as_posix(),
        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
        "density_video": resolve(args.density_video).as_posix(),
        "video": video_path.as_posix(),
        "preview": preview_path.as_posix(),
        "sheet": sheet_path.as_posix(),
        "scene_xml": scene_xml.as_posix(),
        "frames_written": written,
        "fps": fps,
        "width": width,
        "height": height,
        "crop_mode": args.crop_mode,
        "top_crop": list(top_crop),
        "side_crop": list(side_crop),
        "parts": [
            {
                "name": part.name,
                "vertices": int(part.local_vertices.shape[0]),
                "faces": int(part.faces.shape[0]),
                "priority": int(part.priority),
            }
            for part in parts
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"mujoco_eef_render_video={video_path}")
    print(f"mujoco_eef_render_preview={preview_path}")
    print(f"mujoco_eef_render_sheet={sheet_path}")
    print(f"metadata={metadata_path}")
    print(f"frames_written={written}")
    print(f"top_crop={top_crop}")
    print(f"side_crop={side_crop}")

    if not args.skip_property_overlay:
        from scripts.render_rollout_property_overlay import main_with_paths

        main_with_paths(
            video=video_path,
            rollout_csv=resolve(args.property_csv),
            output=property_video,
            preview=property_preview,
            title="Rendered MuJoCo EEF over MPM density + online Mohr-Coulomb property estimation",
        )


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def render_crops(
    frame_count: int,
    substeps_per_frame: int,
    mpm_dt: float,
    crop_mode: str,
    top_margin: int,
    side_margin: int,
    parts: list[MeshPart],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if crop_mode == "full":
        return (0, 0, TOP_RECT[2], TOP_RECT[3]), (0, 0, SIDE_RECT[2], SIDE_RECT[3])

    top_bounds = np.asarray([np.inf, np.inf, -np.inf, -np.inf], dtype=np.float32)
    side_bounds = np.asarray([np.inf, np.inf, -np.inf, -np.inf], dtype=np.float32)
    for frame_id in range(frame_count):
        tool_time = (frame_id * substeps_per_frame + max(0, substeps_per_frame - 1)) * mpm_dt
        tool = blade_state(tool_time, mpm_dt)
        basis = tool_basis(tool.angle)
        for part in parts:
            vertices = tool.center.astype(np.float32) + part.local_vertices @ basis.T
            update_bounds(top_bounds, project_vertices_uncropped(vertices, TOP_RECT, (0, 1), (DOMAIN_X, DOMAIN_Y)))
            update_bounds(side_bounds, project_vertices_uncropped(vertices, SIDE_RECT, (0, 2), (DOMAIN_X, DOMAIN_Z)))

    return (
        crop_from_bounds(top_bounds, TOP_RECT[2], TOP_RECT[3], top_margin),
        crop_from_bounds(side_bounds, SIDE_RECT[2], SIDE_RECT[3], side_margin),
    )


def update_bounds(bounds: np.ndarray, points: np.ndarray) -> None:
    bounds[0] = min(float(bounds[0]), float(points[:, 0].min()))
    bounds[1] = min(float(bounds[1]), float(points[:, 1].min()))
    bounds[2] = max(float(bounds[2]), float(points[:, 0].max()))
    bounds[3] = max(float(bounds[3]), float(points[:, 1].max()))


def project_vertices_uncropped(
    vertices: np.ndarray,
    rect: tuple[int, int, int, int],
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    _x0, _y0, width, height = rect
    x = (vertices[:, axes[0]] - limits[0][0]) / (limits[0][1] - limits[0][0]) * width
    y = height - (vertices[:, axes[1]] - limits[1][0]) / (limits[1][1] - limits[1][0]) * height
    return np.stack([x, y], axis=1)


def crop_from_bounds(bounds: np.ndarray, width: int, height: int, margin: int) -> tuple[int, int, int, int]:
    x_min = float(np.clip(bounds[0] - margin, 0, width - 1))
    y_min = float(np.clip(bounds[1] - margin, 0, height - 1))
    x_max = float(np.clip(bounds[2] + margin, 1, width))
    y_max = float(np.clip(bounds[3] + margin, 1, height))
    x_min, y_min, x_max, y_max = expand_to_aspect(x_min, y_min, x_max, y_max, width, height, width / height)
    return int(round(x_min)), int(round(y_min)), int(round(x_max - x_min)), int(round(y_max - y_min))


def extract_eef_parts(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> list[MeshPart]:
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    capsule_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    selected_bodies = {"hand", "left_finger", "right_finger"}
    selected_names = {"mpm3d_tool_handle", "mpm3d_blade"}
    parts: list[MeshPart] = []

    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        geom_type = int(model.geom_type[geom_id])
        geom_group = int(model.geom_group[geom_id])
        is_visual_eef = body_name in selected_bodies and geom_type == mesh_type and geom_group == 2
        is_custom_tool = geom_name in selected_names and geom_type in {box_type, capsule_type}
        if not (is_visual_eef or is_custom_tool):
            continue

        if geom_type == mesh_type:
            vertices, faces = mesh_geom_arrays(model, geom_id)
        elif geom_type == box_type:
            vertices, faces = box_geom_arrays(model.geom_size[geom_id])
        elif geom_type == capsule_type:
            vertices, faces = capsule_geom_arrays(model.geom_size[geom_id])
        else:
            continue

        local_vertices = geom_vertices_to_site_local(vertices, model, data, geom_id, site_id)
        base_bgr = geom_base_bgr(model, geom_id, custom_tool=is_custom_tool)
        priority = 0 if geom_name == "mpm3d_blade" else 1 if geom_name == "mpm3d_tool_handle" else 2
        edge = (226, 228, 224) if priority < 2 else (66, 68, 68)
        parts.append(
            MeshPart(
                name=geom_name,
                local_vertices=local_vertices.astype(np.float32),
                faces=faces.astype(np.int32),
                base_bgr=base_bgr.astype(np.float32),
                edge_bgr=edge,
                outline=priority < 2,
                priority=priority,
            )
        )

    if not parts:
        raise RuntimeError("No EEF mesh parts were extracted")
    return sorted(parts, key=lambda part: part.priority)


def mesh_geom_arrays(model: mujoco.MjModel, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    mesh_id = int(model.geom_dataid[geom_id])
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    face_adr = int(model.mesh_faceadr[mesh_id])
    face_num = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float32)
    faces = np.asarray(model.mesh_face[face_adr : face_adr + face_num], dtype=np.int32).reshape(-1, 3)
    return vertices, faces


def box_geom_arrays(size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = [float(v) for v in size[:3]]
    vertices = np.asarray(
        [
            [-sx, -sy, -sz],
            [-sx, -sy, sz],
            [-sx, sy, -sz],
            [-sx, sy, sz],
            [sx, -sy, -sz],
            [sx, -sy, sz],
            [sx, sy, -sz],
            [sx, sy, sz],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 7],
            [1, 7, 5],
            [0, 1, 5],
            [0, 5, 4],
            [2, 6, 7],
            [2, 7, 3],
            [0, 2, 3],
            [0, 3, 1],
            [4, 5, 7],
            [4, 7, 6],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def capsule_geom_arrays(size: np.ndarray, segments: int = 18) -> tuple[np.ndarray, np.ndarray]:
    radius = float(size[0])
    half_length = float(size[1])
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    bottom = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, -half_length)],
        axis=1,
    )
    top = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, half_length)],
        axis=1,
    )
    vertices = [*bottom.astype(np.float32), *top.astype(np.float32)]
    bottom_center = len(vertices)
    vertices.append(np.asarray([0.0, 0.0, -half_length - radius], dtype=np.float32))
    top_center = len(vertices)
    vertices.append(np.asarray([0.0, 0.0, half_length + radius], dtype=np.float32))
    faces: list[list[int]] = []
    for idx in range(segments):
        nxt = (idx + 1) % segments
        faces.append([idx, nxt, segments + nxt])
        faces.append([idx, segments + nxt, segments + idx])
        faces.append([bottom_center, nxt, idx])
        faces.append([top_center, segments + idx, segments + nxt])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def geom_vertices_to_site_local(
    vertices: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    site_id: int,
) -> np.ndarray:
    del model
    geom_mat = data.geom_xmat[geom_id].reshape(3, 3)
    site_mat = data.site_xmat[site_id].reshape(3, 3)
    world = data.geom_xpos[geom_id] + vertices @ geom_mat.T
    return (world - data.site_xpos[site_id]) @ site_mat


def geom_base_bgr(model: mujoco.MjModel, geom_id: int, custom_tool: bool) -> np.ndarray:
    if custom_tool:
        rgba = model.geom_rgba[geom_id]
    else:
        mat_id = int(model.geom_matid[geom_id])
        rgba = model.mat_rgba[mat_id] if mat_id >= 0 else model.geom_rgba[geom_id]
    rgb = np.asarray(rgba[:3], dtype=np.float32)
    return np.clip(rgb[::-1] * 255.0, 0, 255)


def render_frame(
    frame: np.ndarray,
    tool: ToolState3D,
    parts: list[MeshPart],
    views: list[RenderView],
    alpha: float,
    show_center: bool,
) -> np.ndarray:
    base = frame.copy()
    for view in views:
        base = zoom_density_panel(base, view.rect, view.crop)

    overlay = base.copy()
    basis = tool_basis(tool.angle)
    world_parts = [(part, tool.center.astype(np.float32) + part.local_vertices @ basis.T) for part in parts]
    for view in views:
        render_view(overlay, world_parts, view)
    if show_center:
        draw_center_marker(overlay, tool.center, views)

    return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)


def tool_basis(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    x_axis = np.asarray([c, 0.0, -s], dtype=np.float32)
    y_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    # The Panda hand and fingers sit at negative site-local z from the custom blade site.
    # Flipping this axis puts the rendered EEF above the blade instead of under the sand.
    z_axis = -np.asarray([s, 0.0, c], dtype=np.float32)
    return np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)


def render_view(
    frame: np.ndarray,
    world_parts: list[tuple[MeshPart, np.ndarray]],
    view: RenderView,
) -> None:
    draw_items: list[tuple[float, int, MeshPart, np.ndarray, np.ndarray]] = []
    for part_idx, (part, vertices) in enumerate(world_parts):
        projected = project_vertices(vertices, view)
        tri_world = vertices[part.faces]
        depths = view.depth_sign * tri_world[:, :, view.depth_axis].mean(axis=1)
        for face_idx, depth in enumerate(depths):
            pts = projected[part.faces[face_idx]]
            if triangle_is_visible(pts, view.rect):
                draw_items.append((float(depth), part_idx, part, pts, tri_world[face_idx]))

    draw_items.sort(key=lambda item: (item[0], item[1]))
    for _depth, _part_idx, part, pts, tri in draw_items:
        color = shaded_color(part.base_bgr, tri)
        cv2.fillConvexPoly(frame, pts.astype(np.int32), color, lineType=cv2.LINE_AA)

    for part, vertices in world_parts:
        if part.outline:
            draw_outline(frame, part, project_vertices(vertices, view), view.rect)


def project_vertices(vertices: np.ndarray, view: RenderView) -> np.ndarray:
    _x0, _y0, width, height = view.rect
    x0, y0, panel_w, panel_h = view.rect
    crop_x, crop_y, crop_w, crop_h = view.crop
    u = (vertices[:, view.axes[0]] - view.limits[0][0]) / (view.limits[0][1] - view.limits[0][0]) * width
    v = height - (vertices[:, view.axes[1]] - view.limits[1][0]) / (view.limits[1][1] - view.limits[1][0]) * height
    x = x0 + (u - crop_x) / max(1, crop_w) * panel_w
    y = y0 + (v - crop_y) / max(1, crop_h) * panel_h
    return np.stack([x, y], axis=1).astype(np.int32)


def triangle_is_visible(points: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    x0, y0, width, height = rect
    x_min = int(points[:, 0].min())
    x_max = int(points[:, 0].max())
    y_min = int(points[:, 1].min())
    y_max = int(points[:, 1].max())
    if x_max < x0 or x_min >= x0 + width or y_max < y0 or y_min >= y0 + height:
        return False
    area = cv2.contourArea(points.astype(np.float32))
    return area >= 0.25


def shaded_color(base_bgr: np.ndarray, triangle: np.ndarray) -> tuple[int, int, int]:
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    norm = float(np.linalg.norm(normal))
    if norm < 1.0e-9:
        intensity = 0.55
    else:
        normal = normal / norm
        lambert = abs(float(np.dot(normal, LIGHT_DIR)))
        intensity = 0.46 + 0.54 * lambert
    color = np.clip(base_bgr * intensity, 0, 255).astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def draw_outline(
    frame: np.ndarray,
    part: MeshPart,
    projected: np.ndarray,
    rect: tuple[int, int, int, int],
) -> None:
    x0, y0, width, height = rect
    clipped = projected[
        (projected[:, 0] >= x0 - width)
        & (projected[:, 0] <= x0 + 2 * width)
        & (projected[:, 1] >= y0 - height)
        & (projected[:, 1] <= y0 + 2 * height)
    ]
    if clipped.shape[0] < 3:
        return
    hull = cv2.convexHull(clipped.astype(np.int32))
    if cv2.contourArea(hull) < 2.0:
        return
    cv2.polylines(frame, [hull], isClosed=True, color=part.edge_bgr, thickness=1, lineType=cv2.LINE_AA)


def draw_center_marker(frame: np.ndarray, center: np.ndarray, views: list[RenderView]) -> None:
    for view in views:
        point = project_vertices(center[None, :], view)[0]
        cv2.circle(frame, tuple(int(v) for v in point), 5, (52, 190, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(int(v) for v in point), 8, (20, 24, 25), 1, cv2.LINE_AA)


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
