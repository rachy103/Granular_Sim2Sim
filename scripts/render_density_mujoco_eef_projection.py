from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from scripts.render_density_franka_two_view_overlay import reset_home, write_scene_xml
from scripts.run_3d_blade_demo import blade_state


TOP_RECT = (34, 108, 820, 382)
SIDE_RECT = (34, 542, 820, 132)
DOMAIN_X = (0.12, 0.92)
DOMAIN_Y = (0.08, 0.48)
DOMAIN_Z = (0.02, 0.50)
TOOL_GEOMS = {"mpm3d_tool_handle", "mpm3d_blade"}
EEF_BODIES = {"hand", "left_finger", "right_finger"}


@dataclass(frozen=True)
class MeshAsset:
    name: str
    vertices_site: np.ndarray
    faces: np.ndarray
    color_bgr: tuple[int, int, int]
    draw_edges: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--density-video",
        type=Path,
        default=ROOT / "outputs/3d_mpm_density_render/sand3d_density_render.mp4",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/density_mujoco_eef_projection")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--substeps-per-frame", type=int, default=34)
    parser.add_argument("--mpm-dt", type=float, default=8.0e-4)
    parser.add_argument("--robot-alpha", type=float, default=0.98)
    parser.add_argument("--edge-alpha", type=float, default=0.38)
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
    scene_xml = out_dir / "mujoco_eef_projection_scene.xml"
    video_path = out_dir / "density_mujoco_eef_projection.mp4"
    preview_path = out_dir / "density_mujoco_eef_projection_preview.png"
    sheet_path = out_dir / "density_mujoco_eef_projection_sheet.png"
    property_video = out_dir / "density_mujoco_eef_property_overlay.mp4"
    property_preview = out_dir / "density_mujoco_eef_property_overlay_preview.png"

    write_scene_xml(scene_xml)
    model = mujoco.MjModel.from_xml_path(scene_xml.as_posix())
    data = mujoco.MjData(model)
    reset_home(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mpm3d_blade_center")
    if site_id < 0:
        raise RuntimeError("Missing mpm3d_blade_center site")
    assets = collect_eef_assets(model, data, site_id)
    print(f"mesh_assets={len(assets)}")
    print(f"mesh_faces={sum(asset.faces.shape[0] for asset in assets)}")

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
        rendered = render_eef_projection(frame, assets, tool.center, tool.angle, args.robot_alpha, args.edge_alpha)
        writer.write(rendered)
        if frame_id in sample_ids:
            sheet_frames.append(rendered.copy())
        if frame_id == frame_count // 2:
            preview = rendered.copy()

    density_cap.release()
    writer.release()
    if preview is None:
        raise RuntimeError("No frames were written")
    cv2.imwrite(preview_path.as_posix(), preview)
    write_sheet(sheet_path, sheet_frames)
    print(f"mujoco_eef_video={video_path}")
    print(f"mujoco_eef_preview={preview_path}")
    print(f"mujoco_eef_sheet={sheet_path}")
    print(f"frames_written={frame_count}")

    if not args.skip_property_overlay:
        from scripts.render_rollout_property_overlay import main_with_paths

        main_with_paths(
            video=video_path,
            rollout_csv=resolve(args.property_csv),
            output=property_video,
            preview=property_preview,
            title="Rendered MuJoCo EEF projection + online Mohr-Coulomb property estimation",
        )


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def collect_eef_assets(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> list[MeshAsset]:
    site_pos = data.site_xpos[site_id].copy()
    site_mat = data.site_xmat[site_id].reshape(3, 3).copy()
    assets: list[MeshAsset] = []
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        group = int(model.geom_group[geom_id])
        include_visual_mesh = body_name in EEF_BODIES and group == 2
        include_tool = geom_name in TOOL_GEOMS
        if not include_visual_mesh and not include_tool:
            continue

        geom_type = int(model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            vertices_world, faces = mesh_geom_world(model, data, geom_id)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            vertices_world, faces = box_geom_world(model, data, geom_id)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            vertices_world, faces = capsule_geom_world(model, data, geom_id)
        else:
            continue
        vertices_site = (vertices_world - site_pos) @ site_mat
        color = geom_color_bgr(model, geom_id, geom_name, body_name)
        assets.append(
            MeshAsset(
                name=geom_name or f"{body_name}_{geom_id}",
                vertices_site=vertices_site.astype(np.float32),
                faces=faces.astype(np.int32),
                color_bgr=color,
                draw_edges=include_tool,
            )
        )
    if not assets:
        raise RuntimeError("No EEF mesh assets collected")
    return assets


def mesh_geom_world(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    mesh_id = int(model.geom_dataid[geom_id])
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    face_adr = int(model.mesh_faceadr[mesh_id])
    face_num = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float64)
    faces = np.asarray(model.mesh_face[face_adr : face_adr + face_num], dtype=np.int32)
    geom_pos = data.geom_xpos[geom_id].copy()
    geom_mat = data.geom_xmat[geom_id].reshape(3, 3).copy()
    return geom_pos + vertices @ geom_mat.T, faces


def box_geom_world(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = model.geom_size[geom_id, :3]
    vertices = np.asarray(
        [[x * sx, y * sy, z * sz] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int32,
    )
    geom_pos = data.geom_xpos[geom_id].copy()
    geom_mat = data.geom_xmat[geom_id].reshape(3, 3).copy()
    return geom_pos + vertices @ geom_mat.T, faces


def capsule_geom_world(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int, segments: int = 18) -> tuple[np.ndarray, np.ndarray]:
    radius = float(model.geom_size[geom_id, 0])
    half_length = float(model.geom_size[geom_id, 1])
    rings = []
    for z in (-half_length, half_length):
        ring = []
        for idx in range(segments):
            theta = 2.0 * np.pi * idx / segments
            ring.append([radius * np.cos(theta), radius * np.sin(theta), z])
        rings.append(ring)
    vertices = np.asarray(rings[0] + rings[1] + [[0.0, 0.0, -half_length - radius], [0.0, 0.0, half_length + radius]], dtype=np.float64)
    bottom_tip = 2 * segments
    top_tip = bottom_tip + 1
    faces: list[list[int]] = []
    for idx in range(segments):
        nxt = (idx + 1) % segments
        faces.append([idx, nxt, segments + nxt])
        faces.append([idx, segments + nxt, segments + idx])
        faces.append([bottom_tip, nxt, idx])
        faces.append([top_tip, segments + idx, segments + nxt])
    geom_pos = data.geom_xpos[geom_id].copy()
    geom_mat = data.geom_xmat[geom_id].reshape(3, 3).copy()
    return geom_pos + vertices @ geom_mat.T, np.asarray(faces, dtype=np.int32)


def geom_color_bgr(model: mujoco.MjModel, geom_id: int, geom_name: str, body_name: str) -> tuple[int, int, int]:
    if geom_name == "mpm3d_blade":
        return (20, 23, 26)
    if geom_name == "mpm3d_tool_handle":
        return (28, 31, 33)
    if body_name in {"hand", "left_finger", "right_finger"}:
        return (236, 238, 235)
    rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float32)
    return tuple(int(np.clip(255.0 * rgba[i], 0, 255)) for i in (2, 1, 0))


def render_eef_projection(
    frame: np.ndarray,
    assets: list[MeshAsset],
    center: np.ndarray,
    angle: float,
    alpha: float,
    edge_alpha: float,
) -> np.ndarray:
    overlay = frame.copy()
    basis = tool_basis(angle)
    draw_assets_for_view(overlay, assets, center, basis, TOP_RECT, (0, 1), (DOMAIN_X, DOMAIN_Y), depth_axis=2, near_sign=1.0, edge_alpha=edge_alpha)
    draw_assets_for_view(overlay, assets, center, basis, SIDE_RECT, (0, 2), (DOMAIN_X, DOMAIN_Z), depth_axis=1, near_sign=-1.0, edge_alpha=edge_alpha)
    return cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0.0)


def draw_assets_for_view(
    frame: np.ndarray,
    assets: list[MeshAsset],
    center: np.ndarray,
    basis: np.ndarray,
    rect: tuple[int, int, int, int],
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
    depth_axis: int,
    near_sign: float,
    edge_alpha: float,
) -> None:
    triangles: list[tuple[float, np.ndarray, tuple[int, int, int], bool]] = []
    light = normalized(np.asarray([-0.35, -0.55, 0.78], dtype=np.float32))
    for asset in assets:
        vertices = center[None, :] + asset.vertices_site @ basis.T
        pixels = map_points(vertices, rect, axes, limits)
        for face in asset.faces:
            tri_world = vertices[face]
            tri_pixels = pixels[face]
            if np.max(tri_pixels[:, 0]) < rect[0] - 8 or np.min(tri_pixels[:, 0]) > rect[0] + rect[2] + 8:
                continue
            if np.max(tri_pixels[:, 1]) < rect[1] - 8 or np.min(tri_pixels[:, 1]) > rect[1] + rect[3] + 8:
                continue
            area = abs(cv2.contourArea(tri_pixels.astype(np.float32)))
            if area < 0.15:
                continue
            normal = np.cross(tri_world[1] - tri_world[0], tri_world[2] - tri_world[0])
            n_norm = float(np.linalg.norm(normal))
            if n_norm < 1.0e-9:
                continue
            normal = normal / n_norm
            intensity = 0.42 + 0.58 * abs(float(np.dot(normal, light)))
            color = shade(asset.color_bgr, intensity)
            depth = near_sign * float(np.mean(tri_world[:, depth_axis]))
            triangles.append((depth, tri_pixels.astype(np.int32), color, asset.draw_edges))
    triangles.sort(key=lambda item: item[0])
    for _depth, tri, color, draw_edges in triangles:
        cv2.fillConvexPoly(frame, tri, color, lineType=cv2.LINE_AA)
        if draw_edges and edge_alpha > 0.0:
            edge = tuple(int((1.0 - edge_alpha) * c + edge_alpha * 245) for c in color)
            cv2.polylines(frame, [tri], True, edge, 1, cv2.LINE_AA)


def tool_basis(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    x_axis = np.array([c, 0.0, -s], dtype=np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    z_axis = np.array([s, 0.0, c], dtype=np.float32)
    return np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)


def map_points(
    points: np.ndarray,
    rect: tuple[int, int, int, int],
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    x0, y0, w, h = rect
    px = x0 + ((points[:, axes[0]] - limits[0][0]) / (limits[0][1] - limits[0][0]) * w)
    py = y0 + h - ((points[:, axes[1]] - limits[1][0]) / (limits[1][1] - limits[1][0]) * h)
    return np.stack([px, py], axis=1).astype(np.int32)


def normalized(v: np.ndarray) -> np.ndarray:
    return v / max(1.0e-9, float(np.linalg.norm(v)))


def shade(color_bgr: tuple[int, int, int], intensity: float) -> tuple[int, int, int]:
    ambient = np.asarray([18.0, 20.0, 22.0], dtype=np.float32)
    color = np.asarray(color_bgr, dtype=np.float32)
    shaded = ambient + color * float(np.clip(intensity, 0.0, 1.25))
    return tuple(int(np.clip(v, 0, 255)) for v in shaded)


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
