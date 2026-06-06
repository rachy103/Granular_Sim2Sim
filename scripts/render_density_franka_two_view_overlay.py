from __future__ import annotations

import argparse
import os
import sys
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

from granular_mpm.composite_render import project_points
from granular_mpm.viz import tool_corners
from scripts.run_3d_blade_demo import blade_state


MENAGERIE_PANDA = ROOT / "mujoco_menagerie" / "franka_emika_panda"
PANDA_XML = MENAGERIE_PANDA / "panda.xml"
TOP_RECT = (34, 108, 820, 382)
SIDE_RECT = (34, 542, 820, 132)
DOMAIN_X = (0.12, 0.92)
DOMAIN_Y = (0.08, 0.48)
DOMAIN_Z = (0.02, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--density-video",
        type=Path,
        default=ROOT / "outputs/3d_mpm_density_render/sand3d_density_render.mp4",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/density_franka_two_view_zoomed")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--substeps-per-frame", type=int, default=34)
    parser.add_argument("--mpm-dt", type=float, default=8.0e-4)
    parser.add_argument("--ik-iterations", type=int, default=90)
    parser.add_argument("--mask-threshold", type=int, default=150)
    parser.add_argument("--robot-alpha", type=float, default=0.96)
    parser.add_argument("--crop-mode", choices=["eef_path", "full"], default="eef_path")
    parser.add_argument("--top-crop-margin", type=int, default=34)
    parser.add_argument("--side-crop-margin", type=int, default=20)
    parser.add_argument("--show-full-robot", action="store_true")
    parser.add_argument("--include-wrist-link", action="store_true")
    parser.add_argument("--property-csv", type=Path, default=ROOT / "outputs/online_mohr_coulomb_bestval_quick/rollout_predictions.csv")
    parser.add_argument("--skip-property-overlay", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_xml = out_dir / "density_franka_two_view_scene.xml"
    video_path = out_dir / "density_franka_two_view.mp4"
    preview_path = out_dir / "density_franka_two_view_preview.png"
    sheet_path = out_dir / "density_franka_two_view_sheet.png"
    property_video = out_dir / "density_franka_two_view_property_overlay.mp4"
    property_preview = out_dir / "density_franka_two_view_property_overlay_preview.png"

    write_scene_xml(scene_xml)
    model = mujoco.MjModel.from_xml_path(scene_xml.as_posix())
    if not args.show_full_robot:
        set_eef_only_visibility(model, include_wrist_link=args.include_wrist_link)
    data = mujoco.MjData(model)
    reset_home(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mpm3d_blade_center")
    top_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "density_top_cam")
    side_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "density_side_cam")
    if min(site_id, top_camera_id, side_camera_id) < 0:
        raise RuntimeError("Missing required MuJoCo site/camera")

    top_renderer = mujoco.Renderer(model, height=TOP_RECT[3], width=TOP_RECT[2])
    side_renderer = mujoco.Renderer(model, height=SIDE_RECT[3], width=SIDE_RECT[2])
    disable_hidden_geom_group(top_renderer)
    disable_hidden_geom_group(side_renderer)
    density_cap = cv2.VideoCapture(resolve(args.density_video).as_posix())
    if not density_cap.isOpened():
        raise RuntimeError(f"Could not open density video: {args.density_video}")
    fps = float(density_cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(density_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(density_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = min(int(density_cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(args.frames))
    top_crop, side_crop = interaction_crops(
        frame_count=frame_count,
        substeps_per_frame=args.substeps_per_frame,
        mpm_dt=args.mpm_dt,
        crop_mode=args.crop_mode,
        top_margin=args.top_crop_margin,
        side_margin=args.side_crop_margin,
    )
    print(f"top_crop={top_crop}")
    print(f"side_crop={side_crop}")
    writer = cv2.VideoWriter(video_path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    frames_for_sheet: list[np.ndarray] = []
    preview: np.ndarray | None = None
    qpos = data.qpos.copy()
    for frame_id in range(frame_count):
        ok, base = density_cap.read()
        if not ok:
            break
        tool_time = (frame_id * args.substeps_per_frame + max(0, args.substeps_per_frame - 1)) * args.mpm_dt
        tool = blade_state(tool_time, args.mpm_dt)
        base = zoom_density_panel(base, TOP_RECT, top_crop)
        base = zoom_density_panel(base, SIDE_RECT, side_crop)
        qpos = solve_site_pose_ik(model, data, site_id, tool.center, tool.angle, qpos, args.ik_iterations)
        mujoco.mj_forward(model, data)
        top_rgb, top_mask = render_robot_masked(
            model,
            data,
            top_renderer,
            top_camera_id,
            site_id,
            args.mask_threshold,
        )
        side_rgb, side_mask = render_robot_masked(
            model,
            data,
            side_renderer,
            side_camera_id,
            site_id,
            args.mask_threshold,
        )
        base = composite_view(
            base,
            top_rgb,
            top_mask,
            TOP_RECT,
            top_crop,
            desired=_map_point(tool.center, TOP_RECT, (0, 1), (DOMAIN_X, DOMAIN_Y), local=True),
            rendered=project_site(model, data, top_camera_id, site_id, TOP_RECT[2], TOP_RECT[3]),
            alpha=args.robot_alpha,
        )
        base = composite_view(
            base,
            side_rgb,
            side_mask,
            SIDE_RECT,
            side_crop,
            desired=_map_point(tool.center, SIDE_RECT, (0, 2), (DOMAIN_X, DOMAIN_Z), local=True),
            rendered=project_site(model, data, side_camera_id, site_id, SIDE_RECT[2], SIDE_RECT[3]),
            alpha=args.robot_alpha,
        )
        draw_shared_contact_marker(base, tool.center, top_crop, side_crop)
        writer.write(base)
        if frame_id in np.linspace(0, max(0, frame_count - 1), 5).astype(int):
            frames_for_sheet.append(base.copy())
        if frame_id == frame_count // 2:
            preview = base.copy()

    density_cap.release()
    writer.release()
    top_renderer.close()
    side_renderer.close()
    if preview is None:
        raise RuntimeError("No frames were written")
    cv2.imwrite(preview_path.as_posix(), preview)
    write_sheet(sheet_path, frames_for_sheet)

    print(f"two_view_video={video_path}")
    print(f"two_view_preview={preview_path}")
    print(f"two_view_sheet={sheet_path}")
    print(f"scene={scene_xml}")

    if not args.skip_property_overlay:
        from scripts.render_rollout_property_overlay import main_with_paths

        main_with_paths(
            video=video_path,
            rollout_csv=resolve(args.property_csv),
            output=property_video,
            preview=property_preview,
        )


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def write_scene_xml(path: Path) -> None:
    text = PANDA_XML.read_text(encoding="utf-8")
    text = text.replace(
        'meshdir="assets"',
        f'meshdir="{(MENAGERIE_PANDA / "assets").as_posix()}"',
        1,
    )
    text = text.replace(
        '<option integrator="implicitfast"/>',
        '<option timestep="0.004" gravity="0 0 -9.81" cone="pyramidal" '
        'solver="CG" iterations="35" ls_iterations="40" integrator="implicitfast"/>',
        1,
    )
    text = text.replace(
        "  <asset>",
        '  <visual>\n'
        '    <headlight diffuse="0.72 0.72 0.72" ambient="0.36 0.36 0.36" specular="0.1 0.1 0.1"/>\n'
        '    <global offwidth="1280" offheight="720" azimuth="124" elevation="-25"/>\n'
        '    <rgba haze="0 0 0 1"/>\n'
        '  </visual>\n\n'
        "  <asset>",
        1,
    )
    tool = """
                      <geom name="mpm3d_tool_handle" type="capsule" fromto="0 0 0.035 0 0 0.145"
                        size="0.007" rgba="0.08 0.09 0.10 1" contype="0" conaffinity="0"/>
                      <geom name="mpm3d_blade" type="box" pos="0 0 0.168"
                        size="0.082 0.122 0.015" rgba="0.015 0.018 0.022 1"
                        contype="0" conaffinity="0"/>
                      <site name="mpm3d_blade_center" pos="0 0 0.168" size="0.010"
                        rgba="1.0 0.58 0.16 1"/>
"""
    text = text.replace(
        '                      <body name="left_finger" pos="0 0 0.0584">',
        tool + '                      <body name="left_finger" pos="0 0 0.0584">',
        1,
    )
    world = """
    <light name="density_top_key" pos="0.50 0.28 1.35" dir="0 0 -1" directional="true"/>
    <light name="density_side_key" pos="0.50 -0.55 0.70" dir="0 1 -0.35" directional="true"/>
    <camera name="density_top_cam" pos="0.52 0.28 2.25" xyaxes="1 0 0 0 1 0" fovy="23"/>
    <camera name="density_side_cam" pos="0.52 -2.20 0.27" xyaxes="1 0 0 0 0 1" fovy="17"/>
    <geom name="render_floor" type="plane" size="1.30 1.30 0.04" rgba="0.02 0.023 0.025 1"
      contype="0" conaffinity="0"/>
"""
    text = text.replace("</worldbody>", world + "  </worldbody>", 1)
    for old, new in {
        'gainprm="4500" biasprm="0 -4500 -450"': 'gainprm="1700" biasprm="0 -1700 -155"',
        'gainprm="3500" biasprm="0 -3500 -350"': 'gainprm="1450" biasprm="0 -1450 -125"',
        'gainprm="2000" biasprm="0 -2000 -200"': 'gainprm="820" biasprm="0 -820 -78"',
    }.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    if model.nq >= 9:
        data.qpos[7:9] = 0.02
    mujoco.mj_forward(model, data)


def set_eef_only_visibility(model: mujoco.MjModel, include_wrist_link: bool = False) -> None:
    visible_bodies = {"hand", "left_finger", "right_finger"}
    if include_wrist_link:
        visible_bodies.add("link7")
    visible_geoms = {"mpm3d_tool_handle", "mpm3d_blade"}
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id]),
        )
        if body_name in visible_bodies or geom_name in visible_geoms:
            continue
        model.geom_rgba[geom_id, 3] = 0.0
        model.geom_group[geom_id] = 5


def disable_hidden_geom_group(renderer: mujoco.Renderer) -> None:
    if hasattr(renderer, "scene_option"):
        renderer.scene_option.geomgroup[5] = 0


def solve_site_pose_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target: np.ndarray,
    target_angle: float,
    initial_qpos: np.ndarray,
    iterations: int,
) -> np.ndarray:
    qpos = initial_qpos.copy()
    lo, hi = joint_ranges(model)
    qpos = np.clip(qpos, lo, hi)
    data.qpos[:] = qpos
    if model.nq >= 9:
        data.qpos[7:9] = 0.02
    desired_rot = desired_tool_rotation(target_angle)
    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        pos_error = np.asarray(target, dtype=np.float64) - data.site_xpos[site_id]
        rot_error = orientation_error(data.site_xmat[site_id].reshape(3, 3), desired_rot)
        if np.linalg.norm(pos_error) < 0.002 and np.linalg.norm(rot_error) < 0.025:
            break
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        orientation_weight = 0.36
        j = np.vstack([jacp[:, :7], orientation_weight * jacr[:, :7]])
        error = np.concatenate([pos_error, orientation_weight * rot_error])
        lhs = j.T @ j + 2.5e-3 * np.eye(7)
        rhs = j.T @ error
        dq = np.linalg.solve(lhs, rhs)
        data.qpos[:7] += np.clip(0.62 * dq, -0.040, 0.040)
        data.qpos[:] = np.clip(data.qpos, lo, hi)
        if model.nq >= 9:
            data.qpos[7:9] = 0.02
    mujoco.mj_forward(model, data)
    return data.qpos.copy()


def desired_tool_rotation(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    x_axis = np.asarray([c, 0.0, -s], dtype=np.float64)
    y_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def orientation_error(current_rot: np.ndarray, desired_rot: np.ndarray) -> np.ndarray:
    current = np.asarray(current_rot, dtype=np.float64).reshape(3, 3)
    desired = np.asarray(desired_rot, dtype=np.float64).reshape(3, 3)
    return 0.5 * (
        np.cross(current[:, 0], desired[:, 0])
        + np.cross(current[:, 1], desired[:, 1])
        + np.cross(current[:, 2], desired[:, 2])
    )


def joint_ranges(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(model.nq, -np.inf)
    hi = np.full(model.nq, np.inf)
    for joint_id in range(model.njnt):
        qpos_id = int(model.jnt_qposadr[joint_id])
        if model.jnt_limited[joint_id]:
            lo[qpos_id] = model.jnt_range[joint_id, 0]
            hi[qpos_id] = model.jnt_range[joint_id, 1]
    return lo, hi


def render_robot_masked(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera_id: int,
    site_id: int,
    threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    del model, site_id
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera_id)
    rgb = renderer.render().copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    mask = ((value >= threshold) & (saturation <= 130)).astype(np.uint8) * 255
    dark_tool = ((value >= 18) & (value < threshold) & (saturation <= 90)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask, dark_tool)
    mask[:2, :] = 0
    mask[-2:, :] = 0
    mask[:, :2] = 0
    mask[:, -2:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return bgr, mask.astype(np.float32) / 255.0


def project_site(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    site_id: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    uv, _depth, _valid = project_points(model, data, camera_id, data.site_xpos[site_id][None, :], width, height)
    return int(round(float(uv[0, 0]))), int(round(float(uv[0, 1])))


def composite_view(
    frame: np.ndarray,
    robot_bgr: np.ndarray,
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
    crop: tuple[int, int, int, int],
    desired: tuple[int, int],
    rendered: tuple[int, int],
    alpha: float,
) -> np.ndarray:
    x0, y0, w, h = rect
    dx = int(desired[0] - rendered[0])
    dy = int(desired[1] - rendered[1])
    shifted_rgb = np.zeros_like(robot_bgr)
    shifted_mask = np.zeros_like(mask)
    transform = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    cv2.warpAffine(robot_bgr, transform, (w, h), shifted_rgb, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.warpAffine(mask, transform, (w, h), shifted_mask, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    shifted_rgb = crop_and_resize(shifted_rgb, crop, (w, h), interpolation=cv2.INTER_LINEAR)
    shifted_mask = crop_and_resize(shifted_mask, crop, (w, h), interpolation=cv2.INTER_LINEAR)
    roi = frame[y0 : y0 + h, x0 : x0 + w]
    a = np.clip(shifted_mask[:, :, None] * alpha, 0.0, 1.0)
    roi[:] = np.clip(roi.astype(np.float32) * (1.0 - a) + shifted_rgb.astype(np.float32) * a, 0, 255).astype(
        np.uint8
    )
    return frame


def interaction_crops(
    frame_count: int,
    substeps_per_frame: int,
    mpm_dt: float,
    crop_mode: str,
    top_margin: int,
    side_margin: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if crop_mode == "full":
        return (0, 0, TOP_RECT[2], TOP_RECT[3]), (0, 0, SIDE_RECT[2], SIDE_RECT[3])

    top_points: list[tuple[int, int]] = []
    side_points: list[tuple[int, int]] = []
    for frame_id in range(frame_count):
        tool_time = (frame_id * substeps_per_frame + max(0, substeps_per_frame - 1)) * mpm_dt
        tool = blade_state(tool_time, mpm_dt)
        samples = np.vstack([tool.center[None, :], tool_corners(tool)])
        for point in samples:
            top_points.append(_map_point(point, TOP_RECT, (0, 1), (DOMAIN_X, DOMAIN_Y), local=True))
            side_points.append(_map_point(point, SIDE_RECT, (0, 2), (DOMAIN_X, DOMAIN_Z), local=True))

    top_crop = crop_from_points(top_points, TOP_RECT[2], TOP_RECT[3], top_margin)
    side_crop = crop_from_points(side_points, SIDE_RECT[2], SIDE_RECT[3], side_margin)
    return top_crop, side_crop


def crop_from_points(points: list[tuple[int, int]], width: int, height: int, margin: int) -> tuple[int, int, int, int]:
    arr = np.asarray(points, dtype=np.float32)
    x_min = float(np.clip(arr[:, 0].min() - margin, 0, width - 1))
    x_max = float(np.clip(arr[:, 0].max() + margin, 1, width))
    y_min = float(np.clip(arr[:, 1].min() - margin, 0, height - 1))
    y_max = float(np.clip(arr[:, 1].max() + margin, 1, height))
    x_min, y_min, x_max, y_max = expand_to_aspect(x_min, y_min, x_max, y_max, width, height, width / height)
    return int(round(x_min)), int(round(y_min)), int(round(x_max - x_min)), int(round(y_max - y_min))


def expand_to_aspect(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    image_w: int,
    image_h: int,
    aspect: float,
) -> tuple[float, float, float, float]:
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    if w > image_w:
        w = float(image_w)
        h = min(float(image_h), w / aspect)
    if h > image_h:
        h = float(image_h)
        w = min(float(image_w), h * aspect)
    x_min = cx - 0.5 * w
    x_max = cx + 0.5 * w
    y_min = cy - 0.5 * h
    y_max = cy + 0.5 * h
    if x_min < 0:
        x_max -= x_min
        x_min = 0.0
    if x_max > image_w:
        x_min -= x_max - image_w
        x_max = float(image_w)
    if y_min < 0:
        y_max -= y_min
        y_min = 0.0
    if y_max > image_h:
        y_min -= y_max - image_h
        y_max = float(image_h)
    return (
        float(np.clip(x_min, 0, image_w - 1)),
        float(np.clip(y_min, 0, image_h - 1)),
        float(np.clip(x_max, 1, image_w)),
        float(np.clip(y_max, 1, image_h)),
    )


def crop_and_resize(
    image: np.ndarray,
    crop: tuple[int, int, int, int],
    output_size: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    x, y, w, h = crop
    cropped = image[y : y + h, x : x + w]
    return cv2.resize(cropped, output_size, interpolation=interpolation)


def zoom_density_panel(frame: np.ndarray, rect: tuple[int, int, int, int], crop: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, w, h = rect
    roi = frame[y0 : y0 + h, x0 : x0 + w]
    frame[y0 : y0 + h, x0 : x0 + w] = crop_and_resize(roi, crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return frame


def _map_point(
    point: np.ndarray,
    rect: tuple[int, int, int, int],
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
    local: bool = False,
) -> tuple[int, int]:
    x0, y0, w, h = rect
    ox = 0 if local else x0
    oy = 0 if local else y0
    x = ox + int((float(point[axes[0]]) - limits[0][0]) / (limits[0][1] - limits[0][0]) * w)
    y = oy + h - int((float(point[axes[1]]) - limits[1][0]) / (limits[1][1] - limits[1][0]) * h)
    return x, y


def map_zoomed_point(point: np.ndarray, rect: tuple[int, int, int, int], crop: tuple[int, int, int, int], axes, limits) -> tuple[int, int]:
    full_x, full_y = _map_point(point, rect, axes, limits, local=True)
    crop_x, crop_y, crop_w, crop_h = crop
    x = rect[0] + int((full_x - crop_x) / max(1, crop_w) * rect[2])
    y = rect[1] + int((full_y - crop_y) / max(1, crop_h) * rect[3])
    return x, y


def draw_shared_contact_marker(
    frame: np.ndarray,
    center: np.ndarray,
    top_crop: tuple[int, int, int, int],
    side_crop: tuple[int, int, int, int],
) -> None:
    for rect, crop, axes, limits in [
        (TOP_RECT, top_crop, (0, 1), (DOMAIN_X, DOMAIN_Y)),
        (SIDE_RECT, side_crop, (0, 2), (DOMAIN_X, DOMAIN_Z)),
    ]:
        p = map_zoomed_point(center, rect, crop, axes, limits)
        cv2.circle(frame, p, 5, (58, 198, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 8, (15, 22, 26), 1, cv2.LINE_AA)


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
