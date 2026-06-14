"""Render a cinematic 3D interaction teaser with Blender.

Run with:
  blender --background --python scripts/render_blender_interaction_teaser.py
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
VIDEO = OUT / "videos" / "blender_interaction_teaser.mp4"
POSTER = OUT / "posters" / "blender_interaction_teaser.png"
FRAMES_DIR = ROOT / "outputs" / "blender_interaction_teaser_frames"

FRAME_START = 1
FRAME_END = 96
FPS = 24
WIDTH = 1280
HEIGHT = 720
GRID_X = 150
GRID_Y = 76
BED_X = 4.2
BED_Y = 2.2


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_mat(name: str, color: tuple[float, float, float, float], roughness: float = 0.55, metallic: float = 0.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    return mat


def add_cube(name: str, loc: tuple[float, float, float], scale: tuple[float, float, float], mat: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    return obj


def add_bevel(obj: bpy.types.Object, amount: float, segments: int = 3) -> None:
    bevel = obj.modifiers.new("small bevel", "BEVEL")
    bevel.width = amount
    bevel.segments = segments
    bevel.affect = "EDGES"
    obj.modifiers.new("soft normals", "WEIGHTED_NORMAL")


def sand_height(x: float, y: float, t: float) -> float:
    grain = 0.012 * math.sin(10.7 * x + 1.3 * y) + 0.009 * math.sin(5.3 * x - 13.0 * y)
    blade_x = -1.35 + 2.65 * t
    trough = math.exp(-(((x - blade_x + 0.22) / 0.34) ** 2 + (y / 0.55) ** 2))
    ridge = math.exp(-(((x - blade_x - 0.34) / 0.42) ** 2 + (y / 0.58) ** 2))
    wake = math.exp(-(((x - blade_x - 0.02) / 0.58) ** 2 + (y / 0.78) ** 2))
    return grain + t * (-0.16 * trough + 0.13 * ridge + 0.035 * wake)


def make_sand_mesh(mat: bpy.types.Material) -> bpy.types.Object:
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for iy in range(GRID_Y):
        y = -BED_Y / 2 + BED_Y * iy / (GRID_Y - 1)
        for ix in range(GRID_X):
            x = -BED_X / 2 + BED_X * ix / (GRID_X - 1)
            z = sand_height(x, y, 0.0)
            verts.append((x, y, z))
    for iy in range(GRID_Y - 1):
        for ix in range(GRID_X - 1):
            a = iy * GRID_X + ix
            faces.append((a, a + 1, a + GRID_X + 1, a + GRID_X))

    mesh = bpy.data.meshes.new("animated sand surface")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("animated granular surface", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)

    obj.shape_key_add(name="start")
    dug = obj.shape_key_add(name="dug")
    for idx, v in enumerate(obj.data.vertices):
        x, y, _z = v.co
        dug.data[idx].co.z = sand_height(float(x), float(y), 1.0)
    dug.value = 0.0
    dug.keyframe_insert("value", frame=FRAME_START)
    dug.value = 1.0
    dug.keyframe_insert("value", frame=FRAME_END)
    obj.modifiers.new("surface smoothing", "WEIGHTED_NORMAL")
    return obj


def animate_blade(metal_mat: bpy.types.Material) -> bpy.types.Object:
    blade = add_cube("polished excavation blade", (-1.35, 0.0, 0.25), (0.18, 1.28, 0.56), metal_mat)
    blade.rotation_euler[1] = math.radians(-7.0)
    add_bevel(blade, 0.035, 8)

    arm = add_cube("tool shank", (-1.35, 0.0, 0.78), (0.18, 0.18, 0.92), metal_mat)
    arm.rotation_euler[1] = math.radians(-7.0)
    add_bevel(arm, 0.028, 6)

    for obj in [blade, arm]:
        obj.location.x = -1.35
        obj.keyframe_insert("location", frame=FRAME_START)
        obj.location.x = 1.30
        obj.keyframe_insert("location", frame=FRAME_END)
        obj.keyframe_insert("rotation_euler", frame=FRAME_START)
        obj.keyframe_insert("rotation_euler", frame=FRAME_END)
    return blade


def add_labels() -> None:
    font_mat = make_mat("label blue", (0.08, 0.20, 0.82, 1.0), 0.45)
    green_mat = make_mat("label green", (0.05, 0.48, 0.24, 1.0), 0.45)
    for text, x, y, mat in [
        ("target cut", -0.45, -0.82, font_mat),
        ("deposit", 0.76, -0.82, green_mat),
    ]:
        bpy.ops.object.text_add(location=(x, y, 0.18), rotation=(math.radians(74), 0, 0))
        obj = bpy.context.object
        obj.name = text
        obj.data.body = text
        obj.data.align_x = "CENTER"
        obj.data.size = 0.12
        obj.data.materials.append(mat)

    blue = make_mat("transparent cut blue", (0.05, 0.18, 1.0, 0.34), 0.8)
    green = make_mat("transparent deposit green", (0.02, 0.55, 0.22, 0.34), 0.8)
    for mat in [blue, green]:
        mat.blend_method = "BLEND"
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Alpha"].default_value = 0.28

    cut = add_cube("target cut translucent guide", (-0.45, 0.0, 0.035), (0.98, 0.72, 0.012), blue)
    dep = add_cube("deposit translucent guide", (0.68, 0.0, 0.038), (0.88, 0.72, 0.012), green)
    for obj in [cut, dep]:
        obj.display_type = "TEXTURED"


def setup_scene() -> None:
    clear_scene()
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    POSTER.parent.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.frame_start = FRAME_START
    scene.frame_end = FRAME_END
    scene.frame_set(FRAME_START)
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.fps = FPS
    scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 64
        if hasattr(scene.eevee, "use_raytracing"):
            scene.eevee.use_raytracing = True
    scene.world = bpy.data.worlds.new("soft studio world")
    scene.world.color = (0.78, 0.84, 0.88)

    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    sand_mat = make_mat("moist granular sand", (0.68, 0.47, 0.23, 1.0), 0.92, 0.0)
    metal_mat = make_mat("brushed white metal", (0.86, 0.89, 0.88, 1.0), 0.36, 0.18)
    tray_mat = make_mat("matte black tray", (0.015, 0.020, 0.018, 1.0), 0.64, 0.0)
    table_mat = make_mat("warm lab table", (0.62, 0.62, 0.57, 1.0), 0.78, 0.0)

    make_sand_mesh(sand_mat)
    animate_blade(metal_mat)
    add_labels()

    add_cube("table", (0, 0, -0.14), (5.7, 3.45, 0.16), table_mat)
    for name, loc, scale in [
        ("tray front", (0, -1.28, 0.08), (4.65, 0.18, 0.36)),
        ("tray back", (0, 1.28, 0.08), (4.65, 0.18, 0.36)),
        ("tray left", (-2.28, 0, 0.08), (0.18, 2.72, 0.36)),
        ("tray right", (2.28, 0, 0.08), (0.18, 2.72, 0.36)),
    ]:
        obj = add_cube(name, loc, scale, tray_mat)
        add_bevel(obj, 0.025, 4)

    bpy.ops.object.light_add(type="AREA", location=(-2.4, -2.2, 4.2))
    key = bpy.context.object
    key.name = "large softbox key"
    key.data.energy = 560
    key.data.size = 4.4

    bpy.ops.object.light_add(type="AREA", location=(2.5, 1.7, 2.7))
    fill = bpy.context.object
    fill.name = "small rim fill"
    fill.data.energy = 110
    fill.data.size = 2.0

    bpy.ops.object.camera_add(location=(3.35, -3.15, 2.15), rotation=(math.radians(61), 0, math.radians(42)))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.data.lens = 33
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 4.1
    cam.data.dof.aperture_fstop = 7.5

    empty = bpy.data.objects.new("camera target", None)
    bpy.context.collection.objects.link(empty)
    empty.location = (0.15, 0.0, 0.02)
    constraint = cam.constraints.new(type="TRACK_TO")
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    constraint.target = empty

    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(FRAMES_DIR / "frame_")
    scene.render.image_settings.file_format = "PNG"


def render() -> None:
    setup_scene()
    bpy.ops.render.render(animation=True)
    bpy.context.scene.frame_set((FRAME_START + FRAME_END) // 2)
    bpy.context.scene.render.filepath = str(POSTER)
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print(f"frames={FRAMES_DIR}")
    print(f"poster={POSTER}")


if __name__ == "__main__":
    render()
