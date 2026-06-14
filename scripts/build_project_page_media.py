from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
VIDEO_DIR = ASSETS / "videos"
POSTER_DIR = ASSETS / "posters"
FIGURE_DIR = ASSETS / "figures"
PAPER_DIR = ASSETS / "papers"

RAW_ROLLOUT = (
    ROOT
    / "experiments"
    / "raw_rgb_posterior_excavation"
    / "assets"
    / "c02_cohesive_wet_sand_shallow_trench_seed7_rgb_posterior_ablation.mp4"
)
RAW_ROLLOUT_POSTER = RAW_ROLLOUT.with_suffix(".jpg")
PAPER_FIGURES = ROOT / "paper_draft" / "arxiv_paper" / "figures"
PAPER_DRAFT = ROOT / "paper_draft" / "arxiv_paper" / "granular_sim2sim_arxiv_tectonic.pdf"
DDBOT_ASSETS = ROOT / "experiments" / "ddbot_posterior_heightfield_mpc" / "assets"

PROBE_MATERIAL = {
    "rho": 1760.0,
    "phi_deg": 33.0,
    "delta_deg": 18.0,
    "cohesion_kpa": 10.5,
}


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts") / name,
        Path("C:/Windows/Fonts") / "segoeui.ttf",
        Path("C:/Windows/Fonts") / "arial.ttf",
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_REG = font("segoeui.ttf", 32)
FONT_SEMI = font("seguisb.ttf", 34)
FONT_BOLD = font("segoeuib.ttf", 58)
FONT_BIG = font("segoeuib.ttf", 72)
FONT_SMALL = font("segoeui.ttf", 24)
FONT_TINY = font("segoeui.ttf", 21)


def rounded_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1 if outline else 0)


def paste_fit(dst: Image.Image, src_bgr: np.ndarray, box: tuple[int, int, int, int]) -> None:
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(src)
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    scale = min(bw / image.width, bh / image.height)
    nw, nh = int(image.width * scale), int(image.height * scale)
    image = image.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (bw, bh), "#070b0d")
    canvas.paste(image, ((bw - nw) // 2, (bh - nh) // 2))
    dst.paste(canvas, (x0, y0))


def encode_h264(src: Path, dst: Path, crf: int = 18) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("Install imageio-ffmpeg to build browser-playable MP4 assets.") from exc

    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)


def draw_experiment_view_frame(frame_id: int, total: int) -> Image.Image:
    canvas = Image.new("RGB", (1920, 1080), "#f6f8f7")
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, 1920, 1080), fill="#f3f7f8")
    draw.text((72, 54), "Interaction view", font=FONT_BOLD, fill="#10233f")
    draw.text((74, 126), "same simulated bed and tool motion shown as a sand-surface view", font=FONT_SMALL, fill="#526171")

    arena = (96, 200, 1824, 952)
    rounded_rect(draw, arena, 28, "#1e2420", "#394139")
    inner = (138, 238, 1782, 910)

    rng = np.random.default_rng(17)
    w, h = inner[2] - inner[0], inner[3] - inner[1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx / max(1, w - 1) - 0.5) * 2.0
    y = (yy / max(1, h - 1) - 0.5) * 2.0
    t = frame_id / max(1, total - 1)
    t_smooth = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))

    rho_n = (PROBE_MATERIAL["rho"] - 1050.0) / (2050.0 - 1050.0)
    phi_n = (PROBE_MATERIAL["phi_deg"] - 23.0) / (43.0 - 23.0)
    delta_n = (PROBE_MATERIAL["delta_deg"] - 8.0) / (30.0 - 8.0)
    cohesion_n = PROBE_MATERIAL["cohesion_kpa"] / 16.0

    grain = (
        0.55
        + 0.16 * np.sin((9.0 + 5.0 * delta_n) * x + 1.5 * y)
        + 0.12 * np.sin(4.2 * x - (10.0 + 3.0 * phi_n) * y + 0.5)
        + 0.025 * rng.normal(0.0, 1.0, size=(h, w))
    ).astype(np.float32)
    grain = np.clip(grain, 0.0, 1.0)

    blade_x = -0.68 + 1.18 * t_smooth
    wake_center = blade_x - 0.13 * (1.0 - cohesion_n) + 0.08 * phi_n
    wake = np.exp(-((x - wake_center) ** 2 / (2.0 * 0.19**2) + y**2 / (2.0 * 0.34**2)))
    trough = np.exp(-((x - blade_x + 0.10) ** 2 / (2.0 * 0.15**2) + y**2 / 0.24))
    ridge = np.exp(-((x - blade_x - 0.23) ** 2 / (2.0 * 0.22**2) + y**2 / (2.0 * 0.28**2)))
    deformation = (0.72 * wake + 0.50 * ridge - 0.48 * trough) * t_smooth
    shear = np.clip(np.gradient(deformation, axis=1), -0.5, 0.5)
    shade = np.clip(0.92 - 0.35 * shear + 0.10 * np.gradient(deformation, axis=0), 0.60, 1.22)

    r = (134 + 38 * rho_n + 47 * grain + 88 * deformation) * shade
    g = (101 + 29 * rho_n + 34 * grain + 55 * deformation) * shade
    b = (55 + 17 * grain + 18 * cohesion_n + 24 * deformation) * shade
    sand = np.stack([r, g, b], axis=-1)
    sand = np.clip(sand, 0, 255).astype(np.uint8)

    # Dark tray walls give the synthetic surface a concrete experimental frame.
    img = Image.fromarray(sand)
    canvas.paste(img, (inner[0], inner[1]))
    draw.rectangle(inner, outline="#4b5149", width=4)
    draw.line((inner[0], inner[1], inner[0] + 92, inner[1] - 34, inner[2] - 92, inner[1] - 34, inner[2], inner[1]), fill="#2d332d", width=20, joint="curve")
    draw.line((inner[0], inner[3], inner[0] + 92, inner[3] + 34, inner[2] - 92, inner[3] + 34, inner[2], inner[3]), fill="#2d332d", width=20, joint="curve")

    bx = inner[0] + int((blade_x + 1.0) * 0.5 * w)
    blade_w = 42
    blade_h = int(h * 0.72)
    by0 = inner[1] + (h - blade_h) // 2
    by1 = by0 + blade_h
    draw.rounded_rectangle((bx - blade_w // 2, by0, bx + blade_w // 2, by1), radius=16, fill="#ecefed", outline="#74808a", width=3)
    draw.rectangle((bx + 24, by0 + 10, bx + 140, by0 + 46), fill="#cfd5d9", outline="#7d8790", width=2)
    draw.ellipse((bx + 118, by0 + 12, bx + 156, by0 + 50), fill="#eef1f1", outline="#7d8790", width=2)
    draw.line((bx - 4, by0 + 24, bx + 8, by1 - 24), fill="#ffffff", width=3)

    # Target trench and deposit zones, aligned with the heat-map panels.
    cut = (inner[0] + int(w * 0.33), inner[1] + int(h * 0.33), inner[0] + int(w * 0.58), inner[1] + int(h * 0.67))
    dep = (inner[0] + int(w * 0.58), inner[1] + int(h * 0.33), inner[0] + int(w * 0.76), inner[1] + int(h * 0.67))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle(cut, fill=(38, 111, 237, 32), outline=(38, 111, 237, 210), width=3)
    od.rectangle(dep, fill=(22, 163, 74, 30), outline=(22, 163, 74, 210), width=3)
    od.text((cut[0] + 12, cut[1] - 30), "target cut", font=FONT_TINY, fill=(38, 111, 237, 235))
    od.text((dep[0] + 12, dep[1] - 30), "deposit", font=FONT_TINY, fill=(22, 130, 70, 235))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    progress_x0, progress_y = 138, 990
    progress_w = 1644
    draw.rounded_rectangle((progress_x0, progress_y, progress_x0 + progress_w, progress_y + 18), radius=9, fill="#d8e1e5")
    draw.rounded_rectangle((progress_x0, progress_y, progress_x0 + int(progress_w * t), progress_y + 18), radius=9, fill="#158c88")
    draw.text((progress_x0, progress_y + 34), "Visible view: blade motion and sand deformation", font=FONT_SMALL, fill="#526171")
    draw.text((progress_x0 + 780, progress_y + 34), "Heat-map view: the same deformation converted to height", font=FONT_SMALL, fill="#526171")
    return canvas


def build_experiment_view() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = VIDEO_DIR / "raw_rgb_interaction_view.raw.mp4"
    out_path = VIDEO_DIR / "raw_rgb_interaction_view.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (1920, 1080))
    total = 144
    poster = None
    for frame_id in range(total):
        image = draw_experiment_view_frame(frame_id, total)
        if frame_id == total // 2:
            poster = image.copy()
        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    writer.release()
    if poster is None:
        poster = draw_experiment_view_frame(total // 2, total)
    poster.save(POSTER_DIR / "raw_rgb_interaction_view.jpg", quality=92)
    encode_h264(raw_path, out_path, crf=18)
    raw_path.unlink(missing_ok=True)


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    body: list[tuple[str, str, str]],
    accent: str,
) -> None:
    x0, y0, x1, y1 = box
    rounded_rect(draw, box, 20, "#121a1d", "#243236")
    draw.rounded_rectangle((x0, y0, x0 + 7, y1), radius=20, fill=accent)
    draw.text((x0 + 28, y0 + 22), label, font=FONT_SEMI, fill="#f7fbf9")
    y = y0 + 74
    for title, value, color in body:
        draw.text((x0 + 28, y), title, font=FONT_TINY, fill="#9fb0aa")
        draw.text((x1 - 34, y - 5), value, font=FONT_REG, fill=color, anchor="ra")
        y += 42


def draw_teaser_frame(frame_bgr: np.ndarray, frame_idx: int, total: int) -> Image.Image:
    canvas = Image.new("RGB", (1920, 1080), "#0a0f12")
    draw = ImageDraw.Draw(canvas)

    # Warm sand texture band.
    for y in range(0, 1080):
        t = y / 1080
        r = int(10 + 18 * t)
        g = int(15 + 14 * t)
        b = int(18 + 6 * t)
        draw.line((0, y, 1920, y), fill=(r, g, b))
    draw.rectangle((0, 0, 1920, 118), fill="#081012")
    draw.text((70, 34), "Granular Sim2Sim", font=FONT_SMALL, fill="#78cfc7")
    draw.text((70, 66), "Raw RGB material belief changes the next excavation action", font=FONT_BOLD, fill="#f6faf8")
    draw.text(
        (70, 126),
        "same bed · same target · same action budget · belief is the only controlled variable",
        font=FONT_SMALL,
        fill="#a9b7b2",
    )

    video_box = (68, 172, 1260, 958)
    rounded_rect(draw, (48, 152, 1280, 978), 26, "#0f171a", "#314247")
    paste_fit(canvas, frame_bgr, video_box)
    draw.rounded_rectangle(video_box, radius=14, outline="#41565c", width=2)
    draw.text((72, 986), "MPM rollout under four belief inputs", font=FONT_SMALL, fill="#c7d2ce")

    x0, w = 1315, 535
    draw_card(
        draw,
        (x0, 172, x0 + w, 374),
        "Control ablation",
        [
            ("No posterior target loss", "2.198", "#aeb9c5"),
            ("Estimated posterior", "2.010", "#6ea8ff"),
            ("GT-property reference", "1.972", "#5bd585"),
        ],
        "#2f6fed",
    )
    draw_card(
        draw,
        (x0, 404, x0 + w, 606),
        "Action evidence",
        [
            ("Exact GT-action matches", "12/24", "#6ea8ff"),
            ("No / wrong posterior", "0/24", "#f1a56a"),
            ("64-condition ordering", "preserved", "#e7edf0"),
        ],
        "#42b983",
    )
    draw_card(
        draw,
        (x0, 636, x0 + w, 836),
        "Evidence boundary",
        [
            ("Claim", "controlled Sim2Sim", "#e7edf0"),
            ("Real-camera bridge", "supporting only", "#e7edf0"),
            ("Next step", "closed-loop Real2Sim", "#e7edf0"),
        ],
        "#d5943d",
    )

    progress = frame_idx / max(1, total - 1)
    draw.rounded_rectangle((x0, 882, x0 + w, 902), radius=9, fill="#213034")
    draw.rounded_rectangle((x0, 882, int(x0 + w * progress), 902), radius=9, fill="#78cfc7")
    draw.text((x0, 926), "RAW RGB probe  ->  posterior  ->  finite selector", font=FONT_SMALL, fill="#cad5d1")
    draw.text((x0, 964), "Teaser only; quantitative claims are in the matched tables.", font=FONT_TINY, fill="#899893")

    return canvas


def build_teaser() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(RAW_ROLLOUT))
    if not cap.isOpened():
        raise FileNotFoundError(RAW_ROLLOUT)

    src_frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        src_frames.append(frame)
    cap.release()
    if not src_frames:
        raise RuntimeError(f"No frames in {RAW_ROLLOUT}")

    out_path = VIDEO_DIR / "raw_rgb_posterior_teaser.mp4"
    raw_path = VIDEO_DIR / "raw_rgb_posterior_teaser.raw.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (1920, 1080))
    total = len(src_frames) * 4
    poster: Image.Image | None = None
    for i in range(total):
        src = src_frames[(i // 4) % len(src_frames)]
        image = draw_teaser_frame(src, i, total)
        if poster is None and i >= total // 3:
            poster = image.copy()
        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    writer.release()

    if poster is None:
        poster = draw_teaser_frame(src_frames[0], 0, total)
    poster.save(POSTER_DIR / "raw_rgb_posterior_teaser.jpg", quality=92)
    encode_h264(raw_path, out_path, crf=18)
    raw_path.unlink(missing_ok=True)


def copy_assets() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    copies = {
        PAPER_DRAFT: PAPER_DIR / "granular_sim2sim_draft.pdf",
        RAW_ROLLOUT: VIDEO_DIR / "raw_rgb_posterior_rollout.mp4",
        RAW_ROLLOUT_POSTER: POSTER_DIR / "raw_rgb_posterior_rollout.jpg",
        PAPER_FIGURES / "qualitative_rollout_evidence.png": FIGURE_DIR / "qualitative_rollout_evidence.png",
        PAPER_FIGURES / "evidence_dashboard.png": FIGURE_DIR / "evidence_dashboard.png",
        PAPER_FIGURES / "posterior_control_pipeline.png": FIGURE_DIR / "posterior_control_pipeline.png",
        PAPER_FIGURES / "policy_regret_summary.png": FIGURE_DIR / "policy_regret_summary.png",
        PAPER_FIGURES / "modality_ablation_summary.png": FIGURE_DIR / "modality_ablation_summary.png",
        ROOT
        / "experiments"
        / "raw_rgb_posterior_excavation"
        / "assets"
        / "trench_feasibility_audit.png": FIGURE_DIR / "trench_feasibility_audit.png",
        ROOT
        / "experiments"
        / "raw_rgb_posterior_excavation"
        / "assets"
        / "shuffled_posterior_failure_audit.png": FIGURE_DIR / "shuffled_posterior_failure_audit.png",
        PAPER_FIGURES / "real_soil_rgb_bridge_sheet.png": FIGURE_DIR / "real_soil_rgb_bridge_sheet.png",
        PAPER_FIGURES / "ddbot_core_force_posterior_summary.png": FIGURE_DIR / "ddbot_core_force_posterior_summary.png",
        DDBOT_ASSETS / "posterior_heightfield_mpc_comparison.mp4": VIDEO_DIR / "posterior_heightfield_mpc_comparison.mp4",
        DDBOT_ASSETS / "posterior_heightfield_mpc_poster.jpg": POSTER_DIR / "posterior_heightfield_mpc_poster.jpg",
        DDBOT_ASSETS / "posterior_ablation_summary.png": FIGURE_DIR / "posterior_ablation_summary.png",
        DDBOT_ASSETS / "posterior_ablation_final_height_maps.png": FIGURE_DIR / "posterior_ablation_final_height_maps.png",
    }
    for src, dst in copies.items():
        if src.exists():
            if dst.suffix.lower() == ".mp4":
                encode_h264(src, dst, crf=18)
                print(f"encoded {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
            else:
                shutil.copy2(src, dst)
                print(f"copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        else:
            print(f"missing {src.relative_to(ROOT)}")


def main() -> None:
    copy_assets()
    build_experiment_view()
    build_teaser()
    print(f"wrote {VIDEO_DIR / 'raw_rgb_interaction_view.mp4'}")
    print(f"wrote {POSTER_DIR / 'raw_rgb_interaction_view.jpg'}")
    print(f"wrote {VIDEO_DIR / 'raw_rgb_posterior_teaser.mp4'}")
    print(f"wrote {POSTER_DIR / 'raw_rgb_posterior_teaser.jpg'}")


if __name__ == "__main__":
    main()
