from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

DEFAULT_PATTERNS = [
    "README.md",
    "install.sh",
    "pyproject.toml",
    "configs/*.json",
    "configs/**/*.json",
    "constraints/*.txt",
    "docs/*.md",
    "outputs/mujoco_newton_mpm_bridge/*.mp4",
    "outputs/mujoco_newton_mpm_bridge/mujoco_franka_newton_mpm_bridge_preview.png",
    "outputs/mujoco_newton_mpm_bridge/mujoco_franka_newton_mpm_bridge_sheet.png",
    "outputs/mujoco_newton_mpm_bridge/newton_mpm_bridge_log.npz",
    "outputs/3d_mpm_density_render/*.mp4",
    "outputs/3d_mpm_density_render/*.png",
    "outputs/3d_mpm_blade/*.mp4",
    "outputs/3d_mpm_blade/*.png",
    "outputs/3d_mpm_blade/*.csv",
    "outputs/3d_mpm_blade/resolved_config.json",
]

EXPERIMENT_PATTERNS = [
    "outputs/experiments/*/config/*.json",
    "outputs/experiments/*/dataset_metrics/*.json",
    "outputs/experiments/*/training_metrics/*.json",
    "outputs/experiments/*/inference_results/*.json",
    "outputs/experiments/*/inference_results/*.csv",
    "outputs/experiments/*/inference_results/*.png",
    "outputs/experiments/*/video_set/**/*.mp4",
    "outputs/experiments/*/video_set/**/*.png",
    "outputs/sweeps/*/samples.csv",
    "outputs/sweeps/*/sweep_manifest.json",
    "outputs/sweeps/*/configs/*.json",
    "outputs/sweeps/*/dataset/*.json",
    "outputs/sweeps/*/training_metrics/*.json",
    "outputs/sweeps/*/inference_results/*.json",
    "outputs/sweeps/*/inference_results/*.csv",
    "outputs/sweeps/*/inference_results/*.png",
]

HEAVY_PATTERNS = [
    "outputs/newton_mpm_spike/*.usd",
    "outputs/newton_mpm_twoway/*.usd",
    "outputs/newton_mpm_granular/*.usd",
    "outputs/newton_mpm_grain/*.usd",
    "outputs/ply_sequence_test/*.ply",
]

REPRO_COMMANDS = [
    "python scripts/run_3d_density_render_demo.py",
    "python scripts/run_3d_blade_demo.py --config configs/sand3d_blade_demo.json",
    "python scripts/run_experiment_sequence.py --quick --skip-bridge",
    "python scripts/run_property_sweep.py --quick --skip-bridge --count 2",
    (
        "python scripts/run_mujoco_newton_mpm_bridge.py --voxel-size 0.032 "
        "--particles-per-cell 3.0 --sand-render-mode heightfield "
        "--render-blur 2.4 --alpha-cutoff 0.060 --alpha-gain 0.48"
    ),
]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            if path.is_file() and path not in seen:
                files.append(path)
                seen.add(path)
    return sorted(files, key=lambda p: p.as_posix())


def format_mb(size: int) -> float:
    return round(size / (1024 * 1024), 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--include-experiments", action="store_true")
    parser.add_argument("--include-heavy-usd", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    patterns = list(DEFAULT_PATTERNS)
    if args.include_experiments:
        patterns.extend(EXPERIMENT_PATTERNS)
    if args.include_heavy_usd:
        patterns.extend(HEAVY_PATTERNS)

    files = collect(patterns)
    if not files:
        raise SystemExit("No generated artifacts found. Run a demo first.")

    sha = git_sha()
    out = args.out or DIST / f"granular-robot-demo-artifacts-{sha}.zip"
    manifest = {
        "name": "granular-robot-demo-artifacts",
        "git_sha": sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "include_experiments": bool(args.include_experiments),
        "include_heavy_usd": bool(args.include_heavy_usd),
        "reproduce_commands": REPRO_COMMANDS,
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "mb": format_mb(path.stat().st_size),
                "sha256": sha256(path),
            }
            for path in files
        ],
    }

    total = sum(item["bytes"] for item in manifest["files"])
    print(f"files={len(files)} total={format_mb(total)} MB")
    for item in manifest["files"]:
        print(f"{item['mb']:>9} MB  {item['path']}")

    if args.dry_run:
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("artifact_manifest.json", json.dumps(manifest, indent=2))
        for path in files:
            zf.write(path, path.relative_to(ROOT).as_posix())

    print(f"wrote={out}")


if __name__ == "__main__":
    main()
