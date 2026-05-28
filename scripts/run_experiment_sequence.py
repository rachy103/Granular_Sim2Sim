from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm.metrics import (  # noqa: E402
    baseline_force_metrics,
    blade_log_metrics,
    bridge_log_metrics,
    force_series_from_blade,
    force_series_from_bridge,
    video_metrics,
    write_json,
    write_prediction_csv,
)
from granular_mpm.probing_dataset import (  # noqa: E402
    build_probing_dataset,
    load_blade_wrench_csv,
    load_bridge_npz,
    probing_dataset_metrics,
    write_dataset_npz,
)
from granular_mpm.workspace import scan_workspace  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "reference_heightfield_intrusion.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sequence-name", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-bridge", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("Sequence name is empty after sanitization")
    return cleaned


def rel_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def make_layout(sequence_name: str, output_root: str) -> dict[str, Path]:
    root = ROOT / output_root / sequence_name
    return {
        "root": root,
        "config": root / "config",
        "videos": root / "video_set",
        "dataset_metrics": root / "dataset_metrics",
        "training_metrics": root / "training_metrics",
        "inference_results": root / "inference_results",
        "logs": root / "logs",
        "runs": root / "runs",
    }


def ensure_layout(layout: dict[str, Path]) -> None:
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def copy_matching(src_dir: Path, dst_dir: Path, patterns: list[str]) -> list[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for src in src_dir.glob(pattern):
            if not src.is_file() or src in seen:
                continue
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)
            seen.add(src)
    return copied


def merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def stage_density(config: dict[str, Any], layout: dict[str, Path], quick: bool, metrics_only: bool) -> dict[str, Any]:
    stage = config["stages"].get("density_render", {})
    if not stage.get("enabled", True):
        return {"enabled": False}
    out_dir = layout["runs"] / "density_render"
    if not metrics_only:
        frames = int(stage.get("frames", 60))
        substeps = int(stage.get("substeps", 34))
        if quick:
            frames = int(stage.get("quick_frames", 6))
            substeps = int(stage.get("quick_substeps", 4))
        cmd = [
            sys.executable,
            "scripts/run_3d_density_render_demo.py",
            "--out",
            out_dir.as_posix(),
            "--frames",
            str(frames),
            "--substeps",
            str(substeps),
            "--device",
            str(stage.get("device", "cpu" if quick else "cuda:0")),
        ]
        run_cmd(cmd, layout["logs"] / "density_render.log")
    copied = copy_matching(out_dir, layout["videos"] / "density_render", ["*.mp4", "*.png"])
    return {"enabled": True, "run_dir": out_dir, "videos": copied}


def stage_blade(config: dict[str, Any], layout: dict[str, Path], quick: bool, metrics_only: bool) -> dict[str, Any]:
    stage = config["stages"].get("blade_demo", {})
    if not stage.get("enabled", True):
        return {"enabled": False}
    src_config = ROOT / stage.get("config", "configs/sand3d_blade_demo.json")
    blade_config = load_json(src_config)
    blade_config = merge_dict(blade_config, stage.get("overrides", {}))
    out_dir = layout["runs"] / "blade_demo"
    blade_config["output_dir"] = rel_to_root(out_dir)
    if quick:
        blade_config["frames"] = int(stage.get("quick_frames", 6))
        blade_config["substeps_per_frame"] = int(stage.get("quick_substeps_per_frame", 4))
        blade_config["device"] = str(stage.get("quick_device", "cpu"))
    resolved_config = layout["config"] / "blade_demo_config.json"
    dump_json(resolved_config, blade_config)
    if not metrics_only:
        cmd = [sys.executable, "scripts/run_3d_blade_demo.py", "--config", resolved_config.as_posix()]
        run_cmd(cmd, layout["logs"] / "blade_demo.log")
    videos = copy_matching(out_dir, layout["videos"] / "blade_demo", ["*.mp4", "*.png"])
    logs = copy_matching(out_dir, layout["logs"] / "blade_demo", ["*.csv", "*.npz", "resolved_config.json"])
    return {"enabled": True, "run_dir": out_dir, "videos": videos, "logs": logs}


def stage_bridge(
    config: dict[str, Any],
    layout: dict[str, Path],
    quick: bool,
    skip_bridge: bool,
    metrics_only: bool,
) -> dict[str, Any]:
    stage = config["stages"].get("newton_bridge", {})
    if skip_bridge or not stage.get("enabled", True):
        return {"enabled": False, "skipped": bool(skip_bridge)}
    src_config = ROOT / stage.get("config", "configs/newton_bridge_heightfield.json")
    bridge_config = load_json(src_config)
    bridge_config = merge_dict(bridge_config, stage.get("overrides", {}))
    out_dir = layout["runs"] / "newton_bridge"
    bridge_config["output_dir"] = rel_to_root(out_dir)
    if quick:
        bridge_config["frames"] = int(stage.get("quick_frames", 4))
        bridge_config["steps_per_frame"] = int(stage.get("quick_steps_per_frame", 1))
        bridge_config["voxel_size"] = float(stage.get("quick_voxel_size", 0.060))
        bridge_config["particles_per_cell"] = float(stage.get("quick_particles_per_cell", 1.2))
    resolved_config = layout["config"] / "newton_bridge_config.json"
    dump_json(resolved_config, bridge_config)
    if not metrics_only:
        cmd = [sys.executable, "scripts/run_mujoco_newton_mpm_bridge.py", "--config", resolved_config.as_posix()]
        run_cmd(cmd, layout["logs"] / "newton_bridge.log")
    videos = copy_matching(out_dir, layout["videos"] / "newton_bridge", ["*.mp4", "*.png"])
    logs = copy_matching(out_dir, layout["logs"] / "newton_bridge", ["*.npz", "*.xml"])
    return {"enabled": True, "run_dir": out_dir, "videos": videos, "logs": logs}


def stage_probing_dataset(config: dict[str, Any], layout: dict[str, Path], quick: bool) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    if not dataset_cfg.get("enabled", True):
        return {"enabled": False}

    sources = []
    blade_csv = layout["runs"] / "blade_demo" / "wrench_log.csv"
    bridge_log = layout["runs"] / "newton_bridge" / "newton_mpm_bridge_log.npz"
    if blade_csv.exists():
        sources.append(load_blade_wrench_csv(blade_csv, name="blade_demo"))
    if bridge_log.exists():
        sources.append(
            load_bridge_npz(
                bridge_log,
                name="newton_bridge",
                frame_rate_hz=float(dataset_cfg.get("bridge_frame_rate_hz", 30.0)),
            )
        )

    targets = dict(dataset_cfg.get("targets", {"phi_deg": 34.0, "cohesion_kpa": 0.0}))
    window_length = int(dataset_cfg.get("window_length", 32))
    stride = int(dataset_cfg.get("stride", 8))
    if quick:
        window_length = int(dataset_cfg.get("quick_window_length", min(4, window_length)))
        stride = int(dataset_cfg.get("quick_stride", 1))

    dataset = build_probing_dataset(
        sources=sources,
        targets=targets,
        sample_rate_hz=float(dataset_cfg.get("sample_rate_hz", 50.0)),
        window_length=window_length,
        stride=stride,
        normalization=str(dataset_cfg.get("normalization", "zscore")),
        train_fraction=float(dataset_cfg.get("train_fraction", config.get("metrics", {}).get("train_fraction", 0.7))),
        validation_fraction=float(dataset_cfg.get("validation_fraction", 0.15)),
        seed=int(dataset_cfg.get("seed", config.get("learning", {}).get("seed", 7))),
    )
    dataset_dir = layout["runs"] / "probing_dataset"
    dataset_path = dataset_dir / "probing_windows.npz"
    write_dataset_npz(dataset_path, dataset)
    tensor_metrics = probing_dataset_metrics(dataset)
    dump_json(layout["dataset_metrics"] / "probing_tensor_metrics.json", tensor_metrics)
    dump_json(
        layout["dataset_metrics"] / "normalization_stats.json",
        dataset["metadata"].get("normalization_stats", {"method": dataset_cfg.get("normalization", "zscore")}),
    )
    return {
        "enabled": True,
        "dataset_path": dataset_path,
        "sample_count": int(dataset["x"].shape[0]),
        "metrics_path": layout["dataset_metrics"] / "probing_tensor_metrics.json",
        "status": tensor_metrics["status"],
    }


def stage_learning(
    config: dict[str, Any],
    layout: dict[str, Path],
    dataset_info: dict[str, Any],
    resolved_config: Path,
    quick: bool,
    skip_training: bool,
    metrics_only: bool,
) -> dict[str, Any]:
    learning_cfg = config.get("learning", {})
    if skip_training or metrics_only or not learning_cfg.get("enabled", True):
        return {"enabled": False, "skipped": bool(skip_training or metrics_only)}
    dataset_path = dataset_info.get("dataset_path")
    if not dataset_path or int(dataset_info.get("sample_count", 0)) == 0:
        return {"enabled": False, "skipped": True, "reason": "empty_dataset"}

    cmd = [
        sys.executable,
        "scripts/train_granular_inference.py",
        "--dataset",
        Path(dataset_path).as_posix(),
        "--training-dir",
        layout["training_metrics"].as_posix(),
        "--inference-dir",
        layout["inference_results"].as_posix(),
        "--config",
        resolved_config.as_posix(),
        "--sequence-name",
        config["sequence_name"],
    ]
    if quick:
        cmd.append("--quick")
    run_cmd(cmd, layout["logs"] / "learning.log")
    return {
        "enabled": True,
        "training_metrics": layout["training_metrics"] / "mdn_training_metrics.json",
        "representation_metrics": layout["training_metrics"] / "representation_metrics.json",
        "model": layout["training_metrics"] / "temporal_mdn.pt",
        "inference_metrics": layout["inference_results"] / "learning_inference_metrics.json",
        "predictions": layout["inference_results"] / "mdn_predictions.csv",
    }


def compute_metrics(config: dict[str, Any], layout: dict[str, Path], stages: dict[str, Any]) -> dict[str, Any]:
    video_paths = sorted(layout["videos"].glob("*/*.mp4"))
    video_payload = {"videos": [video_metrics(path) for path in video_paths]}
    dump_json(layout["dataset_metrics"] / "video_metrics.json", video_payload)

    dataset_sources: dict[str, Any] = {}
    force_series = np_empty()
    bridge_log = layout["runs"] / "newton_bridge" / "newton_mpm_bridge_log.npz"
    blade_csv = layout["runs"] / "blade_demo" / "wrench_log.csv"
    if bridge_log.exists():
        dataset_sources["newton_bridge"] = bridge_log_metrics(bridge_log)
        force_series = force_series_from_bridge(bridge_log)
    if blade_csv.exists():
        dataset_sources["blade_demo"] = blade_log_metrics(blade_csv)
        if force_series.size == 0:
            force_series = force_series_from_blade(blade_csv)

    dataset_summary = {
        "sequence_name": config["sequence_name"],
        "stage_status": compact_stage_status(stages),
        "source_count": len(dataset_sources),
        "video_count": len(video_paths),
        "force_sample_count": int(force_series.size),
        "sources": dataset_sources,
    }
    dump_json(layout["dataset_metrics"] / "dataset_summary.json", dataset_summary)

    train_fraction = float(config.get("metrics", {}).get("train_fraction", 0.7))
    train_payload, prediction_rows = baseline_force_metrics(force_series, train_fraction=train_fraction)
    dump_json(layout["training_metrics"] / "baseline_force_model.json", train_payload)
    write_prediction_csv(layout["inference_results"] / "baseline_force_predictions.csv", prediction_rows)
    inference_payload = {
        "sequence_name": config["sequence_name"],
        "baseline": train_payload.get("baseline"),
        "status": train_payload.get("status"),
        "validation_mae": train_payload.get("validation_mae"),
        "validation_rmse": train_payload.get("validation_rmse"),
        "prediction_csv": (layout["inference_results"] / "baseline_force_predictions.csv").as_posix(),
    }
    dump_json(layout["inference_results"] / "inference_metrics.json", inference_payload)
    return {
        "dataset_summary": dataset_summary,
        "training_metrics": train_payload,
        "inference_metrics": inference_payload,
    }


def compact_stage_status(stages: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "enabled": bool(info.get("enabled", False)),
            "run_dir": str(info.get("run_dir", "")),
            "video_count": len(info.get("videos", [])),
            "log_count": len(info.get("logs", [])),
        }
        for name, info in stages.items()
    }


def git_info() -> dict[str, Any]:
    def read_git(args: list[str]) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    try:
        return {
            "commit": read_git(["rev-parse", "HEAD"]),
            "short_commit": read_git(["rev-parse", "--short", "HEAD"]),
            "branch": read_git(["branch", "--show-current"]),
            "dirty": bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()),
        }
    except Exception as exc:
        return {"error": str(exc)}


def np_empty() -> Any:
    import numpy as np

    return np.asarray([], dtype=np.float32)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    sequence_name = sanitize_name(args.sequence_name or config.get("sequence_name", "unnamed_sequence"))
    config["sequence_name"] = sequence_name
    config["quick"] = bool(args.quick)
    config["skip_bridge"] = bool(args.skip_bridge)
    layout = make_layout(sequence_name, config.get("output_root", "outputs/experiments"))
    ensure_layout(layout)

    dump_json(layout["config"] / "source_experiment_config.json", load_json(args.config))
    resolved_config = layout["config"] / "resolved_experiment_config.json"
    dump_json(resolved_config, config)
    dump_json(layout["config"] / "git_info.json", git_info())
    dump_json(
        layout["config"] / "workspace_scan.json",
        scan_workspace(ROOT, max_depth=int(config.get("workspace", {}).get("scan_depth", 2))),
    )

    stages = {
        "density_render": stage_density(config, layout, args.quick, args.metrics_only),
        "blade_demo": stage_blade(config, layout, args.quick, args.metrics_only),
        "newton_bridge": stage_bridge(config, layout, args.quick, args.skip_bridge, args.metrics_only),
    }
    dataset_info = stage_probing_dataset(config, layout, args.quick)
    metrics = compute_metrics(config, layout, stages)
    learning_info = stage_learning(
        config=config,
        layout=layout,
        dataset_info=dataset_info,
        resolved_config=resolved_config,
        quick=args.quick,
        skip_training=args.skip_training,
        metrics_only=args.metrics_only,
    )
    manifest = {
        "sequence_name": sequence_name,
        "root": layout["root"].as_posix(),
        "folders": {key: path.as_posix() for key, path in layout.items() if key != "root"},
        "stages": compact_stage_status(stages),
        "probing_dataset": dataset_info,
        "learning": learning_info,
        "metrics": {
            "dataset_summary": (layout["dataset_metrics"] / "dataset_summary.json").as_posix(),
            "video_metrics": (layout["dataset_metrics"] / "video_metrics.json").as_posix(),
            "probing_tensor_metrics": (layout["dataset_metrics"] / "probing_tensor_metrics.json").as_posix(),
            "training_metrics": (layout["training_metrics"] / "baseline_force_model.json").as_posix(),
            "learning_training_metrics": (layout["training_metrics"] / "mdn_training_metrics.json").as_posix(),
            "inference_metrics": (layout["inference_results"] / "inference_metrics.json").as_posix(),
            "learning_inference_metrics": (layout["inference_results"] / "learning_inference_metrics.json").as_posix(),
        },
        "summary": metrics,
    }
    dump_json(layout["root"] / "experiment_manifest.json", manifest)
    print(f"sequence={sequence_name}")
    print(f"root={layout['root']}")
    print(f"manifest={layout['root'] / 'experiment_manifest.json'}")


if __name__ == "__main__":
    main()
