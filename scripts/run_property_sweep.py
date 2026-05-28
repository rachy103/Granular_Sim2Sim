from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm.metrics import write_json  # noqa: E402
from granular_mpm.sweep import (  # noqa: E402
    aggregate_dataset_npz,
    apply_sample_to_config,
    latin_hypercube,
    sample_to_material_controls,
    sample_to_newton_controls,
    sample_to_trajectory,
    write_samples_csv,
)


DEFAULT_CONFIG = ROOT / "configs" / "sweeps" / "lhs_property_sweep.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sweep-name", default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-bridge", action="store_true")
    parser.add_argument("--train-per-sequence", action="store_true")
    parser.add_argument("--skip-aggregate-training", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._-")
    if not cleaned:
        raise ValueError("Empty sweep name")
    return cleaned


def rel_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def make_layout(output_root: str, sweep_name: str) -> dict[str, Path]:
    root = ROOT / output_root / sweep_name
    return {
        "root": root,
        "configs": root / "configs",
        "sequences": root / "sequences",
        "dataset": root / "dataset",
        "training_metrics": root / "training_metrics",
        "inference_results": root / "inference_results",
        "logs": root / "logs",
    }


def ensure_layout(layout: dict[str, Path]) -> None:
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)


def disable_unneeded_stages(config: dict[str, Any], sweep_cfg: dict[str, Any]) -> None:
    stages = config.setdefault("stages", {})
    if not bool(sweep_cfg.get("run_density", False)):
        stages.setdefault("density_render", {})["enabled"] = False
    if not bool(sweep_cfg.get("run_bridge", False)):
        stages.setdefault("newton_bridge", {})["enabled"] = False


def run_sample_sequences(
    sweep_cfg: dict[str, Any],
    base_config: dict[str, Any],
    samples: list[dict[str, float]],
    layout: dict[str, Path],
    sweep_name: str,
    quick: bool,
    skip_bridge: bool,
    train_per_sequence: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    mapping = dict(sweep_cfg.get("material_mapping", {}))
    sequence_output_root = rel_to_root(layout["sequences"])
    for i, sample in enumerate(samples):
        sequence_name = f"{sweep_name}_{i:04d}"
        sample_config = apply_sample_to_config(base_config, sample, mapping)
        sample_config["sequence_name"] = sequence_name
        sample_config["output_root"] = sequence_output_root
        disable_unneeded_stages(sample_config, sweep_cfg)
        if not train_per_sequence:
            sample_config.setdefault("learning", {})["enabled"] = False

        config_path = layout["configs"] / f"{sequence_name}.json"
        write_json(config_path, sample_config)
        cmd = [sys.executable, "scripts/run_experiment_sequence.py", "--config", config_path.as_posix()]
        if quick:
            cmd.append("--quick")
        if skip_bridge:
            cmd.append("--skip-bridge")
        if not train_per_sequence:
            cmd.append("--skip-training")
        run_cmd(cmd, layout["logs"] / f"{sequence_name}.log")

        sequence_root = layout["sequences"] / sequence_name
        dataset_path = sequence_root / "runs" / "probing_dataset" / "probing_windows.npz"
        record = {
            "sample_id": i,
            "sequence_name": sequence_name,
            "sequence_root": sequence_root.as_posix(),
            "dataset_path": dataset_path.as_posix(),
            **sample,
            **{f"mpm_{k}": v for k, v in sample_to_material_controls(sample, mapping).items()},
            **{f"newton_{k}": v for k, v in sample_to_newton_controls(sample, mapping).items()},
            **{f"trajectory_{k}": v for k, v in sample_to_trajectory(sample).items()},
        }
        records.append(record)
    return records


def train_aggregate(
    sweep_cfg: dict[str, Any],
    base_config: dict[str, Any],
    dataset_path: Path,
    layout: dict[str, Path],
    sweep_name: str,
    quick: bool,
) -> dict[str, Any]:
    learning_cfg = merge_dict(base_config.get("learning", {}), sweep_cfg.get("learning", {}))
    learning_cfg["enabled"] = True
    config_path = layout["configs"] / "aggregate_learning_config.json"
    write_json(config_path, {"learning": learning_cfg})
    cmd = [
        sys.executable,
        "scripts/train_granular_inference.py",
        "--dataset",
        dataset_path.as_posix(),
        "--training-dir",
        layout["training_metrics"].as_posix(),
        "--inference-dir",
        layout["inference_results"].as_posix(),
        "--config",
        config_path.as_posix(),
        "--sequence-name",
        sweep_name,
    ]
    if quick:
        cmd.append("--quick")
    run_cmd(cmd, layout["logs"] / "aggregate_learning.log")
    return {
        "enabled": True,
        "training_metrics": (layout["training_metrics"] / "mdn_training_metrics.json").as_posix(),
        "inference_metrics": (layout["inference_results"] / "learning_inference_metrics.json").as_posix(),
        "predictions": (layout["inference_results"] / "mdn_predictions.csv").as_posix(),
    }


def main() -> None:
    args = parse_args()
    sweep_cfg = load_json(args.config)
    sweep_name = sanitize_name(args.sweep_name or sweep_cfg.get("sweep_name", "lhs_property_sweep"))
    count = int(args.count or sweep_cfg.get("count", 64))
    layout = make_layout(str(sweep_cfg.get("output_root", "outputs/sweeps")), sweep_name)
    ensure_layout(layout)

    base_config_path = ROOT / sweep_cfg.get("base_experiment_config", "configs/experiments/reference_heightfield_intrusion.json")
    base_config = load_json(base_config_path)
    samples = latin_hypercube(dict(sweep_cfg.get("ranges", {})), count=count, seed=int(sweep_cfg.get("seed", 7)))
    write_json(layout["configs"] / "source_sweep_config.json", sweep_cfg)
    write_json(layout["configs"] / "resolved_sweep_config.json", {**sweep_cfg, "sweep_name": sweep_name, "count": count})

    records = run_sample_sequences(
        sweep_cfg=sweep_cfg,
        base_config=base_config,
        samples=samples,
        layout=layout,
        sweep_name=sweep_name,
        quick=args.quick,
        skip_bridge=args.skip_bridge,
        train_per_sequence=args.train_per_sequence,
    )
    write_samples_csv(layout["root"] / "samples.csv", records)

    dataset_paths = [Path(record["dataset_path"]) for record in records]
    aggregate = aggregate_dataset_npz(
        dataset_paths=dataset_paths,
        out_path=layout["dataset"] / "probing_windows.npz",
        normalization=str(sweep_cfg.get("normalization", "zscore")),
        train_fraction=float(sweep_cfg.get("train_fraction", base_config.get("metrics", {}).get("train_fraction", 0.7))),
        validation_fraction=float(sweep_cfg.get("validation_fraction", 0.15)),
        seed=int(sweep_cfg.get("seed", 7)),
    )

    learning_info = {"enabled": False}
    if not args.skip_aggregate_training and int(aggregate["metrics"].get("sample_count", 0)) > 0:
        learning_info = train_aggregate(
            sweep_cfg=sweep_cfg,
            base_config=base_config,
            dataset_path=aggregate["path"],
            layout=layout,
            sweep_name=sweep_name,
            quick=args.quick,
        )

    manifest = {
        "sweep_name": sweep_name,
        "root": layout["root"].as_posix(),
        "count": count,
        "sample_csv": (layout["root"] / "samples.csv").as_posix(),
        "aggregate_dataset": aggregate["path"].as_posix(),
        "aggregate_metrics": (layout["dataset"] / "aggregate_dataset_metrics.json").as_posix(),
        "learning": learning_info,
        "records": records,
    }
    write_json(layout["root"] / "sweep_manifest.json", manifest)
    print(f"sweep={sweep_name}")
    print(f"root={layout['root']}")
    print(f"samples={layout['root'] / 'samples.csv'}")
    print(f"dataset={aggregate['path']}")
    print(f"manifest={layout['root'] / 'sweep_manifest.json'}")


if __name__ == "__main__":
    main()
