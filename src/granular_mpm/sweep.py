from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import write_json
from .probing_dataset import (
    FEATURE_NAMES,
    make_splits,
    normalize_windows,
    probing_dataset_metrics,
    write_dataset_npz,
)


def latin_hypercube(ranges: dict[str, list[float]], count: int, seed: int = 7) -> list[dict[str, float]]:
    if count <= 0:
        return []
    rng = np.random.default_rng(seed)
    keys = list(ranges.keys())
    samples = [dict() for _ in range(count)]
    for key in keys:
        lo, hi = [float(v) for v in ranges[key]]
        bins = (np.arange(count, dtype=np.float32) + rng.random(count)) / float(count)
        rng.shuffle(bins)
        values = lo + bins * (hi - lo)
        for i, value in enumerate(values):
            samples[i][key] = float(value)
    return samples


def dp_alpha_from_phi(phi_deg: float, scale: float = 1.0) -> float:
    sin_phi = math.sin(math.radians(float(phi_deg)))
    alpha = 2.0 * sin_phi / (math.sqrt(3.0) * (3.0 - sin_phi))
    return float(alpha * scale)


def friction_mu_from_phi(phi_deg: float, scale: float = 1.0) -> float:
    return float(math.tan(math.radians(float(phi_deg))) * scale)


def sample_to_material_controls(sample: dict[str, float], mapping: dict[str, Any] | None = None) -> dict[str, float]:
    mapping = mapping or {}
    phi = float(sample.get("phi_deg", 34.0))
    cohesion_kpa = float(sample.get("cohesion_kpa", 0.0))
    alpha_scale = float(mapping.get("dp_alpha_scale", 1.0))
    cohesion_scale = float(mapping.get("cohesion_kpa_to_mpm", 0.004))
    tool_mu_scale = float(mapping.get("tool_mu_scale", 0.58))
    young_base = float(mapping.get("young_base", 1800.0))
    young_per_kpa = float(mapping.get("young_per_kpa", 45.0))
    return {
        "dp_alpha": dp_alpha_from_phi(phi, alpha_scale),
        "cohesion": cohesion_kpa * cohesion_scale,
        "tool_mu": friction_mu_from_phi(phi, tool_mu_scale),
        "young": young_base + young_per_kpa * cohesion_kpa,
    }


def sample_to_newton_controls(sample: dict[str, float], mapping: dict[str, Any] | None = None) -> dict[str, float]:
    mapping = mapping or {}
    phi = float(sample.get("phi_deg", 34.0))
    cohesion_kpa = float(sample.get("cohesion_kpa", 0.0))
    return {
        "sand_friction": friction_mu_from_phi(phi, float(mapping.get("newton_friction_scale", 1.0))),
        "sand_yield_pressure": float(mapping.get("yield_pressure_base", 7.5e5))
        + cohesion_kpa * float(mapping.get("yield_pressure_per_kpa", 4.5e4)),
        "sand_young_modulus": float(mapping.get("newton_young_base", 8.0e5))
        + cohesion_kpa * float(mapping.get("newton_young_per_kpa", 2.5e4)),
    }


def sample_to_trajectory(sample: dict[str, float]) -> dict[str, float]:
    keys = [
        "speed_scale",
        "depth_scale",
        "drag_distance_scale",
        "angle_offset",
        "y_offset",
        "z_offset",
    ]
    return {key: float(sample[key]) for key in keys if key in sample}


def apply_sample_to_config(base_config: dict[str, Any], sample: dict[str, float], mapping: dict[str, Any] | None = None) -> dict[str, Any]:
    config = deepcopy(base_config)
    material = sample_to_material_controls(sample, mapping)
    newton = sample_to_newton_controls(sample, mapping)
    trajectory = sample_to_trajectory(sample)

    config.setdefault("dataset", {})
    config["dataset"]["targets"] = {
        "phi_deg": float(sample.get("phi_deg", 34.0)),
        "cohesion_kpa": float(sample.get("cohesion_kpa", 0.0)),
    }
    blade = config.setdefault("stages", {}).setdefault("blade_demo", {})
    blade_overrides = blade.setdefault("overrides", {})
    blade_overrides.setdefault("mpm", {}).update(material)
    if trajectory:
        blade_overrides.setdefault("trajectory", {}).update(trajectory)

    bridge = config.setdefault("stages", {}).setdefault("newton_bridge", {})
    bridge.setdefault("overrides", {}).update(newton)
    return config


def write_samples_csv(path: Path, samples: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for sample in samples for key in sample.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({key: sample.get(key, "") for key in fieldnames})


def aggregate_dataset_npz(
    dataset_paths: list[Path],
    out_path: Path,
    normalization: str = "zscore",
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 7,
) -> dict[str, Any]:
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sequence_index: list[np.ndarray] = []
    source_metadata: list[dict[str, Any]] = []
    target_names: list[str] | None = None
    sample_rate_hz = None
    window_length = None
    stride = None

    for seq_id, path in enumerate(dataset_paths):
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        x_raw = np.asarray(data["x_raw"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)
        meta = json.loads(str(data["metadata"].item()))
        if x_raw.shape[0] == 0:
            continue
        if target_names is None:
            target_names = list(meta.get("target_names", []))
            sample_rate_hz = meta.get("sample_rate_hz")
            window_length = meta.get("window_length")
            stride = meta.get("stride")
        windows.append(x_raw)
        labels.append(y)
        sequence_index.append(np.full((x_raw.shape[0],), seq_id, dtype=np.int32))
        source_metadata.append(
            {
                "dataset_path": path.as_posix(),
                "windows": int(x_raw.shape[0]),
                "targets": meta.get("targets", {}),
                "sources": meta.get("sources", []),
            }
        )

    if windows:
        x_raw_all = np.concatenate(windows, axis=0).astype(np.float32)
        y_all = np.concatenate(labels, axis=0).astype(np.float32)
        seq_all = np.concatenate(sequence_index, axis=0).astype(np.int32)
    else:
        target_names = target_names or ["phi_deg", "cohesion_kpa"]
        x_raw_all = np.zeros((0, int(window_length or 1), len(FEATURE_NAMES)), dtype=np.float32)
        y_all = np.zeros((0, len(target_names)), dtype=np.float32)
        seq_all = np.zeros((0,), dtype=np.int32)

    split = make_group_splits(
        seq_all,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    x_all, norm_stats = normalize_windows(x_raw_all, split, normalization)
    dataset = {
        "x": x_all,
        "x_raw": x_raw_all,
        "y": y_all,
        "split": split,
        "source_index": seq_all,
        "sequence_index": seq_all,
        "window_start": np.zeros((x_raw_all.shape[0],), dtype=np.int32),
        "metadata": {
            "sample_rate_hz": sample_rate_hz,
            "window_length": window_length,
            "stride": stride,
            "normalization": normalization,
            "feature_names": list(FEATURE_NAMES),
            "target_names": target_names,
            "target_stats": _target_stats(y_all, target_names or []),
            "source_count": len(source_metadata),
            "sources": source_metadata,
            "normalization_stats": norm_stats,
        },
    }
    write_dataset_npz(out_path, dataset)
    metrics = probing_dataset_metrics(dataset)
    write_json(out_path.parent / "aggregate_dataset_metrics.json", metrics)
    write_json(out_path.parent / "normalization_stats.json", norm_stats)
    return {"dataset": dataset, "metrics": metrics, "path": out_path}


def _target_stats(values: np.ndarray, names: list[str]) -> dict[str, dict[str, float | None]]:
    y = np.asarray(values, dtype=np.float32)
    if y.size == 0:
        return {name: {"min": None, "max": None, "mean": None, "std": None} for name in names}
    return {
        names[i]: {
            "min": float(y[:, i].min()),
            "max": float(y[:, i].max()),
            "mean": float(y[:, i].mean()),
            "std": float(y[:, i].std()),
        }
        for i in range(min(len(names), y.shape[1]))
    }


def make_group_splits(
    group_ids: np.ndarray,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 7,
) -> np.ndarray:
    groups = np.asarray(group_ids, dtype=np.int32)
    if groups.size == 0:
        return np.zeros((0,), dtype=np.int32)
    unique = np.unique(groups)
    group_split = make_splits(
        unique.size,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    split_by_group = {int(group): int(group_split[i]) for i, group in enumerate(unique)}
    return np.asarray([split_by_group[int(group)] for group in groups], dtype=np.int32)
