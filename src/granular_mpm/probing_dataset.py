from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES = [
    "fx",
    "fy",
    "fz",
    "tx",
    "ty",
    "tz",
    "px",
    "py",
    "pz",
    "vx",
    "vy",
    "vz",
]

DEFAULT_TARGET_NAMES = ["phi_deg", "cohesion_kpa"]


@dataclass(frozen=True)
class ProbingSource:
    name: str
    time: np.ndarray
    features: np.ndarray
    path: str


def load_blade_wrench_csv(path: Path, name: str = "blade_demo") -> ProbingSource:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if v != ""})
    if not rows:
        return _empty_source(name, path)

    time = np.asarray([row.get("time", float(i)) for i, row in enumerate(rows)], dtype=np.float32)
    pos = np.asarray(
        [[row.get("tool_x", 0.0), row.get("tool_y", 0.0), row.get("tool_z", 0.0)] for row in rows],
        dtype=np.float32,
    )
    vel = finite_difference_velocity(time, pos)
    wrench = np.asarray(
        [
            [
                row.get("raw_fx", row.get("fx", 0.0)),
                row.get("raw_fy", row.get("fy", 0.0)),
                row.get("raw_fz", row.get("fz", row.get("display_force_norm", 0.0))),
                row.get("raw_tx", row.get("tx", 0.0)),
                row.get("raw_ty", row.get("ty", 0.0)),
                row.get("raw_tz", row.get("tz", 0.0)),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    features = np.concatenate([wrench, pos, vel], axis=1)
    return ProbingSource(name=name, time=time, features=features, path=path.as_posix())


def load_bridge_npz(path: Path, name: str = "newton_bridge", frame_rate_hz: float = 30.0) -> ProbingSource:
    data = np.load(path, allow_pickle=True)
    frames = np.asarray(data["frame"], dtype=np.float32) if "frame" in data else np.arange(len(data["force"]))
    time = frames / float(frame_rate_hz)
    force = np.asarray(data["force"], dtype=np.float32)
    if force.ndim != 2 or force.shape[1] != 3:
        return _empty_source(name, path)
    torque = np.zeros_like(force, dtype=np.float32)
    pos = np.asarray(data["tool_pos"], dtype=np.float32)
    vel = finite_difference_velocity(time, pos)
    features = np.concatenate([force, torque, pos, vel], axis=1)
    return ProbingSource(name=name, time=time.astype(np.float32), features=features, path=path.as_posix())


def finite_difference_velocity(time: np.ndarray, pos: np.ndarray) -> np.ndarray:
    t = np.asarray(time, dtype=np.float32)
    p = np.asarray(pos, dtype=np.float32)
    if p.ndim != 2 or p.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if p.shape[0] == 1:
        return np.zeros_like(p, dtype=np.float32)
    dt = np.diff(t)
    dt = np.where(np.abs(dt) < 1.0e-6, 1.0, dt)
    seg = np.diff(p, axis=0) / dt[:, None]
    vel = np.zeros_like(p, dtype=np.float32)
    vel[0] = seg[0]
    vel[-1] = seg[-1]
    if p.shape[0] > 2:
        vel[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    return vel


def resample_source(source: ProbingSource, sample_rate_hz: float) -> ProbingSource:
    time = np.asarray(source.time, dtype=np.float32)
    features = np.asarray(source.features, dtype=np.float32)
    if time.size < 2 or features.shape[0] < 2:
        return source

    order = np.argsort(time)
    time = time[order]
    features = features[order]
    keep = np.concatenate([[True], np.diff(time) > 1.0e-7])
    time = time[keep]
    features = features[keep]
    if time.size < 2:
        return ProbingSource(source.name, time, features, source.path)

    dt = 1.0 / float(sample_rate_hz)
    new_time = np.arange(float(time[0]), float(time[-1]) + 0.5 * dt, dt, dtype=np.float32)
    new_features = np.stack(
        [np.interp(new_time, time, features[:, col]).astype(np.float32) for col in range(features.shape[1])],
        axis=1,
    )
    return ProbingSource(source.name, new_time, new_features, source.path)


def sliding_windows(features: np.ndarray, window_length: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < window_length:
        return np.zeros((0, window_length, len(FEATURE_NAMES)), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    starts = np.arange(0, x.shape[0] - window_length + 1, max(1, stride), dtype=np.int32)
    windows = np.stack([x[start : start + window_length] for start in starts], axis=0).astype(np.float32)
    return windows, starts


def build_probing_dataset(
    sources: list[ProbingSource],
    targets: dict[str, float],
    sample_rate_hz: float = 50.0,
    window_length: int = 32,
    stride: int = 8,
    normalization: str = "zscore",
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 7,
) -> dict[str, Any]:
    target_names = list(targets.keys()) or list(DEFAULT_TARGET_NAMES)
    target_values = np.asarray([targets.get(name, 0.0) for name in target_names], dtype=np.float32)

    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    source_ids: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    source_meta: list[dict[str, Any]] = []

    for source_id, source in enumerate(sources):
        sampled = resample_source(source, sample_rate_hz)
        source_windows, source_starts = sliding_windows(sampled.features, window_length, stride)
        if source_windows.size:
            windows.append(source_windows)
            labels.append(np.repeat(target_values[None, :], source_windows.shape[0], axis=0))
            source_ids.append(np.full((source_windows.shape[0],), source_id, dtype=np.int32))
            starts.append(source_starts)
        source_meta.append(
            {
                "name": source.name,
                "path": source.path,
                "raw_samples": int(source.time.size),
                "resampled_samples": int(sampled.time.size),
                "windows": int(source_windows.shape[0]),
            }
        )

    if windows:
        x_raw = np.concatenate(windows, axis=0).astype(np.float32)
        y = np.concatenate(labels, axis=0).astype(np.float32)
        source_index = np.concatenate(source_ids, axis=0).astype(np.int32)
        window_start = np.concatenate(starts, axis=0).astype(np.int32)
    else:
        x_raw = np.zeros((0, window_length, len(FEATURE_NAMES)), dtype=np.float32)
        y = np.zeros((0, len(target_names)), dtype=np.float32)
        source_index = np.zeros((0,), dtype=np.int32)
        window_start = np.zeros((0,), dtype=np.int32)

    split = make_splits(x_raw.shape[0], train_fraction, validation_fraction, seed)
    x, norm_stats = normalize_windows(x_raw, split, normalization)
    return {
        "x": x,
        "x_raw": x_raw,
        "y": y,
        "split": split,
        "source_index": source_index,
        "window_start": window_start,
        "metadata": {
            "sample_rate_hz": float(sample_rate_hz),
            "window_length": int(window_length),
            "stride": int(stride),
            "normalization": normalization,
            "feature_names": list(FEATURE_NAMES),
            "target_names": target_names,
            "targets": {name: float(targets.get(name, 0.0)) for name in target_names},
            "source_count": len(sources),
            "sources": source_meta,
            "normalization_stats": norm_stats,
        },
    }


def make_splits(
    sample_count: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 7,
) -> np.ndarray:
    split = np.full((sample_count,), 2, dtype=np.int32)
    if sample_count == 0:
        return split
    rng = np.random.default_rng(seed)
    order = rng.permutation(sample_count)
    train_count = int(np.clip(round(sample_count * train_fraction), 1, sample_count))
    remaining = max(0, sample_count - train_count)
    val_count = int(np.clip(round(sample_count * validation_fraction), 0, remaining))
    split[order[:train_count]] = 0
    split[order[train_count : train_count + val_count]] = 1
    split[order[train_count + val_count :]] = 2
    if sample_count >= 3 and np.count_nonzero(split == 2) == 0:
        split[order[-1]] = 2
    if sample_count >= 4 and np.count_nonzero(split == 1) == 0:
        split[order[-2]] = 1
    return split


def normalize_windows(windows: np.ndarray, split: np.ndarray, method: str = "zscore") -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(windows, dtype=np.float32)
    if x.size == 0 or method == "none":
        return x.copy(), {"method": "none"}
    if method != "zscore":
        raise ValueError(f"Unsupported normalization method: {method}")

    train = x[split == 0] if np.any(split == 0) else x
    mean = train.reshape(-1, train.shape[-1]).mean(axis=0)
    std = train.reshape(-1, train.shape[-1]).std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    x_norm = (x - mean[None, None, :]) / std[None, None, :]
    return x_norm.astype(np.float32), {
        "method": "zscore",
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "feature_names": list(FEATURE_NAMES),
    }


def write_dataset_npz(path: Path, dataset: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sequence_index = dataset.get("sequence_index", np.zeros_like(dataset["source_index"], dtype=np.int32))
    material_index = dataset.get("material_index", sequence_index)
    np.savez_compressed(
        path,
        x=dataset["x"],
        x_raw=dataset["x_raw"],
        y=dataset["y"],
        split=dataset["split"],
        source_index=dataset["source_index"],
        sequence_index=sequence_index,
        material_index=material_index,
        window_start=dataset["window_start"],
        metadata=np.asarray(json.dumps(dataset["metadata"], indent=2)),
    )


def load_dataset_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        "x": np.asarray(data["x"], dtype=np.float32),
        "x_raw": np.asarray(data["x_raw"], dtype=np.float32),
        "y": np.asarray(data["y"], dtype=np.float32),
        "split": np.asarray(data["split"], dtype=np.int32),
        "source_index": np.asarray(data["source_index"], dtype=np.int32),
        "sequence_index": np.asarray(data["sequence_index"], dtype=np.int32)
        if "sequence_index" in data
        else np.asarray(data["source_index"], dtype=np.int32),
        "material_index": np.asarray(data["material_index"], dtype=np.int32)
        if "material_index" in data
        else (
            np.asarray(data["sequence_index"], dtype=np.int32)
            if "sequence_index" in data
            else np.asarray(data["source_index"], dtype=np.int32)
        ),
        "window_start": np.asarray(data["window_start"], dtype=np.int32),
        "metadata": json.loads(str(data["metadata"].item())),
    }


def probing_dataset_metrics(dataset: dict[str, Any]) -> dict[str, Any]:
    x = np.asarray(dataset["x"], dtype=np.float32)
    y = np.asarray(dataset["y"], dtype=np.float32)
    split = np.asarray(dataset["split"], dtype=np.int32)
    meta = dict(dataset["metadata"])
    split_names = {0: "train", 1: "validation", 2: "test"}
    return {
        "status": "ok" if x.shape[0] > 0 else "empty",
        "sample_count": int(x.shape[0]),
        "window_shape": list(x.shape[1:]),
        "label_shape": list(y.shape[1:]),
        "feature_names": meta.get("feature_names", FEATURE_NAMES),
        "target_names": meta.get("target_names", DEFAULT_TARGET_NAMES),
        "sample_rate_hz": meta.get("sample_rate_hz"),
        "normalization": meta.get("normalization"),
        "split_counts": {name: int(np.count_nonzero(split == code)) for code, name in split_names.items()},
        "feature_mean": _stats_by_last_dim(x, "mean"),
        "feature_std": _stats_by_last_dim(x, "std"),
        "targets": meta.get("targets", {}),
        "target_stats": meta.get("target_stats", _target_stats(y, meta.get("target_names", DEFAULT_TARGET_NAMES))),
        "sources": meta.get("sources", []),
    }


def _empty_source(name: str, path: Path) -> ProbingSource:
    return ProbingSource(
        name=name,
        time=np.zeros((0,), dtype=np.float32),
        features=np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
        path=path.as_posix(),
    )


def _stats_by_last_dim(values: np.ndarray, op: str) -> dict[str, float | None]:
    x = np.asarray(values, dtype=np.float32)
    if x.size == 0:
        return {name: None for name in FEATURE_NAMES}
    flat = x.reshape(-1, x.shape[-1])
    if op == "mean":
        arr = flat.mean(axis=0)
    elif op == "std":
        arr = flat.std(axis=0)
    else:
        raise ValueError(op)
    return {FEATURE_NAMES[i]: float(arr[i]) for i in range(len(FEATURE_NAMES))}


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
