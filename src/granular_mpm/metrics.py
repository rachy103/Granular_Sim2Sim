from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def video_metrics(path: Path, sample_frames: int = 8) -> dict[str, Any]:
    cap = cv2.VideoCapture(path.as_posix())
    if not cap.isOpened():
        return {"path": path.as_posix(), "exists": path.exists(), "readable": False}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    means: list[float] = []
    nonzero: list[float] = []
    if frame_count > 0:
        indices = np.linspace(0, max(frame_count - 1, 0), min(sample_frames, frame_count), dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            means.append(float(gray.mean()))
            nonzero.append(float(np.count_nonzero(gray > 8) / gray.size))
    cap.release()

    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "readable": True,
        "frame_count": frame_count,
        "fps": fps,
        "duration_sec": frame_count / fps if fps > 0 else None,
        "width": width,
        "height": height,
        "sample_mean_luma": _stats(means),
        "sample_nonzero_fraction": _stats(nonzero),
    }


def bridge_log_metrics(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    frames = np.asarray(data["frame"], dtype=np.int32)
    forces = np.asarray(data["force"], dtype=np.float32)
    tool = np.asarray(data["tool_pos"], dtype=np.float32)
    force_norm = np.linalg.norm(forces, axis=1) if forces.size else np.asarray([], dtype=np.float32)
    particle_entries = list(data["particle_pos"]) if "particle_pos" in data else []

    particle_count = 0
    z_min = z_max = None
    mean_displacement = None
    if particle_entries:
        p0 = np.asarray(particle_entries[0], dtype=np.float32)
        pn = np.asarray(particle_entries[-1], dtype=np.float32)
        particle_count = int(p0.shape[0])
        z_min = float(min(np.min(np.asarray(p, dtype=np.float32)[:, 2]) for p in particle_entries))
        z_max = float(max(np.max(np.asarray(p, dtype=np.float32)[:, 2]) for p in particle_entries))
        if p0.shape == pn.shape:
            mean_displacement = float(np.linalg.norm(pn - p0, axis=1).mean())

    return {
        "path": path.as_posix(),
        "frames_logged": int(frames.size),
        "frame_min": int(frames.min()) if frames.size else None,
        "frame_max": int(frames.max()) if frames.size else None,
        "force_norm": _stats(force_norm),
        "tool_path_length": _path_length(tool),
        "particle_count": particle_count,
        "particle_z_range": [z_min, z_max],
        "mean_particle_displacement": mean_displacement,
        "voxel_size": _scalar(data, "voxel_size"),
        "particles_per_cell": _scalar(data, "particles_per_cell"),
        "sand_density": _scalar(data, "sand_density"),
        "sand_friction": _scalar(data, "sand_friction"),
        "sand_young_modulus": _scalar(data, "sand_young_modulus"),
        "sand_render_mode": str(data["sand_render_mode"]) if "sand_render_mode" in data else None,
    }


def blade_log_metrics(csv_path: Path) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if v != ""})
    force = np.asarray([r.get("display_force_norm", 0.0) for r in rows], dtype=np.float32)
    tool = np.asarray([[r.get("tool_x", 0.0), r.get("tool_y", 0.0), r.get("tool_z", 0.0)] for r in rows], dtype=np.float32)
    z_min = [r.get("z_min", np.nan) for r in rows]
    z_max = [r.get("z_max", np.nan) for r in rows]
    return {
        "path": csv_path.as_posix(),
        "frames_logged": len(rows),
        "force_norm": _stats(force),
        "tool_path_length": _path_length(tool),
        "particle_z_range": [float(np.nanmin(z_min)), float(np.nanmax(z_max))] if rows else None,
    }


def force_series_from_bridge(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    forces = np.asarray(data["force"], dtype=np.float32)
    if forces.ndim != 2 or forces.shape[1] != 3:
        return np.asarray([], dtype=np.float32)
    return np.linalg.norm(forces, axis=1).astype(np.float32)


def force_series_from_blade(csv_path: Path) -> np.ndarray:
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            values.append(float(row.get("display_force_norm", 0.0)))
    return np.asarray(values, dtype=np.float32)


def baseline_force_metrics(force_norm: np.ndarray, train_fraction: float = 0.7) -> tuple[dict[str, Any], list[dict[str, float]]]:
    y = np.asarray(force_norm, dtype=np.float32)
    if y.size < 3:
        return (
            {
                "status": "insufficient_data",
                "sample_count": int(y.size),
                "train_fraction": train_fraction,
            },
            [],
        )

    pred = np.empty_like(y)
    pred[0] = y[0]
    pred[1:] = y[:-1]
    split = int(np.clip(round(y.size * train_fraction), 1, y.size - 1))
    train_err = pred[:split] - y[:split]
    val_err = pred[split:] - y[split:]
    rows = [
        {
            "index": float(i),
            "target_force_norm": float(y[i]),
            "pred_force_norm": float(pred[i]),
            "abs_error": float(abs(pred[i] - y[i])),
            "split": 0.0 if i < split else 1.0,
        }
        for i in range(y.size)
    ]
    return (
        {
            "status": "ok",
            "baseline": "previous_force_norm",
            "sample_count": int(y.size),
            "train_fraction": float(train_fraction),
            "train_count": int(split),
            "validation_count": int(y.size - split),
            "train_mae": _mae(train_err),
            "train_rmse": _rmse(train_err),
            "validation_mae": _mae(val_err),
            "validation_rmse": _rmse(val_err),
            "target_force_norm": _stats(y),
        },
        rows,
    )


def write_prediction_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "target_force_norm", "pred_force_norm", "abs_error", "split"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _path_length(points: np.ndarray) -> float:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def _mae(errors: np.ndarray) -> float | None:
    arr = np.asarray(errors, dtype=np.float32)
    if arr.size == 0:
        return None
    return float(np.mean(np.abs(arr)))


def _rmse(errors: np.ndarray) -> float | None:
    arr = np.asarray(errors, dtype=np.float32)
    if arr.size == 0:
        return None
    return float(np.sqrt(np.mean(arr * arr)))


def _scalar(data: Any, key: str) -> Any:
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.shape == ():
        return value.item()
    return value.tolist()
