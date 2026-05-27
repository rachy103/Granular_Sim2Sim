from __future__ import annotations

import csv

import numpy as np

from granular_mpm.metrics import baseline_force_metrics, blade_log_metrics


def test_baseline_force_metrics_has_validation_error() -> None:
    metrics, rows = baseline_force_metrics(np.asarray([0.0, 1.0, 3.0, 6.0], dtype=np.float32), train_fraction=0.5)
    assert metrics["status"] == "ok"
    assert metrics["train_count"] == 2
    assert metrics["validation_count"] == 2
    assert metrics["validation_mae"] > 0.0
    assert len(rows) == 4


def test_blade_log_metrics_reads_force_series(tmp_path) -> None:
    path = tmp_path / "wrench_log.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tool_x", "tool_y", "tool_z", "display_force_norm", "z_min", "z_max"],
        )
        writer.writeheader()
        writer.writerow({"tool_x": 0, "tool_y": 0, "tool_z": 0, "display_force_norm": 1, "z_min": 0.1, "z_max": 0.2})
        writer.writerow({"tool_x": 1, "tool_y": 0, "tool_z": 0, "display_force_norm": 3, "z_min": 0.0, "z_max": 0.4})
    metrics = blade_log_metrics(path)
    assert metrics["frames_logged"] == 2
    assert metrics["force_norm"]["max"] == 3.0
    assert metrics["tool_path_length"] == 1.0
