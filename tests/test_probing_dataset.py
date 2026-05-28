from __future__ import annotations

import csv

import numpy as np

from granular_mpm.probing_dataset import (
    FEATURE_NAMES,
    build_probing_dataset,
    load_blade_wrench_csv,
    probing_dataset_metrics,
)


def test_build_probing_dataset_makes_twelve_channel_windows(tmp_path) -> None:
    log_path = tmp_path / "wrench_log.csv"
    fieldnames = [
        "time",
        "tool_x",
        "tool_y",
        "tool_z",
        "raw_fx",
        "raw_fy",
        "raw_fz",
        "raw_tx",
        "raw_ty",
        "raw_tz",
    ]
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(8):
            writer.writerow(
                {
                    "time": i * 0.02,
                    "tool_x": 0.1 + 0.01 * i,
                    "tool_y": 0.2,
                    "tool_z": 0.3,
                    "raw_fx": i,
                    "raw_fy": 0.0,
                    "raw_fz": 2.0 * i,
                    "raw_tx": 0.1 * i,
                    "raw_ty": 0.0,
                    "raw_tz": 0.0,
                }
            )

    source = load_blade_wrench_csv(log_path)
    dataset = build_probing_dataset(
        [source],
        targets={"phi_deg": 34.0, "cohesion_kpa": 0.05},
        sample_rate_hz=50.0,
        window_length=4,
        stride=2,
        normalization="zscore",
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    assert dataset["x"].shape[1:] == (4, len(FEATURE_NAMES))
    assert dataset["y"].shape[1] == 2
    assert dataset["metadata"]["normalization_stats"]["method"] == "zscore"

    metrics = probing_dataset_metrics(dataset)
    assert metrics["status"] == "ok"
    assert metrics["split_counts"]["train"] >= 1


def test_build_probing_dataset_handles_empty_sources() -> None:
    dataset = build_probing_dataset(
        [],
        targets={"phi_deg": 34.0},
        window_length=4,
        stride=1,
    )
    assert dataset["x"].shape == (0, 4, len(FEATURE_NAMES))
    assert np.count_nonzero(dataset["split"]) == 0
