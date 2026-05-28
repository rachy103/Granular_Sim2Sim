from __future__ import annotations

import csv

from granular_mpm.scatter_analysis import analyze_sweep_scatter


def test_analyze_sweep_scatter_detects_missing_contact(tmp_path) -> None:
    sweep = tmp_path / "sweep"
    seq = sweep / "sequences" / "seq_0000" / "runs" / "blade_demo"
    seq.mkdir(parents=True)
    samples_path = sweep / "samples.csv"
    with samples_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sequence_root", "phi_deg", "cohesion_kpa", "speed_scale", "depth_scale"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sequence_root": (sweep / "sequences" / "seq_0000").as_posix(),
                "phi_deg": 30,
                "cohesion_kpa": 5,
                "speed_scale": 1,
                "depth_scale": 1,
            }
        )

    with (seq / "wrench_log.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["time", "tool_x", "tool_y", "tool_z", "raw_fx", "raw_fy", "raw_fz", "raw_tx", "raw_ty", "raw_tz"],
        )
        writer.writeheader()
        for i in range(3):
            writer.writerow(
                {
                    "time": i * 0.02,
                    "tool_x": 0,
                    "tool_y": 0,
                    "tool_z": 0,
                    "raw_fx": 0,
                    "raw_fy": 0,
                    "raw_fz": 0,
                    "raw_tx": 0,
                    "raw_ty": 0,
                    "raw_tz": 0,
                }
            )

    report = analyze_sweep_scatter(sweep)
    assert "insufficient_contact" in report["diagnostics"]["issues"]
    assert (sweep / "analysis" / "scatter_report.md").exists()
