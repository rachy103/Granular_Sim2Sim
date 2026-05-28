from __future__ import annotations

import numpy as np

from granular_mpm.sweep import (
    apply_sample_to_config,
    dp_alpha_from_phi,
    latin_hypercube,
    make_group_splits,
    paired_lhs_samples,
    sample_to_material_controls,
)


def test_latin_hypercube_stays_inside_ranges_and_varies() -> None:
    samples = latin_hypercube({"phi_deg": [25.0, 45.0], "cohesion_kpa": [0.0, 15.0]}, count=8, seed=3)
    phi = np.asarray([sample["phi_deg"] for sample in samples])
    cohesion = np.asarray([sample["cohesion_kpa"] for sample in samples])
    assert len(samples) == 8
    assert float(phi.min()) >= 25.0
    assert float(phi.max()) <= 45.0
    assert float(cohesion.min()) >= 0.0
    assert float(cohesion.max()) <= 15.0
    assert np.unique(np.round(phi, 3)).size == 8


def test_material_mapping_increases_with_phi_and_cohesion() -> None:
    loose = sample_to_material_controls({"phi_deg": 25.0, "cohesion_kpa": 0.0})
    dense = sample_to_material_controls({"phi_deg": 45.0, "cohesion_kpa": 15.0})
    assert dense["dp_alpha"] > loose["dp_alpha"]
    assert dense["cohesion"] > loose["cohesion"]
    assert dp_alpha_from_phi(34.0) > 0.0


def test_apply_sample_to_config_sets_targets_and_trajectory() -> None:
    base = {"stages": {"blade_demo": {"overrides": {}}, "newton_bridge": {"overrides": {}}}, "dataset": {}}
    sample = {"phi_deg": 31.0, "cohesion_kpa": 4.0, "speed_scale": 1.1, "angle_offset": 0.05}
    out = apply_sample_to_config(base, sample)
    assert out["dataset"]["targets"]["phi_deg"] == 31.0
    assert out["stages"]["blade_demo"]["overrides"]["mpm"]["cohesion"] > 0.0
    assert out["stages"]["blade_demo"]["overrides"]["trajectory"]["speed_scale"] == 1.1
    assert "sand_friction" in out["stages"]["newton_bridge"]["overrides"]


def test_group_splits_keep_sequence_windows_together() -> None:
    group_ids = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    split = make_group_splits(group_ids, train_fraction=0.5, validation_fraction=0.25, seed=4)
    for group in np.unique(group_ids):
        assert np.unique(split[group_ids == group]).size == 1


def test_paired_lhs_repeats_actions_for_each_material() -> None:
    samples = paired_lhs_samples(
        material_ranges={"phi_deg": [25.0, 45.0], "cohesion_kpa": [0.0, 15.0]},
        action_ranges={"speed_scale": [0.8, 1.2]},
        material_count=3,
        actions_per_material=2,
        seed=5,
    )
    assert len(samples) == 6
    material_ids = [int(sample["material_id"]) for sample in samples]
    assert material_ids.count(0) == 2
    assert material_ids.count(1) == 2
    assert material_ids.count(2) == 2
