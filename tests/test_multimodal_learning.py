from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from granular_mpm.multimodal_learning import (  # noqa: E402
    OnlineMohrCoulombBeliefNet,
    SENSOR_FEATURE_NAMES,
    generate_synthetic_multimodal_blade_dataset,
)
from granular_mpm.wild_material_learning import (  # noqa: E402
    WILD_FAMILY_NAMES,
    WildMaterialBeliefNet,
    generate_wild_material_dataset,
)


def test_online_belief_precision_is_cumulative() -> None:
    model = OnlineMohrCoulombBeliefNet(sensor_dim=5, vision_dim=4, context_dim=3, target_dim=4)
    sensor = torch.randn(2, 6, 5)
    vision = torch.randn(2, 6, 4)
    context = torch.randn(2, 3)
    output = model(sensor, vision, context)
    precision = output["precision"].detach().numpy()
    assert precision.shape == (2, 6, 4)
    assert np.all(np.diff(precision, axis=1) >= -1.0e-6)
    assert output["mu"].shape == (2, 6, 4)
    gate = output["modality_gate"].detach().numpy()
    assert gate.shape == (2, 6, 3)
    assert np.allclose(gate.sum(axis=-1), 1.0, atol=1.0e-6)


def test_synthetic_multimodal_dataset_has_grouped_splits() -> None:
    dataset = generate_synthetic_multimodal_blade_dataset(
        {
            "seed": 5,
            "quick_material_count": 8,
            "quick_actions_per_material": 2,
            "quick_frames": 10,
        },
        quick=True,
    )
    assert dataset["sensor"].shape[:2] == (16, 10)
    assert dataset["sensor"].shape[-1] == len(SENSOR_FEATURE_NAMES)
    assert dataset["vision"].shape[:2] == (16, 10)
    assert dataset["target"].shape == (16, 4)
    material_index = dataset["material_index"]
    split = dataset["split"]
    for material_id in np.unique(material_index):
        material_splits = np.unique(split[material_index == material_id])
        assert material_splits.size == 1


def test_wild_material_dataset_has_four_families_and_heldout_actions() -> None:
    dataset = generate_wild_material_dataset(
        {
            "seed": 11,
            "quick_materials_per_family": 5,
            "quick_actions_per_material": 6,
            "quick_frames": 8,
            "heldout_action_ids": [4, 5],
        },
        quick=True,
    )
    assert set(np.unique(dataset["family"]).tolist()) == set(range(len(WILD_FAMILY_NAMES)))
    assert dataset["sensor"].shape[-1] == len(SENSOR_FEATURE_NAMES)
    assert dataset["target"].shape[-1] == 4
    heldout = np.isin(dataset["action_index"], [4, 5])
    assert not np.any(dataset["split"][heldout] == 0)
    assert np.any(dataset["split"][heldout] == 1)
    assert np.any(dataset["split"][heldout] == 2)


def test_wild_material_model_emits_family_logits() -> None:
    model = WildMaterialBeliefNet(sensor_dim=5, vision_dim=4, context_dim=3, target_dim=4, family_count=4)
    sensor = torch.randn(2, 6, 5)
    vision = torch.randn(2, 6, 4)
    context = torch.randn(2, 3)
    output = model(sensor, vision, context)
    assert output["mu"].shape == (2, 6, 4)
    assert output["family_logits"].shape == (2, 6, 4)
    assert output["modality_gate"].shape == (2, 6, 3)
