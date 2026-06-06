from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .metrics import write_json
from .multimodal_learning import (
    CONTEXT_FEATURE_NAMES,
    SENSOR_FEATURE_NAMES,
    TARGET_NAMES,
    VISION_FEATURE_NAMES,
    batch_indices,
    clean_float,
    clean_float_list,
    modality_balance_loss,
    normalize_feature_matrix,
    normalize_feature_tensor,
    online_gaussian_nll,
    rolling_force_descriptors,
    snapshot_model_state,
)

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover - exercised only on minimal installs.
    torch = None
    F = None
    nn = None


WILD_FAMILY_NAMES = ["gravel", "sand", "soil", "crunching"]
WILD_TAG_NAMES = [
    "train_known_action",
    "val_known_action",
    "val_heldout_action",
    "test_heldout_material",
    "test_heldout_action",
    "test_heldout_material_action",
]


if torch is not None:

    class WildMaterialBeliefNet(nn.Module):
        """Online posterior model with a material-family head.

        The model keeps the same cumulative Gaussian posterior contract as the
        Mohr-Coulomb estimator, but also emits a family classifier at every
        frame. The family head is trained on gravel/sand/soil/crunching labels.
        """

        def __init__(
            self,
            sensor_dim: int,
            vision_dim: int,
            context_dim: int,
            target_dim: int = 4,
            family_count: int = 4,
            hidden_dim: int = 128,
            token_dim: int = 128,
            dropout: float = 0.08,
            min_precision: float = 0.035,
            use_property_family_head: bool = True,
        ) -> None:
            super().__init__()
            self.target_dim = target_dim
            self.family_count = family_count
            self.min_precision = float(min_precision)
            self.use_property_family_head = bool(use_property_family_head)
            self.sensor_encoder = nn.Sequential(
                nn.LayerNorm(sensor_dim),
                nn.Linear(sensor_dim, token_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(token_dim, token_dim),
                nn.GELU(),
            )
            self.vision_encoder = nn.Sequential(
                nn.LayerNorm(vision_dim),
                nn.Linear(vision_dim, token_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(token_dim, token_dim),
                nn.GELU(),
            )
            self.context_encoder = nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Linear(context_dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
                nn.GELU(),
            )
            self.modality_gate = nn.Sequential(
                nn.LayerNorm(token_dim * 3),
                nn.Linear(token_dim * 3, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, 3),
            )
            self.fusion = nn.Sequential(
                nn.LayerNorm(token_dim * 5),
                nn.Linear(token_dim * 5, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.recurrent_update = nn.GRUCell(hidden_dim, hidden_dim)
            self.initial_hidden = nn.Linear(token_dim, hidden_dim)
            self.prior_mu = nn.Linear(token_dim, target_dim)
            self.prior_precision = nn.Linear(token_dim, target_dim)
            self.evidence_mu = nn.Linear(hidden_dim, target_dim)
            self.evidence_precision = nn.Linear(hidden_dim, target_dim)
            self.family_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + target_dim),
                nn.Linear(hidden_dim + target_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, family_count),
            )
            self.family_property_head = nn.Sequential(
                nn.LayerNorm(target_dim),
                nn.Linear(target_dim, max(32, hidden_dim // 2)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(32, hidden_dim // 2), family_count),
            )

        def forward(
            self,
            sensor: torch.Tensor,
            vision: torch.Tensor,
            context: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            _batch, frames, _sensor_dim = sensor.shape
            context_token = self.context_encoder(context)
            hidden = torch.tanh(self.initial_hidden(context_token))
            mu = self.prior_mu(context_token)
            precision = F.softplus(self.prior_precision(context_token)) + self.min_precision

            mu_seq: list[torch.Tensor] = []
            log_sigma_seq: list[torch.Tensor] = []
            precision_seq: list[torch.Tensor] = []
            family_logits_seq: list[torch.Tensor] = []
            update_norm_seq: list[torch.Tensor] = []
            modality_gate_seq: list[torch.Tensor] = []

            for frame_id in range(frames):
                sensor_token = self.sensor_encoder(sensor[:, frame_id])
                vision_token = self.vision_encoder(vision[:, frame_id])
                gate_input = torch.cat([sensor_token, vision_token, context_token], dim=-1)
                gate = torch.softmax(self.modality_gate(gate_input), dim=-1)
                cross_token = sensor_token * vision_token
                disagreement_token = sensor_token - vision_token
                fused = self.fusion(
                    torch.cat(
                        [
                            sensor_token * gate[:, 0:1],
                            vision_token * gate[:, 1:2],
                            context_token * gate[:, 2:3],
                            cross_token,
                            disagreement_token,
                        ],
                        dim=-1,
                    )
                )
                hidden = self.recurrent_update(fused, hidden)

                frame_mu = self.evidence_mu(hidden)
                frame_precision = F.softplus(self.evidence_precision(hidden)) + self.min_precision
                previous_mu = mu
                new_precision = precision + frame_precision
                mu = (precision * mu + frame_precision * frame_mu) / new_precision
                precision = new_precision
                family_logits = self.family_head(torch.cat([hidden, mu], dim=-1))
                if self.use_property_family_head:
                    family_logits = family_logits + self.family_property_head(mu)

                mu_seq.append(mu)
                log_sigma_seq.append(-0.5 * torch.log(torch.clamp(precision, min=1.0e-6)))
                precision_seq.append(precision)
                family_logits_seq.append(family_logits)
                update_norm_seq.append(torch.linalg.norm(mu - previous_mu, dim=-1))
                modality_gate_seq.append(gate)

            return {
                "mu": torch.stack(mu_seq, dim=1),
                "log_sigma": torch.stack(log_sigma_seq, dim=1),
                "precision": torch.stack(precision_seq, dim=1),
                "family_logits": torch.stack(family_logits_seq, dim=1),
                "update_norm": torch.stack(update_norm_seq, dim=1),
                "modality_gate": torch.stack(modality_gate_seq, dim=1),
            }

else:

    class WildMaterialBeliefNet:  # pragma: no cover
        pass


def generate_wild_material_dataset(config: dict[str, Any], quick: bool = False) -> dict[str, Any]:
    seed = int(config.get("seed", 101))
    rng = np.random.default_rng(seed)
    materials_per_family = int(config.get("quick_materials_per_family" if quick else "materials_per_family", 12 if quick else 32))
    actions_per_material = int(config.get("quick_actions_per_material" if quick else "actions_per_material", 6))
    frames = int(config.get("quick_frames" if quick else "frames", 30 if quick else 48))
    train_fraction = float(config.get("train_fraction", 0.58))
    validation_fraction = float(config.get("validation_fraction", 0.16))
    heldout_action_ids = set(int(v) for v in config.get("heldout_action_ids", [4, 5]))
    family_ranges = wild_family_ranges(config)

    split_by_family = {
        family_id: split_material_ids(materials_per_family, train_fraction, validation_fraction, seed + 7919 + family_id)
        for family_id in range(len(WILD_FAMILY_NAMES))
    }

    sensors: list[np.ndarray] = []
    visions: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    families: list[int] = []
    material_ids: list[int] = []
    action_ids: list[int] = []
    tag_ids: list[int] = []
    severity_values: list[float] = []

    global_material_id = 0
    for family_id, family_name in enumerate(WILD_FAMILY_NAMES):
        ranges = family_ranges[family_name]
        for local_material_id in range(materials_per_family):
            theta = sample_family_theta(ranges, rng)
            nuisance = sample_family_nuisance(family_id, rng)
            base_split = int(split_by_family[family_id][local_material_id])
            for action_id in range(actions_per_material):
                context = sample_action_context(action_id, rng)
                action_heldout = action_id in heldout_action_ids
                split, tag_id = split_and_tag(base_split, action_heldout)
                severity = sample_corruption_severity(split, tag_id, rng, config)
                sensor, vision = synthetic_wild_sequence(
                    theta=theta,
                    family_id=family_id,
                    nuisance=nuisance,
                    context=context,
                    frames=frames,
                    severity=severity,
                    rng=rng,
                )
                sensors.append(sensor)
                visions.append(vision)
                contexts.append(context)
                targets.append(theta)
                families.append(family_id)
                material_ids.append(global_material_id)
                action_ids.append(action_id)
                tag_ids.append(tag_id)
                severity_values.append(severity)
            global_material_id += 1

    return {
        "sensor": np.stack(sensors, axis=0).astype(np.float32),
        "vision": np.stack(visions, axis=0).astype(np.float32),
        "context": np.stack(contexts, axis=0).astype(np.float32),
        "target": np.stack(targets, axis=0).astype(np.float32),
        "family": np.asarray(families, dtype=np.int64),
        "split": np.asarray([0 if tag == 0 else 1 if tag in {1, 2} else 2 for tag in tag_ids], dtype=np.int32),
        "material_index": np.asarray(material_ids, dtype=np.int32),
        "action_index": np.asarray(action_ids, dtype=np.int32),
        "tag_index": np.asarray(tag_ids, dtype=np.int32),
        "corruption_severity": np.asarray(severity_values, dtype=np.float32),
        "metadata": {
            "generator": "wild_material_multimodal_blade",
            "family_names": WILD_FAMILY_NAMES,
            "tag_names": WILD_TAG_NAMES,
            "target_names": TARGET_NAMES,
            "sensor_feature_names": SENSOR_FEATURE_NAMES,
            "vision_feature_names": VISION_FEATURE_NAMES,
            "context_feature_names": CONTEXT_FEATURE_NAMES,
            "frames": frames,
            "materials_per_family": materials_per_family,
            "actions_per_material": actions_per_material,
            "heldout_action_ids": sorted(heldout_action_ids),
            "family_ranges": family_ranges,
            "wild_corruptions": [
                "sensor_gain_shift",
                "force_impulses",
                "vision_occlusion",
                "camera_scale_jitter",
                "lighting_shift",
                "feature_dropout",
                "grain-family-specific_chatter",
            ],
        },
    }


def wild_family_ranges(config: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    default = {
        "gravel": {
            "rho": [1650.0, 2250.0],
            "phi_deg": [38.0, 54.0],
            "delta_deg": [22.0, 40.0],
            "cohesion_kpa": [0.0, 4.0],
        },
        "sand": {
            "rho": [1250.0, 1750.0],
            "phi_deg": [28.0, 42.0],
            "delta_deg": [14.0, 31.0],
            "cohesion_kpa": [0.0, 3.0],
        },
        "soil": {
            "rho": [1150.0, 1800.0],
            "phi_deg": [20.0, 37.0],
            "delta_deg": [16.0, 35.0],
            "cohesion_kpa": [6.0, 28.0],
        },
        "crunching": {
            "rho": [180.0, 980.0],
            "phi_deg": [18.0, 46.0],
            "delta_deg": [8.0, 33.0],
            "cohesion_kpa": [0.0, 10.0],
        },
    }
    user_ranges = dict(config.get("family_ranges", {}))
    for family_name, ranges in user_ranges.items():
        if family_name not in default:
            continue
        for target_name, value in dict(ranges).items():
            if target_name in default[family_name]:
                default[family_name][target_name] = [float(value[0]), float(value[1])]
    return default


def sample_family_theta(ranges: dict[str, list[float]], rng: np.random.Generator) -> np.ndarray:
    rho = rng.uniform(*ranges["rho"])
    phi = rng.uniform(*ranges["phi_deg"])
    delta_hi = min(float(ranges["delta_deg"][1]), phi - 0.8)
    delta_lo = float(ranges["delta_deg"][0])
    delta = rng.uniform(delta_lo, max(delta_lo + 0.4, delta_hi))
    cohesion = rng.uniform(*ranges["cohesion_kpa"])
    return np.asarray([rho, phi, delta, cohesion], dtype=np.float32)


def sample_family_nuisance(family_id: int, rng: np.random.Generator) -> np.ndarray:
    if family_id == 0:  # gravel
        center = np.asarray([0.85, 0.18, 0.12, 0.82, 0.25, 0.70], dtype=np.float32)
    elif family_id == 1:  # sand
        center = np.asarray([0.35, 0.05, 0.04, 0.30, 0.18, 0.20], dtype=np.float32)
    elif family_id == 2:  # soil
        center = np.asarray([0.45, 0.72, 0.10, 0.46, 0.80, 0.24], dtype=np.float32)
    else:  # crunching
        center = np.asarray([0.62, 0.18, 0.88, 0.56, 0.36, 0.94], dtype=np.float32)
    return np.clip(center + rng.normal(0.0, 0.10, size=center.shape).astype(np.float32), 0.0, 1.0)


def split_material_ids(
    material_count: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.arange(material_count, dtype=np.int32))
    train_count = int(np.clip(round(material_count * train_fraction), 1, max(1, material_count - 2)))
    val_count = int(np.clip(round(material_count * validation_fraction), 1, max(1, material_count - train_count - 1)))
    split = np.full(material_count, 2, dtype=np.int32)
    split[order[:train_count]] = 0
    split[order[train_count : train_count + val_count]] = 1
    return split


def split_and_tag(base_split: int, action_heldout: bool) -> tuple[int, int]:
    if base_split == 0 and not action_heldout:
        return 0, 0
    if base_split == 1 and not action_heldout:
        return 1, 1
    if base_split == 1 and action_heldout:
        return 1, 2
    if base_split == 2 and action_heldout:
        return 2, 5
    if action_heldout:
        return 2, 4
    return 2, 3


def sample_corruption_severity(split: int, tag_id: int, rng: np.random.Generator, config: dict[str, Any]) -> float:
    train_range = tuple(float(v) for v in config.get("train_corruption_range", [0.04, 0.34]))
    val_range = tuple(float(v) for v in config.get("validation_corruption_range", [0.16, 0.44]))
    val_heldout_range = tuple(float(v) for v in config.get("validation_heldout_corruption_range", [0.26, 0.58]))
    test_range = tuple(float(v) for v in config.get("test_corruption_range", [0.34, 0.76]))
    test_joint_range = tuple(float(v) for v in config.get("test_joint_corruption_range", [0.46, 0.84]))
    if split == 0:
        return float(rng.uniform(*train_range))
    if split == 1:
        if tag_id == 2:
            return float(rng.uniform(*val_heldout_range))
        return float(rng.uniform(*val_range))
    if tag_id == 5:
        return float(rng.uniform(*test_joint_range))
    return float(rng.uniform(*test_range))


def sample_action_context(action_id: int, rng: np.random.Generator) -> np.ndarray:
    templates = [
        (0.12, 0.010, 22.0, 0.055, 0.050, 0.17),
        (0.16, 0.014, 28.0, 0.095, 0.078, 0.28),
        (0.14, 0.018, 34.0, 0.135, 0.095, 0.36),
        (0.18, 0.016, 42.0, 0.115, 0.060, 0.40),
        (0.11, 0.020, 58.0, 0.155, 0.120, 0.18),
        (0.17, 0.012, 16.0, 0.075, 0.145, 0.46),
    ]
    base = templates[action_id % len(templates)]
    jitter = np.asarray(
        [
            rng.normal(0.0, 0.012),
            rng.normal(0.0, 0.002),
            rng.normal(0.0, 3.0),
            rng.normal(0.0, 0.012),
            rng.normal(0.0, 0.018),
            rng.normal(0.0, 0.035),
            rng.uniform(460.0, 840.0),
            rng.uniform(-0.16, 0.16),
        ],
        dtype=np.float32,
    )
    values = np.asarray([*base, 0.0, 0.0], dtype=np.float32)
    context = values + jitter
    context[0] = np.clip(context[0], 0.08, 0.22)
    context[1] = np.clip(context[1], 0.007, 0.026)
    context[2] = np.clip(context[2], 10.0, 66.0)
    context[3] = np.clip(context[3], 0.035, 0.18)
    context[4] = np.clip(context[4], 0.025, 0.18)
    context[5] = np.clip(context[5], 0.12, 0.54)
    return context.astype(np.float32)


def synthetic_wild_sequence(
    theta: np.ndarray,
    family_id: int,
    nuisance: np.ndarray,
    context: np.ndarray,
    frames: int,
    severity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    rho, phi_deg, delta_deg, cohesion_kpa = [float(v) for v in theta]
    grain_size, moisture, crushability, irregularity, plasticity, brittleness = [float(v) for v in nuisance]
    blade_width, blade_thickness, rake_angle, target_depth, target_speed, drag_distance, camera_scale, phase = [
        float(v) for v in context
    ]
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    progress = np.clip((t + phase) / max(1.0 + abs(phase), 1.0e-6), 0.0, 1.0)
    action_wave = 1.0 + 0.10 * np.sin(2.0 * math.pi * progress + 0.5 * family_id)
    depth = target_depth * np.sin(0.5 * math.pi * progress) ** (1.05 + 0.35 * plasticity)
    if rake_angle > 50.0:
        depth *= 1.0 + 0.22 * np.sin(6.0 * math.pi * progress) ** 2
    speed = target_speed * (0.55 + 0.55 * np.sin(math.pi * progress) ** 2) * action_wave
    accel = np.gradient(speed, edge_order=1).astype(np.float32) * frames
    rake_rad = math.radians(rake_angle)
    phi_rad = math.radians(phi_deg)
    delta_rad = math.radians(delta_deg)

    normal_stress = rho * 9.81 * np.maximum(depth, 0.002)
    shear_strength = cohesion_kpa * 1000.0 + normal_stress * math.tan(phi_rad)
    interface_strength = normal_stress * math.tan(delta_rad)
    contact_area = blade_width * (depth + blade_thickness)
    dynamic_gain = 1.0 + (1.5 + 0.9 * grain_size) * speed + 0.10 * np.abs(accel)

    family_force_gain = [1.28, 1.00, 1.15, 0.68][family_id]
    family_flow_gain = [0.64, 1.10, 0.58, 0.80][family_id]
    base_force = contact_area * (0.58 * shear_strength + 0.42 * interface_strength) * dynamic_gain * family_force_gain
    base_force *= 0.0175

    slip = event_pulses(frames, count=2 + family_id, width=0.025 + 0.015 * severity, rng=rng)
    chatter = np.sin((18.0 + 19.0 * grain_size + 12.0 * brittleness) * math.pi * progress + rng.uniform(0, math.pi))
    crush_drop = 1.0 - (0.18 + 0.42 * crushability) * event_pulses(frames, count=3, width=0.018, rng=rng)
    crush_drop = np.clip(crush_drop, 0.22, 1.18)
    if family_id == 3:
        base_force = base_force * crush_drop + base_force.max(initial=1.0) * 0.10 * slip
    elif family_id == 0:
        base_force = base_force * (1.0 + 0.24 * irregularity * chatter + 0.18 * slip)
    elif family_id == 2:
        base_force = base_force * (1.0 + 0.18 * moisture * progress + 0.12 * plasticity)
    else:
        base_force = base_force * (1.0 + 0.04 * chatter)

    fx = base_force * math.cos(rake_rad) * (0.75 + 0.50 * progress)
    fz = base_force * math.sin(rake_rad) + 0.003 * rho * depth
    fy = 0.10 * fx * np.sin(2.0 * math.pi * progress + phase) + 0.03 * base_force * chatter
    tx = 0.014 * fz * blade_width * (1.0 + grain_size)
    ty = 0.021 * fx * (depth + blade_thickness) * (1.0 + 0.5 * plasticity)
    tz = 0.010 * fy * drag_distance
    wrench = np.stack([fx, fy, fz, tx, ty, tz], axis=1).astype(np.float32)

    gain_shift = rng.normal(1.0, 0.08 * severity, size=(1, 6)).astype(np.float32)
    noise = rng.normal(0.0, 0.018 + 0.065 * severity, size=wrench.shape).astype(np.float32)
    wrench = wrench * gain_shift * (1.0 + noise)
    if severity > 0.45:
        dropout_mask = rng.random(size=wrench.shape) < (0.010 + 0.025 * severity)
        wrench = np.where(dropout_mask, 0.0, wrench).astype(np.float32)

    force_norm = np.linalg.norm(wrench[:, :3], axis=1).astype(np.float32)
    torque_norm = np.linalg.norm(wrench[:, 3:6], axis=1).astype(np.float32)
    force_rms, force_slope, force_highfreq_ratio, force_spectral_centroid = rolling_force_descriptors(force_norm)
    q_base = np.stack(
        [
            0.33 * np.sin(math.pi * progress + 0.2 * family_id),
            -0.20 + 0.62 * depth / max(target_depth, 1.0e-6),
            0.15 * np.cos(math.pi * progress),
            -1.05 + 0.24 * np.sin(2.0 * math.pi * progress),
            0.30 * np.sin(progress + rake_rad),
            1.03 - 0.42 * progress,
            0.08 * np.cos(2.0 * math.pi * progress + family_id),
        ],
        axis=1,
    ).astype(np.float32)
    q = q_base + rng.normal(0.0, 0.008 + 0.010 * severity, size=q_base.shape).astype(np.float32)
    qdot = np.gradient(q, axis=0, edge_order=1).astype(np.float32) * frames
    work = np.cumsum(force_norm * speed / frames).astype(np.float32)
    sensor = np.concatenate(
        [
            wrench,
            q,
            qdot,
            depth[:, None],
            speed[:, None],
            accel[:, None],
            np.full((frames, 1), rake_angle, dtype=np.float32),
            work[:, None],
            force_norm[:, None],
            torque_norm[:, None],
            force_rms[:, None],
            force_slope[:, None],
            force_highfreq_ratio[:, None],
            force_spectral_centroid[:, None],
        ],
        axis=1,
    )

    deformation = (depth / max(target_depth, 1.0e-6)) * (0.45 + 0.55 * np.tanh((cohesion_kpa + 4.0 * plasticity) / 10.0))
    pile_height = depth * (0.12 + 0.025 * phi_deg + 0.12 * plasticity) + 0.0045 * cohesion_kpa
    pile_front = drag_distance * progress * family_flow_gain * (0.62 + 0.004 * phi_deg + 0.004 * delta_deg)
    pile_front -= 0.05 * np.tanh((cohesion_kpa + 8.0 * plasticity) / 14.0)
    flow_mean = speed * family_flow_gain * (0.54 + 0.017 * delta_deg + 0.006 * (phi_deg - 30.0))
    flow_mean /= 1.0 + 0.028 * cohesion_kpa + 0.65 * plasticity
    flow_peak = flow_mean * (1.22 + 0.16 * grain_size + 0.28 * brittleness + 0.08 * np.sin(2.0 * math.pi * progress))
    sand_area = 0.50 + 0.26 * deformation + 0.10 * grain_size - 0.16 * crushability * progress
    tool_area = blade_width * (0.48 + 3.3 * depth) * camera_scale * 1.0e-3
    visible_depth = depth * camera_scale * (0.78 + 0.20 * math.cos(rake_rad))
    shadow_free_luma = np.full(frames, 0.46 + 0.00022 * rho - 0.006 * cohesion_kpa, dtype=np.float32)
    shadow_free_luma += 0.04 * np.sin(4.0 * math.pi * progress + family_id)
    surface_slope = progress * (0.16 + 0.018 * phi_deg + 0.10 * plasticity) + 0.020 * np.sin(math.pi * progress)
    contact_edge_density = 0.10 + 0.009 * delta_deg + 0.20 * deformation + 0.23 * grain_size + 0.14 * brittleness
    surface_curvature = np.gradient(np.gradient(pile_height, edge_order=1), edge_order=1).astype(np.float32) * frames
    surface_curvature += 0.010 * grain_size * chatter + 0.016 * crushability * slip
    vision = np.stack(
        [
            tool_area,
            sand_area,
            flow_mean,
            flow_peak,
            deformation,
            pile_height,
            pile_front,
            surface_slope,
            shadow_free_luma,
            contact_edge_density,
            surface_curvature,
            visible_depth,
        ],
        axis=1,
    ).astype(np.float32)
    vision += rng.normal(0.0, 0.010 + 0.040 * severity, size=vision.shape).astype(np.float32)
    if severity > 0.35:
        occlusion = rng.random(size=(frames, 1)) < (0.05 + 0.18 * severity)
        vision = np.where(occlusion, vision * rng.uniform(0.35, 0.75), vision).astype(np.float32)
        vision[:, [0, 11]] *= rng.normal(1.0, 0.08 * severity)
    return sensor.astype(np.float32), vision.astype(np.float32)


def event_pulses(frames: int, count: int, width: float, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    pulses = np.zeros(frames, dtype=np.float32)
    for _ in range(max(1, count)):
        center = rng.uniform(0.12, 0.94)
        amp = rng.uniform(0.35, 1.00)
        pulses += amp * np.exp(-0.5 * ((t - center) / max(width, 1.0e-3)) ** 2).astype(np.float32)
    if pulses.max(initial=0.0) > 1.0e-6:
        pulses /= pulses.max()
    return pulses.astype(np.float32)


def train_wild_material_model(
    output_dir: Path,
    config: dict[str, Any],
    quick: bool = False,
) -> dict[str, Any]:
    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is required. Install with: pip install -e '.[learning]'")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_wild_material_dataset(config, quick=quick)
    write_wild_dataset(output_dir / "wild_material_windows.npz", dataset)

    seed = int(config.get("seed", 101))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    split = np.asarray(dataset["split"], dtype=np.int32)
    train_idx = np.where(split == 0)[0]
    val_idx = np.where(split == 1)[0]
    test_idx = np.where(split == 2)[0]
    if train_idx.size == 0:
        raise RuntimeError("wild material dataset has no training samples")

    sensor, sensor_stats = normalize_feature_tensor(dataset["sensor"], train_idx)
    vision, vision_stats = normalize_feature_tensor(dataset["vision"], train_idx)
    context, context_stats = normalize_feature_matrix(dataset["context"], train_idx)
    target, target_stats = normalize_feature_matrix(dataset["target"], train_idx)
    family = np.asarray(dataset["family"], dtype=np.int64)
    modality_ablation = str(config.get("modality_ablation", "all"))
    if modality_ablation == "sensor_only":
        vision = np.zeros_like(vision)
    elif modality_ablation == "vision_only":
        sensor = np.zeros_like(sensor)
    elif modality_ablation == "context_only":
        sensor = np.zeros_like(sensor)
        vision = np.zeros_like(vision)
    elif modality_ablation != "all":
        raise ValueError(f"unknown modality_ablation: {modality_ablation}")

    tensors = {
        "sensor": torch.from_numpy(sensor).to(device),
        "vision": torch.from_numpy(vision).to(device),
        "context": torch.from_numpy(context).to(device),
        "target": torch.from_numpy(target).to(device),
        "family": torch.from_numpy(family).to(device),
    }

    model = WildMaterialBeliefNet(
        sensor_dim=sensor.shape[-1],
        vision_dim=vision.shape[-1],
        context_dim=context.shape[-1],
        target_dim=target.shape[-1],
        family_count=len(WILD_FAMILY_NAMES),
        hidden_dim=int(config.get("hidden_dim", 128)),
        token_dim=int(config.get("token_dim", 128)),
        dropout=float(config.get("dropout", 0.08)),
        use_property_family_head=bool(config.get("use_property_family_head", True)),
    ).to(device)

    epochs = int(config.get("quick_epochs" if quick else "epochs", 45 if quick else 90))
    batch_size = int(config.get("quick_batch_size" if quick else "batch_size", 24 if quick else 48))
    family_loss_weight = float(config.get("family_loss_weight", 0.85))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1.2e-3)),
        weight_decay=float(config.get("weight_decay", 2.0e-4)),
    )
    rng = np.random.default_rng(seed + 41)
    losses: list[float] = []
    val_losses: list[float] = []
    best_validation_loss = math.inf
    best_epoch = -1
    best_model_state: dict[str, Any] | None = None
    started = time.perf_counter()

    for epoch in range(epochs):
        epoch_losses: list[float] = []
        for batch in batch_indices(train_idx, batch_size, rng):
            output = model(tensors["sensor"][batch], tensors["vision"][batch], tensors["context"][batch])
            loss = wild_training_loss(output, tensors["target"][batch], tensors["family"][batch], family_loss_weight)
            loss = loss + 0.003 * output["update_norm"].mean()
            loss = loss + 0.001 * modality_balance_loss(output["modality_gate"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else math.nan)
        if val_idx.size:
            val_loss = evaluate_wild_loss(
                model,
                tensors["sensor"][val_idx],
                tensors["vision"][val_idx],
                tensors["context"][val_idx],
                tensors["target"][val_idx],
                tensors["family"][val_idx],
                family_loss_weight,
            )
            val_losses.append(val_loss)
            if np.isfinite(val_loss) and val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_epoch = epoch
                best_model_state = snapshot_model_state(model)

    restored_best_validation = False
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        restored_best_validation = True

    calibration_scale = np.ones(len(TARGET_NAMES), dtype=np.float32)
    calibrate_sigma = bool(config.get("calibrate_sigma", True))
    if calibrate_sigma and val_idx.size:
        with torch.no_grad():
            val_output = model(tensors["sensor"][val_idx], tensors["vision"][val_idx], tensors["context"][val_idx])
        val_rollout = denormalize_wild_rollout(val_output, target_stats)
        calibration_scale = estimate_sigma_calibration_scale(
            mu=val_rollout["mu"][:, -1, :],
            sigma=val_rollout["sigma"][:, -1, :],
            target=np.asarray(dataset["target"], dtype=np.float32)[val_idx],
        )

    eval_idx = test_idx if test_idx.size else (val_idx if val_idx.size else train_idx)
    with torch.no_grad():
        eval_output = model(tensors["sensor"][eval_idx], tensors["vision"][eval_idx], tensors["context"][eval_idx])
    rollout = denormalize_wild_rollout(eval_output, target_stats, sigma_scale=calibration_scale)
    eval_dataset = {
        "target": np.asarray(dataset["target"], dtype=np.float32)[eval_idx],
        "family": family[eval_idx],
        "tag_index": np.asarray(dataset["tag_index"], dtype=np.int32)[eval_idx],
        "corruption_severity": np.asarray(dataset["corruption_severity"], dtype=np.float32)[eval_idx],
    }
    metrics = wild_rollout_metrics(rollout, eval_dataset, config)
    eval_predictions_npz = output_dir / "wild_eval_predictions.npz"
    write_wild_eval_predictions(eval_predictions_npz, rollout, eval_dataset)
    selected_row = select_hard_wild_row(rollout, eval_dataset)
    predictions_csv = output_dir / "wild_material_predictions.csv"
    write_wild_predictions_csv(predictions_csv, rollout, eval_dataset, selected_row)
    rollout_preview = output_dir / "wild_rollout_preview.png"
    draw_wild_rollout_preview(rollout_preview, rollout, eval_dataset, selected_row)
    summary_figure = output_dir / "wild_robustness_summary.png"
    draw_wild_summary_figure(summary_figure, metrics)

    train_payload = {
        "status": "ok",
        "architecture": "WildMaterialBeliefNet",
        "belief_update": "precision_weighted_online_filter",
        "family_names": WILD_FAMILY_NAMES,
        "target_names": TARGET_NAMES,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "family_loss_weight": family_loss_weight,
        "modality_ablation": modality_ablation,
        "use_property_family_head": bool(config.get("use_property_family_head", True)),
        "calibrate_sigma": calibrate_sigma,
        "restored_best_validation": restored_best_validation,
        "best_validation_epoch": int(best_epoch) if best_epoch >= 0 else None,
        "best_validation_loss": clean_float(best_validation_loss) if np.isfinite(best_validation_loss) else None,
        "train_loss": clean_float_list(losses),
        "validation_loss": clean_float_list(val_losses),
        "train_loss_final": clean_float(losses[-1]) if losses else None,
        "validation_loss_final": clean_float(val_losses[-1]) if val_losses else None,
        "elapsed_sec": float(time.perf_counter() - started),
        "posterior_sigma_calibration_scale": {
            TARGET_NAMES[i]: float(calibration_scale[i]) for i in range(len(TARGET_NAMES))
        },
        "normalization": {
            "sensor": sensor_stats,
            "vision": vision_stats,
            "context": context_stats,
            "target": target_stats,
        },
    }
    eval_payload = {
        "status": "ok",
        "eval_count": int(eval_idx.size),
        "eval_split": "test" if test_idx.size else ("validation" if val_idx.size else "train"),
        "selected_eval_row": int(selected_row),
        "predictions_csv": predictions_csv.as_posix(),
        "eval_predictions": eval_predictions_npz.as_posix(),
        "rollout_preview": rollout_preview.as_posix(),
        "summary_figure": summary_figure.as_posix(),
        **metrics,
    }
    model_payload = {
        "model_state": model.state_dict(),
        "config": config,
        "target_stats": target_stats,
        "sensor_stats": sensor_stats,
        "vision_stats": vision_stats,
        "context_stats": context_stats,
        "posterior_sigma_calibration_scale": calibration_scale.tolist(),
        "metadata": dataset["metadata"],
    }
    torch.save(model_payload, output_dir / "wild_material_model.pt")
    write_json(output_dir / "wild_training_metrics.json", train_payload)
    write_json(output_dir / "wild_eval_metrics.json", eval_payload)
    write_json(
        output_dir / "manifest.json",
        {
            "status": "ok",
            "dataset": (output_dir / "wild_material_windows.npz").as_posix(),
            "training_metrics": (output_dir / "wild_training_metrics.json").as_posix(),
            "eval_metrics": (output_dir / "wild_eval_metrics.json").as_posix(),
            "eval_predictions": (output_dir / "wild_eval_predictions.npz").as_posix(),
            "checkpoint": (output_dir / "wild_material_model.pt").as_posix(),
            "metadata": dataset["metadata"],
        },
    )
    return {"training": train_payload, "evaluation": eval_payload}


def write_wild_dataset(path: Path, dataset: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sensor=dataset["sensor"],
        vision=dataset["vision"],
        context=dataset["context"],
        target=dataset["target"],
        family=dataset["family"],
        split=dataset["split"],
        material_index=dataset["material_index"],
        action_index=dataset["action_index"],
        tag_index=dataset["tag_index"],
        corruption_severity=dataset["corruption_severity"],
        metadata=np.asarray(str(dataset["metadata"])),
    )


def wild_training_loss(output: dict[str, Any], target: Any, family: Any, family_loss_weight: float) -> Any:
    regression = online_gaussian_nll(output["mu"], output["log_sigma"], target)
    classification = sequence_family_loss(output["family_logits"], family)
    return regression + family_loss_weight * classification


def sequence_family_loss(logits: Any, family: Any) -> Any:
    batch, frames, family_count = logits.shape
    y = family[:, None].repeat(1, frames).reshape(-1)
    ce = F.cross_entropy(logits.reshape(batch * frames, family_count), y, reduction="none", label_smoothing=0.035).reshape(batch, frames)
    weights = torch.linspace(0.35, 1.0, frames, device=logits.device, dtype=logits.dtype)
    weighted = (ce * weights[None, :]).sum() / (weights.sum() * batch)
    final = F.cross_entropy(logits[:, -1], family, label_smoothing=0.035)
    return weighted + 0.9 * final


def evaluate_wild_loss(
    model: Any,
    sensor: Any,
    vision: Any,
    context: Any,
    target: Any,
    family: Any,
    family_loss_weight: float,
) -> float:
    model.eval()
    with torch.no_grad():
        output = model(sensor, vision, context)
        loss = wild_training_loss(output, target, family, family_loss_weight)
    model.train()
    return float(loss.detach().cpu())


def denormalize_wild_rollout(
    output: dict[str, Any],
    target_stats: dict[str, Any],
    sigma_scale: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    mean = np.asarray(target_stats["mean"], dtype=np.float32)
    std = np.asarray(target_stats["std"], dtype=np.float32)
    scale = np.ones_like(std, dtype=np.float32) if sigma_scale is None else np.asarray(sigma_scale, dtype=np.float32)
    mu_norm = output["mu"].detach().cpu().numpy().astype(np.float32)
    sigma_norm = np.exp(output["log_sigma"].detach().cpu().numpy().astype(np.float32))
    logits = output["family_logits"].detach().cpu().numpy().astype(np.float32)
    logits = logits - logits.max(axis=-1, keepdims=True)
    prob = np.exp(logits)
    prob = prob / np.maximum(prob.sum(axis=-1, keepdims=True), 1.0e-8)
    return {
        "mu": mu_norm * std[None, None, :] + mean[None, None, :],
        "sigma": sigma_norm * std[None, None, :] * scale[None, None, :],
        "family_prob": prob.astype(np.float32),
        "family_pred": np.argmax(prob, axis=-1).astype(np.int64),
        "precision": output["precision"].detach().cpu().numpy().astype(np.float32),
        "update_norm": output["update_norm"].detach().cpu().numpy().astype(np.float32),
    }


def estimate_sigma_calibration_scale(mu: np.ndarray, sigma: np.ndarray, target: np.ndarray) -> np.ndarray:
    err = np.abs(np.asarray(mu, dtype=np.float32) - np.asarray(target, dtype=np.float32))
    sig = np.maximum(np.asarray(sigma, dtype=np.float32), 1.0e-6)
    z = err / sig
    q68 = np.quantile(z, 0.6827, axis=0)
    q95 = np.quantile(z, 0.9545, axis=0) / 2.0
    rms = np.sqrt(np.mean(z * z, axis=0))
    scale = 0.40 * q68 + 0.35 * q95 + 0.25 * rms
    return np.clip(scale.astype(np.float32), 0.75, 12.0)


def wild_rollout_metrics(rollout: dict[str, np.ndarray], eval_dataset: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    target = np.asarray(eval_dataset["target"], dtype=np.float32)
    family = np.asarray(eval_dataset["family"], dtype=np.int64)
    tag_index = np.asarray(eval_dataset["tag_index"], dtype=np.int32)
    mu = np.asarray(rollout["mu"], dtype=np.float32)
    sigma = np.maximum(np.asarray(rollout["sigma"], dtype=np.float32), 1.0e-6)
    family_pred = np.asarray(rollout["family_pred"], dtype=np.int64)[:, -1]
    family_prob = np.asarray(rollout["family_prob"], dtype=np.float32)[:, -1]
    final_abs = np.abs(mu[:, -1, :] - target)
    ranges = np.maximum(target.max(axis=0) - target.min(axis=0), 1.0e-6)
    final_nmae = final_abs / ranges[None, :]
    final_error = mu[:, -1, :] - target
    final_sigma = sigma[:, -1, :]
    final_z = np.abs(final_error) / final_sigma
    final_nll = 0.5 * ((final_error / final_sigma) ** 2 + 2.0 * np.log(final_sigma) + math.log(2.0 * math.pi))
    confusion = confusion_matrix(family, family_pred, len(WILD_FAMILY_NAMES))
    accuracy = float(np.mean(family_pred == family))
    by_family: dict[str, Any] = {}
    for family_id, family_name in enumerate(WILD_FAMILY_NAMES):
        mask = family == family_id
        by_family[family_name] = group_metric_payload(mask, family, family_pred, final_abs, final_nmae)
    by_tag: dict[str, Any] = {}
    for tag_id, tag_name in enumerate(WILD_TAG_NAMES):
        mask = tag_index == tag_id
        if np.any(mask):
            by_tag[tag_name] = group_metric_payload(mask, family, family_pred, final_abs, final_nmae)

    thresholds = dict(config.get("robustness_thresholds", {}))
    min_accuracy = float(thresholds.get("min_family_accuracy", 0.82))
    min_worst_family_accuracy = float(thresholds.get("min_worst_family_accuracy", 0.70))
    max_nmae = float(thresholds.get("max_property_nmae_mean", 0.18))
    max_coverage_error = float(thresholds.get("max_coverage_error_mean", 0.24))
    family_acc_values = [float(v["family_accuracy"]) for v in by_family.values() if int(v["count"]) > 0]
    worst_family_accuracy = min(family_acc_values) if family_acc_values else 0.0
    coverage_1sigma = np.mean(final_z <= 1.0, axis=0)
    coverage_2sigma = np.mean(final_z <= 2.0, axis=0)
    coverage_error = float((np.abs(coverage_1sigma - 0.6827).mean() + np.abs(coverage_2sigma - 0.9545).mean()) * 0.5)
    paper_ready = bool(
        accuracy >= min_accuracy
        and worst_family_accuracy >= min_worst_family_accuracy
        and float(final_nmae.mean()) <= max_nmae
        and coverage_error <= max_coverage_error
    )
    return {
        "family_accuracy": accuracy,
        "worst_family_accuracy": float(worst_family_accuracy),
        "family_confusion": confusion.tolist(),
        "family_confusion_labels": WILD_FAMILY_NAMES,
        "mean_family_confidence": float(np.max(family_prob, axis=1).mean()),
        "final_frame_mae": {TARGET_NAMES[i]: float(final_abs[:, i].mean()) for i in range(len(TARGET_NAMES))},
        "final_frame_mae_mean": float(final_abs.mean()),
        "final_frame_nmae": {TARGET_NAMES[i]: float(final_nmae[:, i].mean()) for i in range(len(TARGET_NAMES))},
        "final_frame_nmae_mean": float(final_nmae.mean()),
        "final_nll": {TARGET_NAMES[i]: float(final_nll[:, i].mean()) for i in range(len(TARGET_NAMES))},
        "final_nll_mean": float(final_nll.mean()),
        "final_coverage_1sigma": {TARGET_NAMES[i]: float(coverage_1sigma[i]) for i in range(len(TARGET_NAMES))},
        "final_coverage_2sigma": {TARGET_NAMES[i]: float(coverage_2sigma[i]) for i in range(len(TARGET_NAMES))},
        "coverage_error_mean": coverage_error,
        "by_family": by_family,
        "by_eval_tag": by_tag,
        "mean_corruption_severity": float(np.asarray(eval_dataset["corruption_severity"], dtype=np.float32).mean()),
        "robustness_thresholds": {
            "min_family_accuracy": min_accuracy,
            "min_worst_family_accuracy": min_worst_family_accuracy,
            "max_property_nmae_mean": max_nmae,
            "max_coverage_error_mean": max_coverage_error,
        },
        "paper_ready": paper_ready,
        "paper_ready_reason": (
            "all robustness thresholds passed"
            if paper_ready
            else "at least one robustness threshold failed; do not claim paper-ready robustness"
        ),
    }


def group_metric_payload(
    mask: np.ndarray,
    family: np.ndarray,
    family_pred: np.ndarray,
    final_abs: np.ndarray,
    final_nmae: np.ndarray,
) -> dict[str, Any]:
    if not np.any(mask):
        return {"count": 0, "family_accuracy": None, "mae_mean": None, "nmae_mean": None}
    return {
        "count": int(np.sum(mask)),
        "family_accuracy": float(np.mean(family_pred[mask] == family[mask])),
        "mae": {TARGET_NAMES[i]: float(final_abs[mask, i].mean()) for i in range(len(TARGET_NAMES))},
        "mae_mean": float(final_abs[mask].mean()),
        "nmae": {TARGET_NAMES[i]: float(final_nmae[mask, i].mean()) for i in range(len(TARGET_NAMES))},
        "nmae_mean": float(final_nmae[mask].mean()),
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, count: int) -> np.ndarray:
    matrix = np.zeros((count, count), dtype=np.int32)
    for true_id, pred_id in zip(y_true, y_pred):
        matrix[int(true_id), int(pred_id)] += 1
    return matrix


def select_hard_wild_row(rollout: dict[str, np.ndarray], eval_dataset: dict[str, np.ndarray]) -> int:
    target = np.asarray(eval_dataset["target"], dtype=np.float32)
    family = np.asarray(eval_dataset["family"], dtype=np.int64)
    mu = np.asarray(rollout["mu"], dtype=np.float32)
    pred = np.asarray(rollout["family_pred"], dtype=np.int64)[:, -1]
    ranges = np.maximum(target.max(axis=0) - target.min(axis=0), 1.0e-6)
    score = np.abs(mu[:, -1, :] - target).mean(axis=1) / float(ranges.mean())
    score = score + 1.5 * (pred != family).astype(np.float32)
    return int(np.argmax(score))


def write_wild_predictions_csv(
    path: Path,
    rollout: dict[str, np.ndarray],
    eval_dataset: dict[str, np.ndarray],
    selected_row: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mu = np.asarray(rollout["mu"], dtype=np.float32)[selected_row]
    sigma = np.asarray(rollout["sigma"], dtype=np.float32)[selected_row]
    prob = np.asarray(rollout["family_prob"], dtype=np.float32)[selected_row]
    target = np.asarray(eval_dataset["target"], dtype=np.float32)[selected_row]
    family = int(np.asarray(eval_dataset["family"], dtype=np.int64)[selected_row])
    fieldnames = ["frame", "target_family", "pred_family"]
    for family_name in WILD_FAMILY_NAMES:
        fieldnames.append(f"prob_{family_name}")
    for name in TARGET_NAMES:
        fieldnames.extend([f"target_{name}", f"pred_{name}", f"sigma_{name}", f"abs_error_{name}"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame_id in range(mu.shape[0]):
            pred_family = int(np.argmax(prob[frame_id]))
            row: dict[str, Any] = {
                "frame": frame_id,
                "target_family": WILD_FAMILY_NAMES[family],
                "pred_family": WILD_FAMILY_NAMES[pred_family],
            }
            for family_id, family_name in enumerate(WILD_FAMILY_NAMES):
                row[f"prob_{family_name}"] = float(prob[frame_id, family_id])
            for target_id, name in enumerate(TARGET_NAMES):
                row[f"target_{name}"] = float(target[target_id])
                row[f"pred_{name}"] = float(mu[frame_id, target_id])
                row[f"sigma_{name}"] = float(sigma[frame_id, target_id])
                row[f"abs_error_{name}"] = float(abs(mu[frame_id, target_id] - target[target_id]))
            writer.writerow(row)


def write_wild_eval_predictions(path: Path, rollout: dict[str, np.ndarray], eval_dataset: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mu=np.asarray(rollout["mu"], dtype=np.float32),
        sigma=np.asarray(rollout["sigma"], dtype=np.float32),
        family_prob=np.asarray(rollout["family_prob"], dtype=np.float32),
        family_pred=np.asarray(rollout["family_pred"], dtype=np.int64),
        target=np.asarray(eval_dataset["target"], dtype=np.float32),
        family=np.asarray(eval_dataset["family"], dtype=np.int64),
        tag_index=np.asarray(eval_dataset["tag_index"], dtype=np.int32),
        corruption_severity=np.asarray(eval_dataset["corruption_severity"], dtype=np.float32),
        family_names=np.asarray(WILD_FAMILY_NAMES),
        tag_names=np.asarray(WILD_TAG_NAMES),
        target_names=np.asarray(TARGET_NAMES),
    )


def draw_wild_summary_figure(path: Path, metrics: dict[str, Any]) -> None:
    width, height = 1800, 1100
    image = np.full((height, width, 3), (248, 248, 246), dtype=np.uint8)
    cv2.putText(image, "Wild material robustness: gravel / sand / soil / crunching", (42, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (28, 32, 34), 2, cv2.LINE_AA)
    cv2.putText(
        image,
        f"accuracy={metrics['family_accuracy']:.3f}  worst-family={metrics['worst_family_accuracy']:.3f}  nMAE={metrics['final_frame_nmae_mean']:.3f}  paper_ready={metrics['paper_ready']}",
        (42, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (45, 52, 54),
        1,
        cv2.LINE_AA,
    )
    draw_confusion_panel(image, np.asarray(metrics["family_confusion"], dtype=np.int32), (52, 150, 560, 560))
    draw_family_bars(image, metrics, (700, 160, 1000, 340), "family accuracy", "family_accuracy", 1.0)
    draw_family_bars(image, metrics, (700, 570, 1000, 340), "property nMAE mean", "nmae_mean", 0.45)
    draw_metric_block(image, metrics, (52, 760, 560, 280))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path.as_posix(), image)


def draw_confusion_panel(image: np.ndarray, matrix: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x0, y0, w, h = rect
    cv2.putText(image, "confusion matrix", (x0, y0 - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 38, 40), 2, cv2.LINE_AA)
    cell = min(w, h) // (len(WILD_FAMILY_NAMES) + 1)
    max_value = max(1, int(matrix.max(initial=1)))
    for row in range(len(WILD_FAMILY_NAMES)):
        cv2.putText(image, WILD_FAMILY_NAMES[row], (x0, y0 + (row + 2) * cell - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 50, 52), 1, cv2.LINE_AA)
        cv2.putText(image, WILD_FAMILY_NAMES[row], (x0 + (row + 1) * cell + 6, y0 + cell - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 50, 52), 1, cv2.LINE_AA)
        for col in range(len(WILD_FAMILY_NAMES)):
            value = int(matrix[row, col])
            intensity = int(240 - 160 * value / max_value)
            color = (intensity, 245 - int(80 * value / max_value), 255)
            x = x0 + (col + 1) * cell
            y = y0 + (row + 1) * cell
            cv2.rectangle(image, (x, y), (x + cell - 6, y + cell - 6), color, -1)
            cv2.rectangle(image, (x, y), (x + cell - 6, y + cell - 6), (130, 138, 140), 1)
            cv2.putText(image, str(value), (x + 18, y + cell // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (25, 30, 32), 2, cv2.LINE_AA)


def draw_family_bars(
    image: np.ndarray,
    metrics: dict[str, Any],
    rect: tuple[int, int, int, int],
    title: str,
    key: str,
    max_value: float,
) -> None:
    x0, y0, w, h = rect
    cv2.putText(image, title, (x0, y0 - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 38, 40), 2, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (184, 190, 190), 1)
    bar_h = 52
    for idx, name in enumerate(WILD_FAMILY_NAMES):
        payload = metrics["by_family"][name]
        value = float(payload[key]) if payload.get(key) is not None else 0.0
        y = y0 + 40 + idx * 72
        x_label = x0 + 24
        cv2.putText(image, name, (x_label, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 45, 46), 1, cv2.LINE_AA)
        x_bar = x0 + 190
        bar_w = int((w - 260) * np.clip(value / max(max_value, 1.0e-6), 0.0, 1.0))
        color = (90, 178, 110) if key == "family_accuracy" else (78, 138, 210)
        cv2.rectangle(image, (x_bar, y), (x_bar + w - 260, y + bar_h), (230, 234, 232), -1)
        cv2.rectangle(image, (x_bar, y), (x_bar + bar_w, y + bar_h), color, -1)
        cv2.putText(image, f"{value:.3f}", (x_bar + w - 238, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 40, 42), 1, cv2.LINE_AA)


def draw_metric_block(image: np.ndarray, metrics: dict[str, Any], rect: tuple[int, int, int, int]) -> None:
    x0, y0, w, h = rect
    cv2.putText(image, "robustness gates", (x0, y0 - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 38, 40), 2, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (184, 190, 190), 1)
    rows = [
        ("family accuracy", metrics["family_accuracy"], metrics["robustness_thresholds"]["min_family_accuracy"], ">="),
        ("worst family acc", metrics["worst_family_accuracy"], metrics["robustness_thresholds"]["min_worst_family_accuracy"], ">="),
        ("property nMAE", metrics["final_frame_nmae_mean"], metrics["robustness_thresholds"]["max_property_nmae_mean"], "<="),
        ("coverage err", metrics["coverage_error_mean"], metrics["robustness_thresholds"]["max_coverage_error_mean"], "<="),
    ]
    y = y0 + 48
    for label, value, threshold, op in rows:
        passed = value >= threshold if op == ">=" else value <= threshold
        color = (60, 150, 82) if passed else (50, 80, 220)
        cv2.putText(image, f"{label:<18} {value:6.3f} {op} {threshold:6.3f}", (x0 + 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
        y += 46
    cv2.putText(image, str(metrics["paper_ready_reason"])[:58], (x0 + 26, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (55, 60, 62), 1, cv2.LINE_AA)


def draw_wild_rollout_preview(
    path: Path,
    rollout: dict[str, np.ndarray],
    eval_dataset: dict[str, np.ndarray],
    selected_row: int,
) -> None:
    width, height = 1400, 900
    image = np.full((height, width, 3), 248, dtype=np.uint8)
    mu = np.asarray(rollout["mu"], dtype=np.float32)[selected_row]
    sigma = np.asarray(rollout["sigma"], dtype=np.float32)[selected_row]
    prob = np.asarray(rollout["family_prob"], dtype=np.float32)[selected_row]
    target = np.asarray(eval_dataset["target"], dtype=np.float32)[selected_row]
    family = int(np.asarray(eval_dataset["family"], dtype=np.int64)[selected_row])
    cv2.putText(
        image,
        f"Hard wild sample rollout: target={WILD_FAMILY_NAMES[family]}",
        (36, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        (28, 32, 34),
        2,
        cv2.LINE_AA,
    )
    panel_h = 130
    colors = [(35, 92, 180), (30, 145, 90), (180, 105, 35), (135, 60, 150)]
    for target_id, name in enumerate(TARGET_NAMES):
        top = 90 + target_id * panel_h
        draw_rollout_panel(image, mu[:, target_id], sigma[:, target_id], float(target[target_id]), name, (70, top, 1260, 96), colors[target_id])
    draw_family_probability_panel(image, prob, family, (70, 650, 1260, 190))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path.as_posix(), image)


def draw_rollout_panel(
    image: np.ndarray,
    values: np.ndarray,
    sigma: np.ndarray,
    target: float,
    title: str,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x0, y0, w, h = rect
    lo = float(min(values.min(), target, (values - sigma).min()))
    hi = float(max(values.max(), target, (values + sigma).max()))
    if abs(hi - lo) < 1.0e-6:
        lo -= 1.0
        hi += 1.0
    cv2.putText(image, title, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (40, 45, 46), 1, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (185, 190, 190), 1)
    ty = plot_y(target, lo, hi, y0 + h - 12, y0 + 10)
    cv2.line(image, (x0 + 8, ty), (x0 + w - 8, ty), (70, 70, 70), 1, cv2.LINE_AA)
    pts = []
    for idx, value in enumerate(values):
        x = x0 + 8 + int(idx * (w - 16) / max(1, values.shape[0] - 1))
        y = plot_y(float(value), lo, hi, y0 + h - 12, y0 + 10)
        pts.append((x, y))
    cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    cv2.putText(image, f"final={values[-1]:.3g}  GT={target:.3g}", (x0 + w - 260, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (45, 50, 52), 1, cv2.LINE_AA)


def draw_family_probability_panel(
    image: np.ndarray,
    prob: np.ndarray,
    family: int,
    rect: tuple[int, int, int, int],
) -> None:
    x0, y0, w, h = rect
    cv2.putText(image, "family posterior", (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (40, 45, 46), 1, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (185, 190, 190), 1)
    family_colors = [(120, 120, 120), (70, 160, 90), (185, 115, 60), (150, 80, 175)]
    for family_id, name in enumerate(WILD_FAMILY_NAMES):
        pts = []
        for idx, value in enumerate(prob[:, family_id]):
            x = x0 + 12 + int(idx * (w - 24) / max(1, prob.shape[0] - 1))
            y = y0 + h - 16 - int(float(value) * (h - 36))
            pts.append((x, y))
        cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, family_colors[family_id], 2, cv2.LINE_AA)
        label_color = family_colors[family_id] if family_id == family else (90, 94, 96)
        cv2.putText(image, name, (x0 + 22 + family_id * 180, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, label_color, 1, cv2.LINE_AA)


def plot_y(value: float, lo: float, hi: float, y0: int, y1: int) -> int:
    return int(y0 - (value - lo) / max(hi - lo, 1.0e-6) * (y0 - y1))
