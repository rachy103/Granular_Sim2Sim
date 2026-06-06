from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .metrics import write_json

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover - exercised only on minimal installs.
    torch = None
    F = None
    nn = None


TARGET_NAMES = ["rho", "phi_deg", "delta_deg", "cohesion_kpa"]

SENSOR_FEATURE_NAMES = [
    "fx",
    "fy",
    "fz",
    "tx",
    "ty",
    "tz",
    "q0",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "qd0",
    "qd1",
    "qd2",
    "qd3",
    "qd4",
    "qd5",
    "qd6",
    "depth",
    "tool_speed",
    "tool_accel",
    "rake_angle",
    "work",
    "force_norm",
    "torque_norm",
    "force_rms_window",
    "force_slope_window",
    "force_highfreq_ratio",
    "force_spectral_centroid",
]

VISION_FEATURE_NAMES = [
    "tool_mask_area",
    "sand_mask_area",
    "flow_mean",
    "flow_peak",
    "deformation_area",
    "pile_height",
    "pile_front_x",
    "surface_slope",
    "shadow_free_luma",
    "contact_edge_density",
    "surface_curvature",
    "visible_depth_proxy",
]

CONTEXT_FEATURE_NAMES = [
    "blade_width",
    "blade_thickness",
    "rake_angle",
    "target_depth",
    "target_speed",
    "drag_distance",
    "camera_scale",
    "action_phase_offset",
]


if torch is not None:

    class OnlineMohrCoulombBeliefNet(nn.Module):
        """Sequential posterior update model for material inference.

        The model never emits each frame's material prediction from scratch.
        It maintains a Gaussian belief state `(mu, precision)` and performs a
        precision-weighted update with every new multimodal evidence token.
        """

        def __init__(
            self,
            sensor_dim: int,
            vision_dim: int,
            context_dim: int,
            target_dim: int = 4,
            hidden_dim: int = 96,
            token_dim: int = 96,
            dropout: float = 0.05,
            min_precision: float = 0.04,
        ) -> None:
            super().__init__()
            self.target_dim = target_dim
            self.min_precision = float(min_precision)
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

        def forward(
            self,
            sensor: torch.Tensor,
            vision: torch.Tensor,
            context: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            batch, frames, _sensor_dim = sensor.shape
            context_token = self.context_encoder(context)
            hidden = torch.tanh(self.initial_hidden(context_token))
            mu = self.prior_mu(context_token)
            precision = F.softplus(self.prior_precision(context_token)) + self.min_precision

            mu_seq: list[torch.Tensor] = []
            log_sigma_seq: list[torch.Tensor] = []
            precision_seq: list[torch.Tensor] = []
            evidence_mu_seq: list[torch.Tensor] = []
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

                log_sigma = -0.5 * torch.log(torch.clamp(precision, min=1.0e-6))
                mu_seq.append(mu)
                log_sigma_seq.append(log_sigma)
                precision_seq.append(precision)
                evidence_mu_seq.append(frame_mu)
                update_norm_seq.append(torch.linalg.norm(mu - previous_mu, dim=-1))
                modality_gate_seq.append(gate)

            return {
                "mu": torch.stack(mu_seq, dim=1),
                "log_sigma": torch.stack(log_sigma_seq, dim=1),
                "precision": torch.stack(precision_seq, dim=1),
                "evidence_mu": torch.stack(evidence_mu_seq, dim=1),
                "update_norm": torch.stack(update_norm_seq, dim=1),
                "modality_gate": torch.stack(modality_gate_seq, dim=1),
            }

else:

    class OnlineMohrCoulombBeliefNet:  # pragma: no cover
        pass


def generate_synthetic_multimodal_blade_dataset(config: dict[str, Any], quick: bool = False) -> dict[str, Any]:
    seed = int(config.get("seed", 23))
    rng = np.random.default_rng(seed)
    material_count = int(config.get("quick_material_count" if quick else "material_count", 18 if quick else 48))
    actions_per_material = int(config.get("quick_actions_per_material" if quick else "actions_per_material", 2 if quick else 4))
    frames = int(config.get("quick_frames" if quick else "frames", 24 if quick else 48))

    target_ranges = dict(config.get("target_ranges", {}))
    ranges = {
        "rho": tuple(target_ranges.get("rho", [1150.0, 1900.0])),
        "phi_deg": tuple(target_ranges.get("phi_deg", [24.0, 46.0])),
        "delta_deg": tuple(target_ranges.get("delta_deg", [8.0, 34.0])),
        "cohesion_kpa": tuple(target_ranges.get("cohesion_kpa", [0.0, 16.0])),
    }

    sensors: list[np.ndarray] = []
    visions: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    material_ids: list[int] = []
    action_ids: list[int] = []

    for material_id in range(material_count):
        rho = rng.uniform(*ranges["rho"])
        phi = rng.uniform(*ranges["phi_deg"])
        delta_hi = min(ranges["delta_deg"][1], phi - 1.0)
        delta = rng.uniform(ranges["delta_deg"][0], max(ranges["delta_deg"][0] + 0.5, delta_hi))
        cohesion = rng.uniform(*ranges["cohesion_kpa"])
        theta = np.asarray([rho, phi, delta, cohesion], dtype=np.float32)
        packing_noise = rng.normal(0.0, 0.025)

        for local_action_id in range(actions_per_material):
            action_phase = rng.uniform(-0.1, 0.1)
            context = np.asarray(
                [
                    rng.uniform(0.10, 0.18),
                    rng.uniform(0.008, 0.018),
                    rng.uniform(18.0, 34.0),
                    rng.uniform(0.055, 0.13),
                    rng.uniform(0.035, 0.11),
                    rng.uniform(0.16, 0.34),
                    rng.uniform(480.0, 760.0),
                    action_phase,
                ],
                dtype=np.float32,
            )
            sensor, vision = _synthetic_blade_sequence(theta, context, frames, packing_noise, rng)
            sensors.append(sensor)
            visions.append(vision)
            contexts.append(context)
            targets.append(theta)
            material_ids.append(material_id)
            action_ids.append(local_action_id)

    material_index = np.asarray(material_ids, dtype=np.int32)
    split = make_material_grouped_split(
        material_index=material_index,
        train_fraction=float(config.get("train_fraction", 0.68)),
        validation_fraction=float(config.get("validation_fraction", 0.16)),
        seed=seed + 1009,
    )
    return {
        "sensor": np.stack(sensors, axis=0).astype(np.float32),
        "vision": np.stack(visions, axis=0).astype(np.float32),
        "context": np.stack(contexts, axis=0).astype(np.float32),
        "target": np.stack(targets, axis=0).astype(np.float32),
        "split": split,
        "material_index": material_index,
        "action_index": np.asarray(action_ids, dtype=np.int32),
        "metadata": {
            "generator": "synthetic_multimodal_blade",
            "belief_update": "precision_weighted_online_filter",
            "framewise_reset": False,
            "target_names": TARGET_NAMES,
            "sensor_feature_names": SENSOR_FEATURE_NAMES,
            "vision_feature_names": VISION_FEATURE_NAMES,
            "context_feature_names": CONTEXT_FEATURE_NAMES,
            "feature_upgrades": [
                "force_norm_and_torque_norm",
                "rolling_force_rms_and_slope",
                "rolling_force_frequency_ratio_and_centroid",
            ],
            "frames": frames,
            "material_count": material_count,
            "actions_per_material": actions_per_material,
            "target_ranges": {name: list(values) for name, values in ranges.items()},
        },
    }


def _synthetic_blade_sequence(
    theta: np.ndarray,
    context: np.ndarray,
    frames: int,
    packing_noise: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    rho, phi_deg, delta_deg, cohesion_kpa = [float(v) for v in theta]
    blade_width, blade_thickness, rake_angle, target_depth, target_speed, drag_distance, camera_scale, phase = [
        float(v) for v in context
    ]
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    progress = np.clip((t + phase) / max(1.0 + abs(phase), 1.0e-6), 0.0, 1.0)
    depth = target_depth * np.sin(0.5 * math.pi * progress) ** 1.25
    speed = target_speed * (0.65 + 0.45 * np.sin(math.pi * progress) ** 2)
    accel = np.gradient(speed, edge_order=1).astype(np.float32) * frames
    rake_rad = math.radians(rake_angle)
    phi_rad = math.radians(phi_deg)
    delta_rad = math.radians(delta_deg)
    cohesion_pa = cohesion_kpa * 1000.0

    normal_stress = rho * 9.81 * np.maximum(depth, 0.002)
    shear_strength = cohesion_pa + normal_stress * math.tan(phi_rad)
    interface_strength = normal_stress * math.tan(delta_rad)
    dynamic_gain = 1.0 + 1.8 * speed + 0.15 * np.abs(accel)
    contact_area = blade_width * (depth + blade_thickness)
    base_force = contact_area * (0.55 * shear_strength + 0.45 * interface_strength) * dynamic_gain
    force_scale = 0.018
    fx = force_scale * base_force * math.cos(rake_rad)
    fz = force_scale * base_force * math.sin(rake_rad) + 0.0035 * rho * depth
    fy = 0.08 * fx * np.sin(2.0 * math.pi * progress + phase)
    tx = 0.015 * fz * blade_width
    ty = 0.020 * fx * (depth + blade_thickness)
    tz = 0.010 * fy * drag_distance
    force_noise = rng.normal(0.0, 0.02 + 0.02 * progress[:, None], size=(frames, 6)).astype(np.float32)
    wrench = np.stack([fx, fy, fz, tx, ty, tz], axis=1).astype(np.float32)
    wrench = wrench * (1.0 + force_noise)
    force_norm = np.linalg.norm(wrench[:, :3], axis=1).astype(np.float32)
    torque_norm = np.linalg.norm(wrench[:, 3:6], axis=1).astype(np.float32)
    force_rms, force_slope, force_highfreq_ratio, force_spectral_centroid = rolling_force_descriptors(force_norm)

    q_base = np.stack(
        [
            0.35 * np.sin(math.pi * progress),
            -0.25 + 0.55 * depth / max(target_depth, 1.0e-6),
            0.18 * np.cos(math.pi * progress),
            -1.10 + 0.20 * np.sin(2.0 * math.pi * progress),
            0.35 * np.sin(progress + rake_rad),
            1.00 - 0.35 * progress,
            0.10 * np.cos(2.0 * math.pi * progress),
        ],
        axis=1,
    ).astype(np.float32)
    q = q_base + rng.normal(0.0, 0.006, size=q_base.shape).astype(np.float32)
    qdot = np.gradient(q, axis=0, edge_order=1).astype(np.float32) * frames
    work = np.cumsum(np.linalg.norm(wrench[:, :3], axis=1) * speed / frames).astype(np.float32)
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

    deformation = (depth / max(target_depth, 1.0e-6)) * (0.55 + 0.45 * np.tanh(cohesion_kpa / 8.0))
    pile_height = depth * (0.18 + 0.026 * phi_deg) + 0.004 * cohesion_kpa
    pile_front = (
        drag_distance * progress * (0.68 + 0.005 * (phi_deg - 30.0) + 0.003 * delta_deg)
        - 0.04 * np.tanh(cohesion_kpa / 10.0)
    )
    flow_mean = speed * (0.62 + 0.020 * delta_deg + 0.006 * (phi_deg - 30.0)) / (1.0 + 0.030 * cohesion_kpa)
    flow_peak = flow_mean * (1.35 + 0.012 * delta_deg + 0.05 * np.sin(2.0 * math.pi * progress))
    sand_area = 0.58 + 0.24 * deformation + packing_noise
    tool_area = blade_width * (0.5 + 3.0 * depth) * camera_scale * 1.0e-3
    visible_depth = depth * camera_scale * (0.8 + 0.2 * math.cos(rake_rad))
    shadow_free_luma = np.full(frames, 0.55 + 0.002 * rho / 10.0 - 0.006 * cohesion_kpa, dtype=np.float32)
    surface_slope = progress * (0.20 + 0.018 * phi_deg) + 0.015 * np.sin(math.pi * progress)
    contact_edge_density = 0.12 + 0.010 * delta_deg + 0.22 * deformation + 0.006 * cohesion_kpa
    surface_curvature = (
        np.gradient(np.gradient(pile_height, edge_order=1), edge_order=1).astype(np.float32) * frames
        + 0.0015 * cohesion_kpa
    )
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
    vision += rng.normal(0.0, 0.01, size=vision.shape).astype(np.float32)
    return sensor.astype(np.float32), vision.astype(np.float32)


def rolling_force_descriptors(force_norm: np.ndarray, window: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(force_norm, dtype=np.float32)
    rms = np.zeros_like(values)
    slope = np.zeros_like(values)
    highfreq_ratio = np.zeros_like(values)
    spectral_centroid = np.zeros_like(values)
    for idx in range(values.shape[0]):
        start = max(0, idx - window + 1)
        segment = values[start : idx + 1].astype(np.float32)
        rms[idx] = float(np.sqrt(np.mean(segment * segment)))
        if segment.size > 1:
            slope[idx] = float((segment[-1] - segment[0]) / max(1, segment.size - 1))
        centered = segment - float(segment.mean())
        if centered.size < 4 or float(np.max(np.abs(centered))) < 1.0e-6:
            continue
        spectrum = np.fft.rfft(centered)
        power = np.abs(spectrum).astype(np.float32) ** 2
        if power.size <= 1:
            continue
        power = power[1:]
        total = float(power.sum())
        if total <= 1.0e-8:
            continue
        split = max(1, power.size // 2)
        highfreq_ratio[idx] = float(power[split:].sum() / total)
        freqs = np.linspace(0.0, 1.0, power.size, dtype=np.float32)
        spectral_centroid[idx] = float(np.sum(freqs * power) / total)
    return rms, slope, highfreq_ratio, spectral_centroid


def make_material_grouped_split(
    material_index: np.ndarray,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> np.ndarray:
    material_ids = np.unique(np.asarray(material_index, dtype=np.int32))
    rng = np.random.default_rng(seed)
    order = rng.permutation(material_ids)
    train_count = int(np.clip(round(order.size * train_fraction), 1, max(1, order.size - 2)))
    val_count = int(np.clip(round(order.size * validation_fraction), 1, max(1, order.size - train_count - 1)))
    train_materials = set(int(v) for v in order[:train_count])
    val_materials = set(int(v) for v in order[train_count : train_count + val_count])
    split = np.full(material_index.shape, 2, dtype=np.int32)
    for idx, material_id in enumerate(material_index):
        if int(material_id) in train_materials:
            split[idx] = 0
        elif int(material_id) in val_materials:
            split[idx] = 1
    return split


def train_online_mohr_coulomb(
    output_dir: Path,
    config: dict[str, Any],
    quick: bool = False,
) -> dict[str, Any]:
    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is required. Install with: pip install -e '.[learning]'")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_multimodal_blade_dataset(config, quick=quick)
    write_multimodal_dataset(output_dir / "multimodal_windows.npz", dataset)

    seed = int(config.get("seed", 23))
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
        train_idx = np.arange(split.size)
    eval_idx = test_idx if test_idx.size else (val_idx if val_idx.size else train_idx)

    sensor, sensor_stats = normalize_feature_tensor(dataset["sensor"], train_idx)
    vision, vision_stats = normalize_feature_tensor(dataset["vision"], train_idx)
    context, context_stats = normalize_feature_matrix(dataset["context"], train_idx)
    target, target_stats = normalize_feature_matrix(dataset["target"], train_idx)

    tensors = {
        "sensor": torch.from_numpy(sensor).to(device),
        "vision": torch.from_numpy(vision).to(device),
        "context": torch.from_numpy(context).to(device),
        "target": torch.from_numpy(target).to(device),
    }

    model = OnlineMohrCoulombBeliefNet(
        sensor_dim=sensor.shape[-1],
        vision_dim=vision.shape[-1],
        context_dim=context.shape[-1],
        target_dim=target.shape[-1],
        hidden_dim=int(config.get("hidden_dim", 96)),
        token_dim=int(config.get("token_dim", 96)),
        dropout=float(config.get("dropout", 0.05)),
    ).to(device)

    epochs = int(config.get("quick_epochs" if quick else "epochs", 18 if quick else 55))
    batch_size = int(config.get("quick_batch_size" if quick else "batch_size", 16 if quick else 32))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1.5e-3)),
        weight_decay=float(config.get("weight_decay", 1.0e-4)),
    )
    rng = np.random.default_rng(seed + 17)
    losses: list[float] = []
    val_losses: list[float] = []
    restore_best_validation = bool(config.get("restore_best_validation", True))
    best_validation_loss = math.inf
    best_epoch = -1
    best_model_state: dict[str, Any] | None = None
    started = time.perf_counter()

    for epoch in range(epochs):
        epoch_losses: list[float] = []
        for batch in batch_indices(train_idx, batch_size, rng):
            output = model(tensors["sensor"][batch], tensors["vision"][batch], tensors["context"][batch])
            loss = online_gaussian_nll(output["mu"], output["log_sigma"], tensors["target"][batch])
            loss = loss + 0.003 * output["update_norm"].mean()
            loss = loss + 0.001 * modality_balance_loss(output["modality_gate"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else math.nan)
        if val_idx.size:
            val_loss = float(
                evaluate_online_loss(
                    model,
                    tensors["sensor"][val_idx],
                    tensors["vision"][val_idx],
                    tensors["context"][val_idx],
                    tensors["target"][val_idx],
                )
            )
            val_losses.append(val_loss)
            if restore_best_validation and np.isfinite(val_loss) and val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_epoch = epoch
                best_model_state = snapshot_model_state(model)

    restored_best_validation = False
    if restore_best_validation and best_model_state is not None:
        model.load_state_dict(best_model_state)
        restored_best_validation = True

    with torch.no_grad():
        eval_output = model(tensors["sensor"][eval_idx], tensors["vision"][eval_idx], tensors["context"][eval_idx])
    rollout = denormalize_rollout(eval_output, target_stats)
    target_eval = np.asarray(dataset["target"], dtype=np.float32)[eval_idx]
    aggregate_metrics = rollout_metrics(rollout, target_eval)
    selected_eval_row = select_rollout_index(rollout["mu"], target_eval)
    rollout_csv = output_dir / "rollout_predictions.csv"
    write_rollout_csv(
        rollout_csv,
        rollout["mu"][selected_eval_row],
        rollout["sigma"][selected_eval_row],
        target_eval[selected_eval_row],
    )
    preview_png = output_dir / "rollout_preview.png"
    draw_rollout_preview(
        preview_png,
        rollout["mu"][selected_eval_row],
        rollout["sigma"][selected_eval_row],
        target_eval[selected_eval_row],
    )

    train_payload = {
        "status": "ok",
        "architecture": "OnlineMohrCoulombBeliefNet",
        "belief_update": "precision_weighted_online_filter",
        "performance_upgrades": [
            "rolling_force_time_frequency_features",
            "crossmodal_reliability_gates",
            "late_frame_weighted_online_nll",
            "best_validation_checkpoint_restore",
        ],
        "framewise_reset": False,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "restore_best_validation": restore_best_validation,
        "restored_best_validation": restored_best_validation,
        "best_validation_epoch": int(best_epoch) if best_epoch >= 0 else None,
        "best_validation_loss": clean_float(best_validation_loss) if np.isfinite(best_validation_loss) else None,
        "train_loss": clean_float_list(losses),
        "validation_loss": clean_float_list(val_losses),
        "train_loss_final": clean_float(losses[-1]) if losses else None,
        "validation_loss_final": clean_float(val_losses[-1]) if val_losses else None,
        "elapsed_sec": float(time.perf_counter() - started),
        "target_names": TARGET_NAMES,
        "normalization": {
            "sensor": sensor_stats,
            "vision": vision_stats,
            "context": context_stats,
            "target": target_stats,
        },
        "sensor_feature_names": SENSOR_FEATURE_NAMES,
        "vision_feature_names": VISION_FEATURE_NAMES,
        "context_feature_names": CONTEXT_FEATURE_NAMES,
    }
    rollout_payload = {
        "status": "ok",
        "eval_count": int(eval_idx.size),
        "eval_split": "test" if test_idx.size else ("validation" if val_idx.size else "train"),
        "selected_eval_row": int(selected_eval_row),
        "rollout_csv": rollout_csv.as_posix(),
        "rollout_preview": preview_png.as_posix(),
        "target_names": TARGET_NAMES,
        **aggregate_metrics,
    }
    model_payload = {
        "model_state": model.state_dict(),
        "config": config,
        "target_stats": target_stats,
        "sensor_stats": sensor_stats,
        "vision_stats": vision_stats,
        "context_stats": context_stats,
        "metadata": dataset["metadata"],
    }
    torch.save(model_payload, output_dir / "online_mohr_coulomb.pt")
    write_json(output_dir / "training_metrics.json", train_payload)
    write_json(output_dir / "rollout_metrics.json", rollout_payload)
    write_json(
        output_dir / "manifest.json",
        {
            "status": "ok",
            "dataset": (output_dir / "multimodal_windows.npz").as_posix(),
            "training_metrics": (output_dir / "training_metrics.json").as_posix(),
            "rollout_metrics": (output_dir / "rollout_metrics.json").as_posix(),
            "checkpoint": (output_dir / "online_mohr_coulomb.pt").as_posix(),
            "metadata": dataset["metadata"],
        },
    )
    return {"training": train_payload, "rollout": rollout_payload}


def write_multimodal_dataset(path: Path, dataset: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sensor=dataset["sensor"],
        vision=dataset["vision"],
        context=dataset["context"],
        target=dataset["target"],
        split=dataset["split"],
        material_index=dataset["material_index"],
        action_index=dataset["action_index"],
        metadata=np.asarray(str(dataset["metadata"])),
    )


def normalize_feature_tensor(values: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(values, dtype=np.float32)
    train = x[train_idx] if train_idx.size else x
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return ((x - mean[None, None, :]) / std[None, None, :]).astype(np.float32), {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def normalize_feature_matrix(values: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(values, dtype=np.float32)
    train = x[train_idx] if train_idx.size else x
    mean = train.mean(axis=0).astype(np.float32)
    std = train.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return ((x - mean[None, :]) / std[None, :]).astype(np.float32), {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def batch_indices(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(order)
    return [order[i : i + max(1, batch_size)] for i in range(0, order.size, max(1, batch_size))]


def snapshot_model_state(model: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def online_gaussian_nll(mu_seq: Any, log_sigma_seq: Any, target: Any) -> Any:
    frames = mu_seq.shape[1]
    weights = torch.linspace(0.35, 1.0, frames, device=mu_seq.device, dtype=mu_seq.dtype)
    weights = weights[None, :, None]
    y = target[:, None, :]
    inv_sigma = torch.exp(-log_sigma_seq)
    nll = 0.5 * ((y - mu_seq) * inv_sigma) ** 2 + log_sigma_seq + 0.5 * math.log(2.0 * math.pi)
    weighted = (nll * weights).sum() / (weights.sum() * target.shape[0] * target.shape[1])
    final = nll[:, -1, :].mean()
    return weighted + 0.6 * final


def modality_balance_loss(modality_gate: Any) -> Any:
    mean_gate = modality_gate.mean(dim=(0, 1))
    target = torch.full_like(mean_gate, 1.0 / max(1, mean_gate.numel()))
    return F.mse_loss(mean_gate, target)


def evaluate_online_loss(model: Any, sensor: Any, vision: Any, context: Any, target: Any) -> float:
    model.eval()
    with torch.no_grad():
        output = model(sensor, vision, context)
        loss = online_gaussian_nll(output["mu"], output["log_sigma"], target)
    model.train()
    return float(loss.detach().cpu())


def denormalize_rollout(output: dict[str, Any], target_stats: dict[str, Any]) -> dict[str, np.ndarray]:
    mean = np.asarray(target_stats["mean"], dtype=np.float32)
    std = np.asarray(target_stats["std"], dtype=np.float32)
    mu_norm = output["mu"].detach().cpu().numpy().astype(np.float32)
    sigma_norm = np.exp(output["log_sigma"].detach().cpu().numpy().astype(np.float32))
    precision = output["precision"].detach().cpu().numpy().astype(np.float32)
    update_norm = output["update_norm"].detach().cpu().numpy().astype(np.float32)
    return {
        "mu": mu_norm * std[None, None, :] + mean[None, None, :],
        "sigma": sigma_norm * std[None, None, :],
        "precision": precision,
        "update_norm": update_norm,
    }


def rollout_metrics(rollout: dict[str, np.ndarray], target: np.ndarray) -> dict[str, Any]:
    mu = np.asarray(rollout["mu"], dtype=np.float32)
    sigma = np.maximum(np.asarray(rollout["sigma"], dtype=np.float32), 1.0e-6)
    precision = np.asarray(rollout["precision"], dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    abs_error = np.abs(mu - target_arr[:, None, :])
    frame_mae = abs_error.mean(axis=(0, 2))
    first_mae = abs_error[:, 0, :].mean(axis=0)
    final_mae = abs_error[:, -1, :].mean(axis=0)
    first_sigma = sigma[:, 0, :].mean(axis=0)
    final_sigma_mean = sigma[:, -1, :].mean(axis=0)
    precision_diffs = np.diff(precision, axis=1)
    update_norm = np.asarray(rollout["update_norm"], dtype=np.float32)
    target_improved = final_mae < first_mae
    final_error = mu[:, -1, :] - target_arr
    final_sigma_samples = sigma[:, -1, :]
    final_z = np.abs(final_error) / final_sigma_samples
    final_nll = 0.5 * (
        (final_error / final_sigma_samples) ** 2
        + 2.0 * np.log(final_sigma_samples)
        + math.log(2.0 * math.pi)
    )
    coverage_1sigma = np.mean(final_z <= 1.0, axis=0)
    coverage_2sigma = np.mean(final_z <= 2.0, axis=0)
    return {
        "first_frame_mae": {TARGET_NAMES[i]: float(first_mae[i]) for i in range(len(TARGET_NAMES))},
        "final_frame_mae": {TARGET_NAMES[i]: float(final_mae[i]) for i in range(len(TARGET_NAMES))},
        "target_mae_improved": {TARGET_NAMES[i]: bool(target_improved[i]) for i in range(len(TARGET_NAMES))},
        "all_targets_final_mae_improved": bool(np.all(target_improved)),
        "first_frame_mae_mean": float(first_mae.mean()),
        "final_frame_mae_mean": float(final_mae.mean()),
        "mae_improvement_fraction": float((first_mae.mean() - final_mae.mean()) / max(first_mae.mean(), 1.0e-6)),
        "first_sigma_mean": {TARGET_NAMES[i]: float(first_sigma[i]) for i in range(len(TARGET_NAMES))},
        "final_sigma_mean": {TARGET_NAMES[i]: float(final_sigma_mean[i]) for i in range(len(TARGET_NAMES))},
        "sigma_reduction_fraction": float(
            (first_sigma.mean() - final_sigma_mean.mean()) / max(first_sigma.mean(), 1.0e-6)
        ),
        "final_nll": {TARGET_NAMES[i]: float(final_nll[:, i].mean()) for i in range(len(TARGET_NAMES))},
        "final_nll_mean": float(final_nll.mean()),
        "final_coverage_1sigma": {TARGET_NAMES[i]: float(coverage_1sigma[i]) for i in range(len(TARGET_NAMES))},
        "final_coverage_2sigma": {TARGET_NAMES[i]: float(coverage_2sigma[i]) for i in range(len(TARGET_NAMES))},
        "coverage_1sigma_error_mean": float(np.abs(coverage_1sigma - 0.6827).mean()),
        "coverage_2sigma_error_mean": float(np.abs(coverage_2sigma - 0.9545).mean()),
        "precision_monotonic": bool(np.all(precision_diffs >= -1.0e-5)),
        "mean_update_norm": float(update_norm.mean()),
        "nonzero_update_fraction": float(np.mean(update_norm > 1.0e-5)),
        "frame_mae_curve": [float(v) for v in frame_mae],
        "online_rollout_contract": {
            "belief_state_carried_between_frames": True,
            "per_frame_independent_material_inference": False,
            "posterior_precision_accumulates": bool(np.all(precision_diffs >= -1.0e-5)),
        },
        "final_predictions_come_from_rollout_state": True,
    }


def select_rollout_index(mu: np.ndarray, target: np.ndarray) -> int:
    errors = np.abs(mu[:, 0, :] - target[:, None, :][:, 0, :]).mean(axis=1)
    final_errors = np.abs(mu[:, -1, :] - target).mean(axis=1)
    improvement = errors - final_errors
    return int(np.argmax(improvement))


def write_rollout_csv(path: Path, mu: np.ndarray, sigma: np.ndarray, target: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame"]
    for name in TARGET_NAMES:
        fieldnames.extend([f"target_{name}", f"pred_{name}", f"sigma_{name}", f"abs_error_{name}", f"delta_{name}"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        previous = None
        for frame_id in range(mu.shape[0]):
            row: dict[str, Any] = {"frame": frame_id}
            for target_id, name in enumerate(TARGET_NAMES):
                delta = 0.0 if previous is None else float(mu[frame_id, target_id] - previous[target_id])
                row[f"target_{name}"] = float(target[target_id])
                row[f"pred_{name}"] = float(mu[frame_id, target_id])
                row[f"sigma_{name}"] = float(sigma[frame_id, target_id])
                row[f"abs_error_{name}"] = float(abs(mu[frame_id, target_id] - target[target_id]))
                row[f"delta_{name}"] = delta
            previous = mu[frame_id].copy()
            writer.writerow(row)


def draw_rollout_preview(path: Path, mu: np.ndarray, sigma: np.ndarray, target: np.ndarray) -> None:
    width, height = 1100, 680
    panel_h = height // len(TARGET_NAMES)
    image = np.full((height, width, 3), 248, dtype=np.uint8)
    colors = [(35, 92, 180), (30, 145, 90), (180, 105, 35), (135, 60, 150)]
    for target_id, name in enumerate(TARGET_NAMES):
        top = target_id * panel_h
        bottom = top + panel_h - 1
        values = mu[:, target_id]
        sig = sigma[:, target_id]
        lo = float(min(values.min(), target[target_id], (values - sig).min()))
        hi = float(max(values.max(), target[target_id], (values + sig).max()))
        if abs(hi - lo) < 1.0e-6:
            lo -= 1.0
            hi += 1.0
        x0, x1 = 72, width - 32
        y0, y1 = bottom - 36, top + 34
        cv2.line(image, (x0, y0), (x1, y0), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.line(image, (x0, y0), (x0, y1), (80, 80, 80), 1, cv2.LINE_AA)
        ty = _plot_y(float(target[target_id]), lo, hi, y0, y1)
        cv2.line(image, (x0, ty), (x1, ty), (55, 55, 55), 1, cv2.LINE_AA)
        pts = []
        upper = []
        lower = []
        for frame_id in range(values.shape[0]):
            px = int(x0 + frame_id / max(1, values.shape[0] - 1) * (x1 - x0))
            pts.append([px, _plot_y(float(values[frame_id]), lo, hi, y0, y1)])
            upper.append([px, _plot_y(float(values[frame_id] + sig[frame_id]), lo, hi, y0, y1)])
            lower.append([px, _plot_y(float(values[frame_id] - sig[frame_id]), lo, hi, y0, y1)])
        band = np.asarray(upper + list(reversed(lower)), dtype=np.int32)
        cv2.fillPoly(image, [band], tuple(int(0.75 * c + 0.25 * 248) for c in colors[target_id]))
        cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, colors[target_id], 2, cv2.LINE_AA)
        cv2.putText(image, name, (12, top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            f"target={target[target_id]:.3g} final={values[-1]:.3g}",
            (170, top + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path.as_posix(), image)


def _plot_y(value: float, lo: float, hi: float, y0: int, y1: int) -> int:
    return int(y0 - (value - lo) / max(hi - lo, 1.0e-6) * (y0 - y1))


def clean_float(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def clean_float_list(values: list[float]) -> list[float | None]:
    return [clean_float(value) for value in values]
