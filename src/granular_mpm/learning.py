from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .metrics import write_json
from .probing_dataset import load_dataset_npz

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover - exercised only on minimal installs.
    torch = None
    F = None
    nn = None


if torch is not None:

    class TemporalMDN(nn.Module):
        def __init__(
            self,
            input_dim: int,
            target_dim: int,
            latent_dim: int = 64,
            conv_channels: int = 64,
            transformer_layers: int = 2,
            attention_heads: int = 4,
            mixture_components: int = 3,
            dropout: float = 0.05,
        ):
            super().__init__()
            self.target_dim = target_dim
            self.mixture_components = mixture_components
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, conv_channels, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.GELU(),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=conv_channels,
                nhead=attention_heads,
                dim_feedforward=max(128, conv_channels * 4),
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self.latent = nn.Sequential(
                nn.LayerNorm(conv_channels),
                nn.Linear(conv_channels, latent_dim),
                nn.GELU(),
                nn.LayerNorm(latent_dim),
            )
            self.projection = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
            self.pi = nn.Linear(latent_dim, mixture_components)
            self.mu = nn.Linear(latent_dim, mixture_components * target_dim)
            self.log_sigma = nn.Linear(latent_dim, mixture_components * target_dim)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)
            h = self.encoder(h)
            pooled = h.mean(dim=1)
            return self.latent(pooled)

        def contrastive(self, x: torch.Tensor) -> torch.Tensor:
            z = self.projection(self.encode(x))
            return F.normalize(z, dim=-1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            z = self.encode(x)
            pi_logits = self.pi(z)
            mu = self.mu(z).view(x.shape[0], self.mixture_components, self.target_dim)
            log_sigma = self.log_sigma(z).view(x.shape[0], self.mixture_components, self.target_dim)
            log_sigma = torch.clamp(log_sigma, min=-5.0, max=3.0)
            return pi_logits, mu, log_sigma, z

else:

    class TemporalMDN:  # pragma: no cover
        pass


def train_temporal_mdn(
    dataset_path: Path,
    training_dir: Path,
    inference_dir: Path,
    config: dict[str, Any],
    sequence_name: str,
    quick: bool = False,
) -> dict[str, Any]:
    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is required for the learning pipeline. Install with: pip install -e '.[learning]'")

    dataset = load_dataset_npz(dataset_path)
    x = np.asarray(dataset["x"], dtype=np.float32)
    y_raw = np.asarray(dataset["y"], dtype=np.float32)
    split = np.asarray(dataset["split"], dtype=np.int32)
    metadata = dataset["metadata"]
    target_names = list(metadata.get("target_names", [f"target_{i}" for i in range(y_raw.shape[1])]))

    training_dir.mkdir(parents=True, exist_ok=True)
    inference_dir.mkdir(parents=True, exist_ok=True)
    if x.shape[0] == 0:
        payload = {"status": "empty_dataset", "sequence_name": sequence_name}
        write_json(training_dir / "mdn_training_metrics.json", payload)
        write_json(inference_dir / "learning_inference_metrics.json", payload)
        return payload

    seed = int(config.get("seed", 7))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    if quick:
        representation_epochs = int(config.get("quick_representation_epochs", 1))
        mdn_epochs = int(config.get("quick_mdn_epochs", 1))
        batch_size = int(config.get("quick_batch_size", min(8, max(1, x.shape[0]))))
    else:
        representation_epochs = int(config.get("representation_epochs", 6))
        mdn_epochs = int(config.get("mdn_epochs", 12))
        batch_size = int(config.get("batch_size", 32))

    train_idx = np.where(split == 0)[0]
    val_idx = np.where(split == 1)[0]
    test_idx = np.where(split == 2)[0]
    if train_idx.size == 0:
        train_idx = np.arange(x.shape[0])
    eval_idx = test_idx if test_idx.size else (val_idx if val_idx.size else np.arange(x.shape[0]))

    target_mean, target_std = target_normalization(y_raw, train_idx)
    y_norm = ((y_raw - target_mean[None, :]) / target_std[None, :]).astype(np.float32)

    model = TemporalMDN(
        input_dim=x.shape[-1],
        target_dim=y_raw.shape[-1],
        latent_dim=int(config.get("latent_dim", 64)),
        conv_channels=int(config.get("conv_channels", 64)),
        transformer_layers=int(config.get("transformer_layers", 2)),
        attention_heads=int(config.get("attention_heads", 4)),
        mixture_components=int(config.get("num_mixtures", 3)),
        dropout=float(config.get("dropout", 0.05)),
    ).to(device)

    x_tensor = torch.from_numpy(x).to(device)
    y_tensor = torch.from_numpy(y_norm).to(device)
    rng = np.random.default_rng(seed)

    rep_optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("representation_lr", 1.0e-3)))
    rep_losses: list[float] = []
    noise_std = float(config.get("augmentation_noise_std", 0.015))
    for _epoch in range(representation_epochs):
        epoch_losses: list[float] = []
        for batch in batch_indices(train_idx, batch_size, rng, shuffle=True):
            if len(batch) < 2:
                continue
            xb = x_tensor[batch]
            x1 = xb + torch.randn_like(xb) * noise_std
            x2 = xb + torch.randn_like(xb) * noise_std
            loss = info_nce_loss(model.contrastive(x1), model.contrastive(x2), float(config.get("temperature", 0.12)))
            rep_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            rep_optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        rep_losses.append(float(np.mean(epoch_losses)) if epoch_losses else math.nan)

    mdn_optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("mdn_lr", 8.0e-4)))
    mdn_losses: list[float] = []
    val_losses: list[float] = []
    for _epoch in range(mdn_epochs):
        epoch_losses = []
        for batch in batch_indices(train_idx, batch_size, rng, shuffle=True):
            xb = x_tensor[batch]
            yb = y_tensor[batch]
            pi_logits, mu, log_sigma, _z = model(xb)
            loss = mdn_nll(pi_logits, mu, log_sigma, yb)
            mdn_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            mdn_optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mdn_losses.append(float(np.mean(epoch_losses)) if epoch_losses else math.nan)
        if val_idx.size:
            val_losses.append(float(evaluate_nll(model, x_tensor[val_idx], y_tensor[val_idx])))

    inference = run_inference(model, x_tensor, eval_idx, target_mean, target_std, device)
    prediction_rows = prediction_rows_from_arrays(
        sample_indices=eval_idx,
        split=split,
        target_names=target_names,
        target=y_raw[eval_idx],
        prediction=inference["expected"],
    )
    write_prediction_csv(inference_dir / "mdn_predictions.csv", prediction_rows, target_names)

    mae = np.mean(np.abs(inference["expected"] - y_raw[eval_idx]), axis=0) if eval_idx.size else np.zeros(y_raw.shape[1])
    rmse = np.sqrt(np.mean((inference["expected"] - y_raw[eval_idx]) ** 2, axis=0)) if eval_idx.size else np.zeros(y_raw.shape[1])
    inference_ms = measure_inference_ms(model, x_tensor[eval_idx], repeats=int(config.get("timing_repeats", 10)))

    rep_status = "ok" if any(np.isfinite(loss) for loss in rep_losses) else "insufficient_pairs"
    representation_payload = {
        "status": rep_status,
        "sequence_name": sequence_name,
        "phase": "representation_learning",
        "loss": clean_float_list(rep_losses),
        "loss_final": clean_float(rep_losses[-1]) if rep_losses else None,
        "epochs": representation_epochs,
        "method": "simclr_info_nce",
    }
    training_payload = {
        "status": "ok",
        "sequence_name": sequence_name,
        "phase": "mdn_finetune",
        "epochs": mdn_epochs,
        "batch_size": batch_size,
        "device": str(device),
        "num_mixtures": int(config.get("num_mixtures", 3)),
        "target_names": target_names,
        "target_normalization": {"mean": target_mean.tolist(), "std": target_std.tolist()},
        "train_loss": clean_float_list(mdn_losses),
        "validation_nll": clean_float_list(val_losses),
        "train_loss_final": clean_float(mdn_losses[-1]) if mdn_losses else None,
        "validation_nll_final": clean_float(val_losses[-1]) if val_losses else None,
    }
    inference_payload = {
        "status": "ok",
        "sequence_name": sequence_name,
        "target_names": target_names,
        "eval_split": split_name(int(split[eval_idx[0]])) if eval_idx.size else "none",
        "eval_count": int(eval_idx.size),
        "inference_ms_per_window": inference_ms,
        "mae": {target_names[i]: float(mae[i]) for i in range(len(target_names))},
        "rmse": {target_names[i]: float(rmse[i]) for i in range(len(target_names))},
        "prediction_csv": (inference_dir / "mdn_predictions.csv").as_posix(),
    }

    posterior_payload = write_posterior_artifacts(
        inference_dir=inference_dir,
        target_names=target_names,
        target_mean=target_mean,
        target_std=target_std,
        target=y_raw[eval_idx[0]] if eval_idx.size else None,
        pi=inference["pi"][0] if eval_idx.size else None,
        mu=inference["mu"][0] if eval_idx.size else None,
        sigma=inference["sigma"][0] if eval_idx.size else None,
    )

    write_json(training_dir / "representation_metrics.json", representation_payload)
    write_json(training_dir / "mdn_training_metrics.json", training_payload)
    write_json(inference_dir / "learning_inference_metrics.json", inference_payload)
    write_json(inference_dir / "posterior_summary.json", posterior_payload)
    write_json(training_dir / "model_config.json", {"metadata": metadata, "learning": config})
    torch.save(
        {
            "model_state": model.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "target_names": target_names,
            "config": config,
            "metadata": metadata,
        },
        training_dir / "temporal_mdn.pt",
    )

    return {
        "representation": representation_payload,
        "training": training_payload,
        "inference": inference_payload,
        "posterior": posterior_payload,
    }


def info_nce_loss(z1: Any, z2: Any, temperature: float = 0.12) -> Any:
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def mdn_nll(pi_logits: Any, mu: Any, log_sigma: Any, target: Any) -> Any:
    y = target[:, None, :]
    inv_sigma = torch.exp(-log_sigma)
    log_prob = -0.5 * ((y - mu) * inv_sigma) ** 2 - log_sigma - 0.5 * math.log(2.0 * math.pi)
    log_prob = log_prob.sum(dim=-1) + F.log_softmax(pi_logits, dim=-1)
    return -torch.logsumexp(log_prob, dim=-1).mean()


def evaluate_nll(model: Any, x: Any, y: Any) -> float:
    model.eval()
    with torch.no_grad():
        pi_logits, mu, log_sigma, _z = model(x)
        loss = mdn_nll(pi_logits, mu, log_sigma, y)
    model.train()
    return float(loss.detach().cpu())


def run_inference(
    model: Any,
    x_tensor: Any,
    indices: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: Any,
) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        if indices.size:
            pi_logits, mu_norm, log_sigma_norm, _z = model(x_tensor[indices].to(device))
            pi = torch.softmax(pi_logits, dim=-1).cpu().numpy()
            mu_norm_np = mu_norm.cpu().numpy()
            sigma_norm_np = torch.exp(log_sigma_norm).cpu().numpy()
        else:
            components = getattr(model, "mixture_components", 1)
            target_dim = target_mean.shape[0]
            pi = np.zeros((0, components), dtype=np.float32)
            mu_norm_np = np.zeros((0, components, target_dim), dtype=np.float32)
            sigma_norm_np = np.zeros((0, components, target_dim), dtype=np.float32)
    mu = mu_norm_np * target_std[None, None, :] + target_mean[None, None, :]
    sigma = sigma_norm_np * target_std[None, None, :]
    expected = (pi[:, :, None] * mu).sum(axis=1) if indices.size else np.zeros((0, target_mean.shape[0]), dtype=np.float32)
    model.train()
    return {"pi": pi, "mu": mu, "sigma": sigma, "expected": expected.astype(np.float32)}


def measure_inference_ms(model: Any, x: Any, repeats: int = 10) -> float | None:
    if x.shape[0] == 0:
        return None
    model.eval()
    with torch.no_grad():
        _ = model(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        start = time.perf_counter()
        for _ in range(max(1, repeats)):
            _ = model(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        elapsed = time.perf_counter() - start
    model.train()
    return float(1000.0 * elapsed / (max(1, repeats) * x.shape[0]))


def target_normalization(y: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = y[train_idx] if train_idx.size else y
    mean = train.mean(axis=0).astype(np.float32)
    std = train.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return mean, std


def batch_indices(indices: np.ndarray, batch_size: int, rng: np.random.Generator, shuffle: bool) -> list[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        rng.shuffle(order)
    return [order[i : i + max(1, batch_size)] for i in range(0, order.size, max(1, batch_size))]


def prediction_rows_from_arrays(
    sample_indices: np.ndarray,
    split: np.ndarray,
    target_names: list[str],
    target: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_id, sample_index in enumerate(sample_indices):
        row: dict[str, Any] = {"sample_index": int(sample_index), "split": split_name(int(split[sample_index]))}
        for target_id, name in enumerate(target_names):
            row[f"target_{name}"] = float(target[row_id, target_id])
            row[f"pred_{name}"] = float(prediction[row_id, target_id])
            row[f"abs_error_{name}"] = float(abs(prediction[row_id, target_id] - target[row_id, target_id]))
        rows.append(row)
    return rows


def write_prediction_csv(path: Path, rows: list[dict[str, Any]], target_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_index", "split"]
    for name in target_names:
        fieldnames.extend([f"target_{name}", f"pred_{name}", f"abs_error_{name}"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_posterior_artifacts(
    inference_dir: Path,
    target_names: list[str],
    target_mean: np.ndarray,
    target_std: np.ndarray,
    target: np.ndarray | None,
    pi: np.ndarray | None,
    mu: np.ndarray | None,
    sigma: np.ndarray | None,
) -> dict[str, Any]:
    if target is None or pi is None or mu is None or sigma is None:
        return {"status": "no_eval_sample"}

    plots: dict[str, str] = {}
    component_rows: list[dict[str, Any]] = []
    for target_id, name in enumerate(target_names):
        plot_path = inference_dir / f"posterior_{name}.png"
        draw_marginal_posterior(
            plot_path,
            title=f"posterior {name}",
            target_value=float(target[target_id]),
            weights=pi,
            means=mu[:, target_id],
            sigmas=sigma[:, target_id],
        )
        plots[name] = plot_path.as_posix()
        component_rows.append(
            {
                "target": name,
                "true": float(target[target_id]),
                "components": [
                    {
                        "weight": float(pi[i]),
                        "mean": float(mu[i, target_id]),
                        "sigma": float(max(sigma[i, target_id], 1.0e-6)),
                    }
                    for i in range(pi.shape[0])
                ],
                "normalization_mean": float(target_mean[target_id]),
                "normalization_std": float(target_std[target_id]),
            }
        )
    return {"status": "ok", "plots": plots, "marginals": component_rows}


def draw_marginal_posterior(
    path: Path,
    title: str,
    target_value: float,
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
) -> None:
    width, height = 720, 420
    margin_l, margin_r, margin_t, margin_b = 72, 28, 48, 58
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    sigmas = np.maximum(np.asarray(sigmas, dtype=np.float32), 1.0e-4)
    means = np.asarray(means, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    lo = float(min(np.min(means - 4.0 * sigmas), target_value - 1.0))
    hi = float(max(np.max(means + 4.0 * sigmas), target_value + 1.0))
    if abs(hi - lo) < 1.0e-6:
        lo -= 1.0
        hi += 1.0
    xs = np.linspace(lo, hi, 400, dtype=np.float32)
    density = np.zeros_like(xs)
    for w, m, s in zip(weights, means, sigmas):
        density += w * np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * math.sqrt(2.0 * math.pi))
    ymax = float(max(density.max(), 1.0e-6))

    x0, x1 = margin_l, width - margin_r
    y0, y1 = height - margin_b, margin_t
    cv2.line(image, (x0, y0), (x1, y0), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(image, (x0, y0), (x0, y1), (70, 70, 70), 1, cv2.LINE_AA)
    pts = []
    for xval, yval in zip(xs, density):
        px = int(x0 + (float(xval) - lo) / (hi - lo) * (x1 - x0))
        py = int(y0 - float(yval) / ymax * (y0 - y1))
        pts.append([px, py])
    if len(pts) > 1:
        cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, (20, 95, 180), 2, cv2.LINE_AA)

    tx = int(x0 + (target_value - lo) / (hi - lo) * (x1 - x0))
    cv2.line(image, (tx, y0), (tx, y1), (35, 35, 210), 2, cv2.LINE_AA)
    cv2.putText(image, title, (margin_l, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(image, f"target={target_value:.4g}", (max(margin_l, tx - 72), y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 210), 1, cv2.LINE_AA)
    cv2.putText(image, f"{lo:.3g}", (x0 - 8, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(image, f"{hi:.3g}", (x1 - 48, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path.as_posix(), image)


def split_name(code: int) -> str:
    return {0: "train", 1: "validation", 2: "test"}.get(code, "unknown")


def clean_float(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def clean_float_list(values: list[float]) -> list[float | None]:
    return [clean_float(value) for value in values]
