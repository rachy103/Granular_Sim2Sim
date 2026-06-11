"""Reproduce the Aalto granular force-identification result on real data.

The official repository currently exposes the dataset, while the paper's
classification code is not published. This script follows the paper-specified
feature extraction and evaluation protocol:

* all six F/T axes, 1600 samples per axis at 500 Hz
* raw signal plus High-Frequency Magnitude Histogram (HFMH)
* 8th-order Butterworth high-pass filter at 23 Hz
* 100 histogram bins over [-1.5, 1.5] for each of the 6 axes
* 10 random stratified splits, with 50 train and 12 test samples per class
* linear hinge-loss multiclass classifier
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


RAW_BASE_URL = "https://raw.githubusercontent.com/samhyn/granular_identification/main"
CLASS_FILES = [
    ("dry_peas", "Dry peas"),
    ("rice", "Rice"),
    ("wheat_flour", "Wheat flour"),
    ("clay_granule", "Clay granule"),
    ("oat_flakes", "Oat flakes"),
    ("potting_gravel", "Potting gravel"),
    ("sunflower_seeds", "Sunflower seeds"),
    ("breadcrumbs", "Breadcrumbs"),
    ("macaroni", "Macaroni"),
    ("fine_sugar", "Fine sugar"),
    ("cat_litter", "Cat litter"),
]

PAPER_RESULTS = {
    "raw_hfmh": (0.9705, 0.0098),
    "norm_raw_hfmh": (0.9303, 0.0155),
    "hfmh": (0.8705, 0.0163),
    "raw": (0.7614, 0.0265),
    "norm_raw": (0.8731, 0.0251),
}


@dataclass(frozen=True)
class Dataset:
    signals: np.ndarray
    labels: np.ndarray
    material_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(__file__).resolve().parent / "results"
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
    )
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fs", type=float, default=500.0)
    parser.add_argument("--cutoff-hz", type=float, default=23.0)
    parser.add_argument("--hist-bins", type=int, default=100)
    parser.add_argument("--hist-min", type=float, default=-1.5)
    parser.add_argument("--hist-max", type=float, default=1.5)
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def download_dataset(data_dir: Path, skip_download: bool) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    if skip_download:
        return
    for slug, _ in CLASS_FILES:
        file_name = f"force_torque_data_all_axes_{slug}.csv"
        dst = data_dir / file_name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        url = f"{RAW_BASE_URL}/{file_name}"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dst)


def load_dataset(data_dir: Path) -> Dataset:
    signals = []
    labels = []
    material_names = [name for _, name in CLASS_FILES]
    for class_index, (slug, _) in enumerate(CLASS_FILES, start=1):
        path = data_dir / f"force_torque_data_all_axes_{slug}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        rows = np.loadtxt(path, delimiter=",", dtype=np.float64)
        if rows.ndim == 1:
            rows = rows[None, :]
        x = rows[:, : 1600 * 6].reshape(rows.shape[0], 6, 1600)
        y = rows[:, -1].astype(np.int64)
        if not np.all(y == class_index):
            print(
                f"Warning: labels in {path.name} do not all equal {class_index}; "
                "using file order labels.",
                file=sys.stderr,
            )
            y = np.full(rows.shape[0], class_index, dtype=np.int64)
        signals.append(x)
        labels.append(y)
    return Dataset(
        signals=np.concatenate(signals, axis=0),
        labels=np.concatenate(labels, axis=0) - 1,
        material_names=material_names,
    )


def highpass(signals: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    sos = signal.butter(
        8,
        cutoff_hz,
        btype="highpass",
        fs=fs,
        output="sos",
    )
    return signal.sosfiltfilt(sos, signals, axis=2)


def hfmh_features(
    signals: np.ndarray,
    fs: float,
    cutoff_hz: float,
    hist_bins: int,
    hist_range: tuple[float, float],
) -> np.ndarray:
    filtered = highpass(signals, fs=fs, cutoff_hz=cutoff_hz)
    feats = np.empty((signals.shape[0], signals.shape[1] * hist_bins), dtype=np.float64)
    for sample_idx in range(signals.shape[0]):
        offset = 0
        for axis_idx in range(signals.shape[1]):
            hist, _ = np.histogram(
                filtered[sample_idx, axis_idx],
                bins=hist_bins,
                range=hist_range,
            )
            feats[sample_idx, offset : offset + hist_bins] = hist
            offset += hist_bins
    return feats


def minmax_rows(x: np.ndarray) -> np.ndarray:
    row_min = x.min(axis=1, keepdims=True)
    row_max = x.max(axis=1, keepdims=True)
    denom = np.maximum(row_max - row_min, 1e-12)
    return (x - row_min) / denom


def make_feature_bank(args: argparse.Namespace, dataset: Dataset, prefix: float = 1.0) -> dict[str, np.ndarray]:
    n_steps = max(64, int(round(dataset.signals.shape[2] * prefix)))
    x = dataset.signals[:, :, :n_steps]
    raw = x.reshape(x.shape[0], -1)
    hfmh = hfmh_features(
        x,
        fs=args.fs,
        cutoff_hz=args.cutoff_hz,
        hist_bins=args.hist_bins,
        hist_range=(args.hist_min, args.hist_max),
    )
    norm_raw = minmax_rows(raw)
    return {
        "raw_hfmh": np.concatenate([raw, hfmh], axis=1),
        "norm_raw_hfmh": np.concatenate([norm_raw, hfmh], axis=1),
        "hfmh": hfmh,
        "raw": raw,
        "norm_raw": norm_raw,
    }


def stratified_split(labels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train_parts = []
    test_parts = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        shuffled = rng.permutation(idx)
        train_parts.append(shuffled[:50])
        test_parts.append(shuffled[50:62])
    return np.concatenate(train_parts), np.concatenate(test_parts)


def build_classifier(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                SGDClassifier(
                    loss="hinge",
                    alpha=1e-4,
                    max_iter=5000,
                    tol=1e-4,
                    random_state=seed,
                    average=True,
                ),
            ),
        ]
    )


def evaluate_feature_bank(
    feature_bank: dict[str, np.ndarray],
    labels: np.ndarray,
    splits: int,
    seed: int,
    track_confusion_for: str = "raw_hfmh",
) -> tuple[list[dict[str, float | int | str]], np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    total_cm = np.zeros((len(np.unique(labels)), len(np.unique(labels))), dtype=np.int64)
    for split_id in range(splits):
        train_idx, test_idx = stratified_split(labels, rng)
        for feature_name, features in feature_bank.items():
            clf = build_classifier(seed + split_id)
            clf.fit(features[train_idx], labels[train_idx])
            pred = clf.predict(features[test_idx])
            acc = accuracy_score(labels[test_idx], pred)
            cm = confusion_matrix(labels[test_idx], pred, labels=np.arange(total_cm.shape[0]))
            per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
            rows.append(
                {
                    "split": split_id,
                    "feature": feature_name,
                    "accuracy": float(acc),
                    "worst_class_accuracy": float(per_class.min()),
                }
            )
            if feature_name == track_confusion_for:
                total_cm += cm
    return rows, total_cm


def summarize_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    by_feature: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_feature.setdefault(str(row["feature"]), []).append(row)

    summary = []
    for feature, feature_rows in sorted(by_feature.items()):
        accuracies = np.array([float(row["accuracy"]) for row in feature_rows])
        worst = np.array([float(row["worst_class_accuracy"]) for row in feature_rows])
        paper_mean, paper_sd = PAPER_RESULTS.get(feature, (math.nan, math.nan))
        summary.append(
            {
                "feature": feature,
                "accuracy_mean": float(accuracies.mean()),
                "accuracy_sd": float(accuracies.std(ddof=1)),
                "worst_class_accuracy_mean": float(worst.mean()),
                "paper_accuracy_mean": paper_mean,
                "paper_accuracy_sd": paper_sd,
            }
        )
    return summary


def evaluate_prefixes(
    args: argparse.Namespace,
    dataset: Dataset,
    prefix_values: list[float],
) -> list[dict[str, float]]:
    rows = []
    for prefix in prefix_values:
        print(f"Evaluating prefix {prefix:.2f}")
        feature_bank = make_feature_bank(args, dataset, prefix=prefix)
        split_rows, _ = evaluate_feature_bank(
            {"raw_hfmh": feature_bank["raw_hfmh"]},
            dataset.labels,
            splits=args.splits,
            seed=args.seed + int(prefix * 1000),
            track_confusion_for="raw_hfmh",
        )
        accuracies = np.array([float(row["accuracy"]) for row in split_rows])
        worst = np.array([float(row["worst_class_accuracy"]) for row in split_rows])
        rows.append(
            {
                "prefix_fraction": prefix,
                "seconds": prefix * 3.2,
                "accuracy_mean": float(accuracies.mean()),
                "accuracy_sd": float(accuracies.std(ddof=1)),
                "worst_class_accuracy_mean": float(worst.mean()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: list[dict[str, float | str]], output_path: Path) -> None:
    ordered = ["raw", "norm_raw", "hfmh", "norm_raw_hfmh", "raw_hfmh"]
    rows = {str(row["feature"]): row for row in summary}
    labels = [name for name in ordered if name in rows]
    ours = [float(rows[name]["accuracy_mean"]) for name in labels]
    ours_sd = [float(rows[name]["accuracy_sd"]) for name in labels]
    paper = [float(rows[name]["paper_accuracy_mean"]) for name in labels]
    paper_sd = [float(rows[name]["paper_accuracy_sd"]) for name in labels]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width / 2, ours, width, yerr=ours_sd, label="This run", color="#2F80ED")
    ax.bar(x + width / 2, paper, width, yerr=paper_sd, label="Paper", color="#BDBDBD")
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Real granular material classification from force/torque signals")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_prefix(prefix_rows: list[dict[str, float]], output_path: Path) -> None:
    xs = np.array([row["seconds"] for row in prefix_rows], dtype=float)
    ys = np.array([row["accuracy_mean"] for row in prefix_rows], dtype=float)
    yerr = np.array([row["accuracy_sd"] for row in prefix_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2.0, color="#EB5757")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Observed prefix length (s)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("How early does real force interaction reveal material class?")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, names: list[str], output_path: Path) -> None:
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Raw + HFMH confusion matrix, aggregated over splits")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    short_names = [
        "peas",
        "rice",
        "flour",
        "clay",
        "oats",
        "gravel",
        "seeds",
        "crumbs",
        "macaroni",
        "sugar",
        "litter",
    ]
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_yticklabels(short_names)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            value = cm_norm[i, j]
            if value >= 0.05:
                color = "white" if value > 0.55 else "#111111"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary: list[dict[str, float | str]],
    prefix_rows: list[dict[str, float]],
    args: argparse.Namespace,
) -> None:
    rows_by_feature = {str(row["feature"]): row for row in summary}
    main = rows_by_feature["raw_hfmh"]
    lines = [
        "# Aalto Force-Based Real Granular Classification",
        "",
        "This is a source-specified Python reproduction of the method described in",
        "\"Interactive Identification of Granular Materials using Force Measurements\".",
        "The public GitHub repository currently exposes the dataset, while the paper's",
        "classification code is not included in the repository.",
        "",
        "## Protocol",
        "",
        f"- Dataset: 11 real granular materials, 62 trials per class, {11 * 62} trials total.",
        "- Signal: six-axis force/torque, 1600 samples per axis, 500 Hz, 3.2 s.",
        f"- HFMH: 8th-order Butterworth high-pass at {args.cutoff_hz:g} Hz, "
        f"{args.hist_bins} bins over [{args.hist_min:g}, {args.hist_max:g}] per axis.",
        "- Main feature: raw signal (9600D) + HFMH (600D) = 10200D.",
        f"- Evaluation: {args.splits} random splits, 50 train and 12 test samples per class.",
        "- Classifier: standardized linear hinge-loss classifier.",
        "",
        "## Key Result",
        "",
        "| Feature | This run acc. | Paper acc. | Worst-class acc. |",
        "|---|---:|---:|---:|",
    ]
    for feature in ["raw", "norm_raw", "hfmh", "norm_raw_hfmh", "raw_hfmh"]:
        row = rows_by_feature[feature]
        lines.append(
            "| "
            f"{feature} | "
            f"{float(row['accuracy_mean']):.4f} +/- {float(row['accuracy_sd']):.4f} | "
            f"{float(row['paper_accuracy_mean']):.4f} +/- {float(row['paper_accuracy_sd']):.4f} | "
            f"{float(row['worst_class_accuracy_mean']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Prefix Result",
            "",
            "| Prefix | Seconds | Accuracy | Worst-class acc. |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in prefix_rows:
        lines.append(
            f"| {float(row['prefix_fraction']):.2f} | "
            f"{float(row['seconds']):.2f} | "
            f"{float(row['accuracy_mean']):.4f} +/- {float(row['accuracy_sd']):.4f} | "
            f"{float(row['worst_class_accuracy_mean']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation for Granular Sim2Sim",
            "",
            f"The real-data Raw+HFMH classifier reaches {float(main['accuracy_mean']):.2%} "
            "accuracy on real robot F/T interaction data. This supports the claim that",
            "direct force interaction contains early, material-specific information.",
            "For our paper, this should be framed as external evidence and a force-only",
            "real-data sanity check, not as a direct excavation or property-regression",
            "comparison.",
            "",
            "Generated files:",
            "",
            "- `aalto_feature_summary.csv`",
            "- `aalto_split_results.csv`",
            "- `aalto_prefix_results.csv`",
            "- `aalto_summary_accuracy.png`",
            "- `aalto_prefix_accuracy.png`",
            "- `aalto_confusion_matrix.png`",
        ]
    )
    (output_dir / "aalto_real_classification_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    download_dataset(data_dir, skip_download=args.skip_download)
    dataset = load_dataset(data_dir)
    print(f"Loaded signals: {dataset.signals.shape}, labels: {dataset.labels.shape}")

    feature_bank = make_feature_bank(args, dataset, prefix=1.0)
    split_rows, cm = evaluate_feature_bank(
        feature_bank,
        dataset.labels,
        splits=args.splits,
        seed=args.seed,
    )
    summary = summarize_rows(split_rows)
    prefix_rows = evaluate_prefixes(args, dataset, prefix_values=[0.10, 0.25, 0.50, 1.00])

    write_csv(output_dir / "aalto_split_results.csv", split_rows)
    write_csv(output_dir / "aalto_feature_summary.csv", summary)
    write_csv(output_dir / "aalto_prefix_results.csv", prefix_rows)
    plot_summary(summary, output_dir / "aalto_summary_accuracy.png")
    plot_prefix(prefix_rows, output_dir / "aalto_prefix_accuracy.png")
    plot_confusion(cm, dataset.material_names, output_dir / "aalto_confusion_matrix.png")
    write_report(output_dir, summary, prefix_rows, args)

    result = {
        "dataset_shape": list(dataset.signals.shape),
        "splits": args.splits,
        "summary": summary,
        "prefix": prefix_rows,
    }
    (output_dir / "aalto_real_classification_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
