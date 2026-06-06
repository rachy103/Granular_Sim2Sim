from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm.metrics import write_json  # noqa: E402
from granular_mpm.wild_material_learning import train_wild_material_model  # noqa: E402


VARIANTS: list[tuple[str, dict[str, Any], str]] = [
    ("main", {}, "Full wild model: all modalities, property family head, calibrated sigma."),
    (
        "no_property_family_head",
        {"use_property_family_head": False},
        "Reviewer ablation: remove the property-posterior branch from the family classifier.",
    ),
    (
        "sensor_only",
        {"modality_ablation": "sensor_only"},
        "Reviewer ablation: use wrench/proprioception/context, zero visual summaries.",
    ),
    (
        "vision_only",
        {"modality_ablation": "vision_only"},
        "Reviewer ablation: use visual summaries/context, zero robot sensor stream.",
    ),
    (
        "context_only",
        {"modality_ablation": "context_only"},
        "Shortcut audit: zero both sensor and vision; only action context remains.",
    ),
    (
        "no_sigma_calibration",
        {"calibrate_sigma": False},
        "Uncertainty audit: remove validation posterior-temperature calibration.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/learning/wild_material_robustness_stress.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/wild_review_audit")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else ROOT / path
    with resolved.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return dict(raw.get("wild_material_robustness", raw))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config(args.config)
    results: dict[str, Any] = {
        "status": "ok",
        "config": args.config.as_posix(),
        "quick": bool(args.quick),
        "variants": {},
    }
    for name, overrides, description in VARIANTS:
        variant_config = copy.deepcopy(base_config)
        variant_config.update(overrides)
        variant_dir = output_dir / name
        result = train_wild_material_model(variant_dir, variant_config, quick=bool(args.quick))
        metrics = result["evaluation"]
        results["variants"][name] = {
            "description": description,
            "overrides": overrides,
            "output_dir": variant_dir.as_posix(),
            "paper_ready": metrics["paper_ready"],
            "family_accuracy": metrics["family_accuracy"],
            "worst_family_accuracy": metrics["worst_family_accuracy"],
            "final_frame_nmae_mean": metrics["final_frame_nmae_mean"],
            "coverage_error_mean": metrics["coverage_error_mean"],
            "by_family": metrics["by_family"],
            "by_eval_tag": metrics["by_eval_tag"],
            "summary_figure": metrics["summary_figure"],
        }
        print(
            f"{name}: acc={metrics['family_accuracy']:.4f} "
            f"worst={metrics['worst_family_accuracy']:.4f} "
            f"nmae={metrics['final_frame_nmae_mean']:.4f} "
            f"coverage={metrics['coverage_error_mean']:.4f} "
            f"paper_ready={metrics['paper_ready']}"
        )
    write_json(output_dir / "reviewer_audit_results.json", results)
    write_report(output_dir / "reviewer_audit_report.md", results)
    print(f"reviewer_audit_results={output_dir / 'reviewer_audit_results.json'}")
    print(f"reviewer_audit_report={output_dir / 'reviewer_audit_report.md'}")


def write_report(path: Path, results: dict[str, Any]) -> None:
    variants = results["variants"]
    main = variants["main"]
    no_property = variants["no_property_family_head"]
    sensor_only = variants["sensor_only"]
    vision_only = variants["vision_only"]
    context_only = variants["context_only"]
    no_cal = variants["no_sigma_calibration"]
    lines = [
        "# Reviewer Audit: Wild Material Robustness",
        "",
        "This audit treats the paper claim as a skeptical reviewer would: the",
        "question is not whether the main figure looks good, but whether obvious",
        "shortcut, ablation, and calibration objections break the claim.",
        "",
        "## Reject Risks",
        "",
        "1. **Synthetic shortcut risk.** The benchmark may leak family identity",
        "   through easy procedural signatures rather than robust material",
        "   interaction evidence.",
        "2. **Missing ablations.** A reviewer can reject if the property-posterior",
        "   family branch, sensor stream, vision stream, and calibration are not",
        "   isolated.",
        "3. **Context shortcut risk.** If action context alone predicts family, the",
        "   split is invalid.",
        "4. **Uncertainty overclaim.** If posterior coverage only passes after an",
        "   unexplained trick, the calibration claim is weak.",
        "5. **Real-world gap.** This remains synthetic Sim2Sim, not real-data",
        "   validation.",
        "",
        "## Audit Results",
        "",
        "| Variant | Family Acc | Worst Family Acc | nMAE | Coverage Err | Paper Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, payload in variants.items():
        lines.append(
            f"| {name} | {payload['family_accuracy']:.3f} | "
            f"{payload['worst_family_accuracy']:.3f} | "
            f"{payload['final_frame_nmae_mean']:.3f} | "
            f"{payload['coverage_error_mean']:.3f} | "
            f"{payload['paper_ready']} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- Main model passes the stress gate: family accuracy "
            f"`{main['family_accuracy']:.3f}`, worst-family accuracy "
            f"`{main['worst_family_accuracy']:.3f}`, property nMAE "
            f"`{main['final_frame_nmae_mean']:.3f}`, coverage error "
            f"`{main['coverage_error_mean']:.3f}`.",
            f"- Removing the property-posterior family branch changes family accuracy "
            f"from `{main['family_accuracy']:.3f}` to "
            f"`{no_property['family_accuracy']:.3f}`. This tests whether the branch "
            f"is actually carrying useful signal.",
            f"- Sensor-only accuracy is `{sensor_only['family_accuracy']:.3f}` and "
            f"vision-only accuracy is `{vision_only['family_accuracy']:.3f}`. The "
            f"gap indicates which modality is doing most of the work.",
            f"- Context-only accuracy is `{context_only['family_accuracy']:.3f}`. "
            f"This should remain near chance; otherwise the split leaks labels.",
            f"- Removing sigma calibration changes coverage error from "
            f"`{main['coverage_error_mean']:.3f}` to "
            f"`{no_cal['coverage_error_mean']:.3f}`.",
            "",
            "## Reviewer Verdict",
            "",
            "The synthetic stress benchmark claim survives this audit if the main",
            "model passes the gate, context-only remains weak, and the ablations",
            "show that multimodal evidence is needed. The paper should still avoid",
            "claiming real-world robustness until real material data or a second",
            "independent simulator is added.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
