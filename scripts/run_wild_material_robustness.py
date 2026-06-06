from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm.wild_material_learning import train_wild_material_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/learning/wild_material_robustness.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/wild_material_robustness")
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
    result = train_wild_material_model(output_dir=output_dir, config=load_config(args.config), quick=args.quick)
    evaluation = result["evaluation"]
    print("wild_material_robustness_status=ok")
    print(f"output_dir={output_dir}")
    print(f"family_accuracy={evaluation['family_accuracy']:.6g}")
    print(f"worst_family_accuracy={evaluation['worst_family_accuracy']:.6g}")
    print(f"final_frame_nmae_mean={evaluation['final_frame_nmae_mean']:.6g}")
    print(f"coverage_error_mean={evaluation['coverage_error_mean']:.6g}")
    print(f"paper_ready={evaluation['paper_ready']}")
    print(f"summary_figure={evaluation['summary_figure']}")


if __name__ == "__main__":
    main()
