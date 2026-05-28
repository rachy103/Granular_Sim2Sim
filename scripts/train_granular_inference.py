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

from granular_mpm.learning import train_temporal_mdn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def load_learning_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return dict(raw.get("learning", {}))


def main() -> None:
    args = parse_args()
    config = load_learning_config(args.config)
    result = train_temporal_mdn(
        dataset_path=args.dataset if args.dataset.is_absolute() else ROOT / args.dataset,
        training_dir=args.training_dir if args.training_dir.is_absolute() else ROOT / args.training_dir,
        inference_dir=args.inference_dir if args.inference_dir.is_absolute() else ROOT / args.inference_dir,
        config=config,
        sequence_name=args.sequence_name,
        quick=args.quick,
    )
    status = result.get("training", result).get("status", "ok") if isinstance(result, dict) else "ok"
    print(f"learning_status={status}")
    print(f"training_dir={args.training_dir}")
    print(f"inference_dir={args.inference_dir}")


if __name__ == "__main__":
    main()
