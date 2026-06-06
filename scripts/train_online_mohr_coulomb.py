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

from granular_mpm.multimodal_learning import train_online_mohr_coulomb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/learning/online_mohr_coulomb_quick.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/online_mohr_coulomb_bestval_quick")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else ROOT / path
    with resolved.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return dict(raw.get("online_mohr_coulomb", raw))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    result = train_online_mohr_coulomb(
        output_dir=output_dir,
        config=load_config(args.config),
        quick=args.quick,
    )
    rollout = result["rollout"]
    print("online_mohr_coulomb_status=ok")
    print(f"output_dir={output_dir}")
    print(f"first_frame_mae_mean={rollout['first_frame_mae_mean']:.6g}")
    print(f"final_frame_mae_mean={rollout['final_frame_mae_mean']:.6g}")
    print(f"mae_improvement_fraction={rollout['mae_improvement_fraction']:.6g}")
    print(f"precision_monotonic={rollout['precision_monotonic']}")
    print(f"rollout_csv={rollout['rollout_csv']}")


if __name__ == "__main__":
    main()
