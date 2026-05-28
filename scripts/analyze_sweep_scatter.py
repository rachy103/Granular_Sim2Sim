from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm.scatter_analysis import analyze_sweep_scatter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--contact-threshold", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root if args.sweep_root.is_absolute() else ROOT / args.sweep_root
    out_dir = args.out_dir if args.out_dir is None or args.out_dir.is_absolute() else ROOT / args.out_dir
    report = analyze_sweep_scatter(
        sweep_root=sweep_root,
        out_dir=out_dir,
        contact_threshold=args.contact_threshold,
    )
    diag = report["diagnostics"]
    print(f"sweep_root={report['sweep_root']}")
    print(f"analysis_dir={report['out_dir']}")
    print(f"summary_csv={report['summary_csv']}")
    print(f"report={Path(report['out_dir']) / 'scatter_report.md'}")
    print(f"issues={','.join(diag['issues']) if diag['issues'] else 'none'}")
    print(f"median_max_raw_force={diag['median_max_raw_force']:.4g}")
    print(f"median_contact_fraction={diag['median_contact_fraction']:.4g}")


if __name__ == "__main__":
    main()
