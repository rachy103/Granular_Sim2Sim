#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="full"
RUN_BRIDGE=1
RUN_PACKAGE=1
RUN_TESTS=1

usage() {
  cat <<'EOF'
Usage: ./scripts/reproduce_demo_bundle.sh [options]

Options:
  --smoke          run fast checks and a tiny density render
  --full           run the default publishable demo sequence (default)
  --skip-bridge    skip the MuJoCo-Newton bridge render
  --no-package     do not build dist/granular-robot-demo-artifacts-<sha>.zip
  --no-tests       skip pytest
  -h, --help       show this help

The full run regenerates the default density render, 3D blade render, and
MuJoCo-Newton bridge render, then packages shareable videos/logs into dist/.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      MODE="smoke"
      RUN_PACKAGE=0
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --skip-bridge)
      RUN_BRIDGE=0
      shift
      ;;
    --no-package)
      RUN_PACKAGE=0
      shift
      ;;
    --no-tests)
      RUN_TESTS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ "$RUN_TESTS" -eq 1 ]]; then
  python -m pytest -q
fi

if [[ "$MODE" == "smoke" ]]; then
  python scripts/run_3d_density_render_demo.py \
    --device "${GRANULAR_SMOKE_DEVICE:-cpu}" \
    --frames 6 \
    --substeps 4 \
    --out outputs/smoke_density_render
  if [[ "$RUN_BRIDGE" -eq 1 ]]; then
    if [[ ! -d "mujoco_menagerie/franka_emika_panda" ]]; then
      echo "Skipping bridge smoke: mujoco_menagerie/franka_emika_panda is missing."
      echo "Run ./install.sh first, or pass --skip-bridge."
    else
      python scripts/run_mujoco_newton_mpm_bridge.py \
        --config configs/newton_bridge_heightfield.json \
        --voxel-size 0.060 \
        --particles-per-cell 1.2 \
        --frames 4 \
        --steps-per-frame 1 \
        --render-blur 1.8
    fi
  fi
  exit 0
fi

python scripts/run_3d_density_render_demo.py --device "${GRANULAR_DEVICE:-cuda:0}"
python scripts/run_3d_blade_demo.py --config configs/sand3d_blade_demo.json

if [[ "$RUN_BRIDGE" -eq 1 ]]; then
  if [[ ! -d "mujoco_menagerie/franka_emika_panda" ]]; then
    echo "mujoco_menagerie/franka_emika_panda is missing. Run ./install.sh first." >&2
    exit 1
  fi
  python scripts/run_mujoco_newton_mpm_bridge.py --config configs/newton_bridge_heightfield.json
fi

if [[ "$RUN_PACKAGE" -eq 1 ]]; then
  python scripts/package_demo_artifacts.py
fi
