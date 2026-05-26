#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_DIR="${ENV_DIR:-.venv}"
EXTRAS="${EXTRAS:-mujoco,newton,dev}"
MENAGERIE_REPO="${MENAGERIE_REPO:-https://github.com/google-deepmind/mujoco_menagerie.git}"
MENAGERIE_DIR="${MENAGERIE_DIR:-mujoco_menagerie}"
RUN_TESTS=1
RUN_SMOKE=0

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --env-dir PATH       Python virtualenv path (default: .venv)
  --extras LIST        pip extras to install (default: mujoco,newton,dev)
  --lite               install only the compact Warp demo dependencies
  --no-menagerie       skip downloading google-deepmind/mujoco_menagerie
  --force-menagerie    re-clone mujoco_menagerie if it already exists
  --no-tests           skip pytest after installation
  --smoke              run a short density-render smoke demo after install
  -h, --help           show this help

Examples:
  ./install.sh
  ./install.sh --lite --smoke
  ENV_DIR=/tmp/granular-venv ./install.sh --extras mujoco,newton,dev
EOF
}

FORCE_MENAGERIE=0
SKIP_MENAGERIE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-dir)
      ENV_DIR="$2"
      shift 2
      ;;
    --extras)
      EXTRAS="$2"
      shift 2
      ;;
    --lite)
      EXTRAS=""
      shift
      ;;
    --no-menagerie)
      SKIP_MENAGERIE=1
      shift
      ;;
    --force-menagerie)
      FORCE_MENAGERIE=1
      shift
      ;;
    --no-tests)
      RUN_TESTS=0
      shift
      ;;
    --smoke)
      RUN_SMOKE=1
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

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 1
fi

if [[ ! -d "$ENV_DIR" ]]; then
  python3 -m venv "$ENV_DIR"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
if [[ -n "$EXTRAS" ]]; then
  python -m pip install -e ".[$EXTRAS]"
else
  python -m pip install -e .
fi

if [[ "$SKIP_MENAGERIE" -eq 0 ]]; then
  if [[ "$FORCE_MENAGERIE" -eq 1 && -e "$MENAGERIE_DIR" ]]; then
    MENAGERIE_TARGET="$(cd "$(dirname "$MENAGERIE_DIR")" && pwd)/$(basename "$MENAGERIE_DIR")"
    case "$MENAGERIE_TARGET" in
      "$ROOT"/*) ;;
      *)
        echo "Refusing to remove MENAGERIE_DIR outside repo: $MENAGERIE_TARGET" >&2
        exit 1
        ;;
    esac
    rm -rf "$MENAGERIE_DIR"
  fi
  if [[ ! -d "$MENAGERIE_DIR/.git" ]]; then
    git clone --depth 1 "$MENAGERIE_REPO" "$MENAGERIE_DIR"
  else
    git -C "$MENAGERIE_DIR" fetch --depth 1 origin main || true
  fi
fi

python - <<'PY'
import importlib

required = ["numpy", "cv2", "warp", "granular_mpm"]
optional = ["mujoco", "newton"]

for name in required:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "ok")
    print(f"{name}: {version}")

for name in optional:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        print(f"{name}: unavailable ({exc})")
    else:
        version = getattr(mod, "__version__", "ok")
        print(f"{name}: {version}")
PY

if [[ "$RUN_TESTS" -eq 1 ]]; then
  python -m pytest -q
fi

if [[ "$RUN_SMOKE" -eq 1 ]]; then
  python scripts/run_3d_density_render_demo.py --frames 6 --substeps 4 --out outputs/smoke_density_render
fi

cat <<EOF

Install complete.

Activate this environment with:
  source $ENV_DIR/bin/activate

Reproduce the main bridge demo with:
  python scripts/run_mujoco_newton_mpm_bridge.py --voxel-size 0.032 --particles-per-cell 3.0 --sand-render-mode heightfield --render-blur 2.4 --alpha-cutoff 0.060 --alpha-gain 0.48

Package generated demo artifacts for Drive or GitHub Release with:
  python scripts/package_demo_artifacts.py
EOF
