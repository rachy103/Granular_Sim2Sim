#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_DIR="${ENV_DIR:-.venv}"
EXTRAS="${EXTRAS:-mujoco,newton,learning,dev}"
MENAGERIE_REPO="${MENAGERIE_REPO:-https://github.com/google-deepmind/mujoco_menagerie.git}"
MENAGERIE_REF="${MENAGERIE_REF:-b846dd12bc459d776cccb3dee0b1d02acbf7a9c7}"
MENAGERIE_DIR="${MENAGERIE_DIR:-mujoco_menagerie}"
LOCK_FILE="${LOCK_FILE:-constraints/reference-linux-py310-cu128.txt}"
RUN_TESTS=1
RUN_SMOKE=0

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --env-dir PATH       Python virtualenv path (default: .venv)
  --extras LIST        pip extras to install (default: mujoco,newton,learning,dev)
  --lite               install only the compact Warp demo dependencies
  --locked             constrain package versions to the tested reference lock
  --no-menagerie       skip downloading google-deepmind/mujoco_menagerie
  --force-menagerie    re-clone mujoco_menagerie if it already exists
  --no-tests           skip pytest after installation
  --smoke              run a short density-render smoke demo after install
  -h, --help           show this help

Examples:
  ./install.sh
  ./install.sh --locked
  ./install.sh --lite --smoke
  ENV_DIR=/tmp/granular-venv ./install.sh --extras mujoco,newton,learning,dev
EOF
}

FORCE_MENAGERIE=0
SKIP_MENAGERIE=0
USE_LOCK=0

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
    --locked)
      USE_LOCK=1
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
PIP_CONSTRAINT_ARGS=()
PIP_INDEX_ARGS=()
if [[ "$USE_LOCK" -eq 1 ]]; then
  if [[ ! -f "$LOCK_FILE" ]]; then
    echo "Lock file not found: $LOCK_FILE" >&2
    exit 1
  fi
  PIP_CONSTRAINT_ARGS=(-c "$LOCK_FILE")
  if [[ ",$EXTRAS," == *",learning,"* ]]; then
    PIP_INDEX_ARGS=(--extra-index-url https://download.pytorch.org/whl/cu128)
  fi
fi
if [[ -n "$EXTRAS" ]]; then
  python -m pip install "${PIP_CONSTRAINT_ARGS[@]}" "${PIP_INDEX_ARGS[@]}" -e ".[$EXTRAS]"
else
  python -m pip install "${PIP_CONSTRAINT_ARGS[@]}" -e .
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
    git clone --filter=blob:none --no-checkout "$MENAGERIE_REPO" "$MENAGERIE_DIR"
  else
    git -C "$MENAGERIE_DIR" remote set-url origin "$MENAGERIE_REPO"
  fi
  git -C "$MENAGERIE_DIR" fetch --depth 1 origin "$MENAGERIE_REF"
  git -C "$MENAGERIE_DIR" checkout --detach "$MENAGERIE_REF"
fi

python - <<'PY'
import importlib

required = ["numpy", "cv2", "warp", "granular_mpm"]
optional = ["mujoco", "newton", "torch"]

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
  python scripts/run_3d_density_render_demo.py \
    --device "${GRANULAR_SMOKE_DEVICE:-cpu}" \
    --frames 6 \
    --substeps 4 \
    --out outputs/smoke_density_render
fi

cat <<EOF

Install complete.

Activate this environment with:
  source $ENV_DIR/bin/activate

Reproduce the main bridge demo with:
  python scripts/run_mujoco_newton_mpm_bridge.py --config configs/newton_bridge_heightfield.json

Package generated demo artifacts for Drive or GitHub Release with:
  python scripts/package_demo_artifacts.py
EOF
