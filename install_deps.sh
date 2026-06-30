#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"

echo "Using Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

if [[ -f .gitmodules ]]; then
  git submodule update --init --recursive
fi

if [[ ! -f ./third_party/lerobot/pyproject.toml ]]; then
  echo "Missing third_party/lerobot. Initialize submodules first." >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  echo "pip is missing for this Python. Trying to bootstrap it with ensurepip..."
  if ! "$PYTHON_BIN" -m ensurepip --upgrade; then
    echo "Failed to bootstrap pip with ensurepip." >&2
    echo "If you are using conda, run: conda install -n \"${CONDA_DEFAULT_ENV:-romi-teleop}\" pip" >&2
    echo "Then re-run: bash install_deps.sh" >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

if command -v conda >/dev/null 2>&1; then
  conda install -y -c conda-forge ffmpeg=7.1.1
else
  echo "conda not found; skipping conda ffmpeg install." >&2
  echo "Install ffmpeg manually if LeRobot video decoding fails." >&2
fi

"$PYTHON_BIN" -m pip install -r requirements.txt

echo "Installation complete."
echo "Sanity checks:"
echo "  $PYTHON_BIN -c 'import lerobot; print(lerobot.__file__)'"
echo "  $PYTHON_BIN -c 'import mujoco, dm_control; print(\"mujoco ok\")'"
echo "  lerobot-info"
echo "  lerobot-find-port"
