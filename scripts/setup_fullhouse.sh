#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python -m venv "$project_dir/../.venvs/fullhouse"
"$project_dir/../.venvs/fullhouse/bin/python" -m pip install -r "$project_dir/configs/fullhouse-inference-requirements.txt"
"$project_dir/../.venvs/fullhouse/bin/python" "$project_dir/scripts/prepare_fullhouse.py"
if [[ "${1:-}" == "--training" ]]; then
  "$project_dir/../.venvs/fullhouse/bin/python" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
  "$project_dir/../.venvs/fullhouse/bin/python" -m pip install numba==0.61.2 scipy==1.15.3
fi
