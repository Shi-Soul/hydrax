#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repository_root}"
export EQX_ON_ERROR=nan
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m examples.to.timeshifting.experiment "$@"
fi
exec uv run --frozen python -m examples.to.timeshifting.experiment "$@"
