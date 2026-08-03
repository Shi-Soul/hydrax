#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repository_root}"
export EQX_ON_ERROR=nan
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

.venv/bin/python -m examples.to.benchmarking.run_grid "$@"
.venv/bin/python -m examples.to.benchmarking.analyze_results "$@"
.venv/bin/python -m examples.to.benchmarking.render_media "$@"
.venv/bin/python -m examples.to.benchmarking.build_dashboard "$@"
