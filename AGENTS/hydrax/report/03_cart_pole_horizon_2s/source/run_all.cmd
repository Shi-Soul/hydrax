#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repository_root}"
export EQX_ON_ERROR=nan
exec .venv/bin/python examples/to/open_loop_benchmark.py "$@"
