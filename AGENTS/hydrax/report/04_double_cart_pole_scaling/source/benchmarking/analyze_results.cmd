#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repository_root}"
exec .venv/bin/python -m examples.to.benchmarking.analyze_results "$@"
