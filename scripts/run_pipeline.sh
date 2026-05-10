#!/usr/bin/env bash
set -euo pipefail

INPUT_VIDEO=${1:?Usage: scripts/run_pipeline.sh <input-video> [run-id]}
RUN_ID=${2:-}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR/src"

if [[ -n "$RUN_ID" ]]; then
  python -m video_kb.cli run --input "$INPUT_VIDEO" --run-id "$RUN_ID"
else
  python -m video_kb.cli run --input "$INPUT_VIDEO"
fi

