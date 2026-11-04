#!/usr/bin/env bash
set -euo pipefail
PIPELINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$PIPELINE_DIR/runs/demo}"
INPUT_FILE="${INPUT_FILE:-$PIPELINE_DIR/example.jsonl}"
mode="${1:-run}"
shift "$(( $# > 0 ? 1 : 0 ))"
case "$mode" in
 check) extra=(--check-only) ;;
 run) extra=() ;;
 *) echo 'Usage: bash run.sh {check|run} [pipeline arguments]' >&2; exit 2 ;;
esac
exec "$PYTHON_BIN" "$PIPELINE_DIR/pipeline.py" --input "$INPUT_FILE" --output "$OUTPUT_DIR" "${extra[@]}" "$@"
