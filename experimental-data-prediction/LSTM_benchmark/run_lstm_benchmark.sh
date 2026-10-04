#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODE="${1:-4}"
OUTPUT_ID="${2:-0}"
THREADS="${JULIA_THREADS:-110}"

echo "Starting LSTM benchmark at $(date)"
echo "Mode: $MODE"
echo "Output ID: $OUTPUT_ID"
echo "Julia threads: $THREADS"

julia --threads="$THREADS" \
    "$SCRIPT_DIR/lstm_benchmark.jl" \
    "$MODE" \
    "$OUTPUT_ID"

echo "LSTM benchmark completed at $(date)"