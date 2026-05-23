#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
    source "${ROOT_DIR}/.venv/bin/activate"
  elif [[ -f "${ROOT_DIR}/venv/bin/activate" ]]; then
    source "${ROOT_DIR}/venv/bin/activate"
  else
    echo "No active virtual environment found. Activate one, or run scripts/install.sh first."
    exit 1
  fi
fi

python -m src.pipeline \
  --data-path "${ROOT_DIR}/Dataset/metaverse_transactions_dataset.csv" \
  --output-dir "${ROOT_DIR}/outputs" \
  --test-size 0.2 \
  --random-state 42 \
  --llm-mode heuristic \
  --llm-max-samples 1000
