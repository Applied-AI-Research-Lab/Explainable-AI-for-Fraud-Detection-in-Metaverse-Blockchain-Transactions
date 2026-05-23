#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
	echo "Using active virtual environment: ${VIRTUAL_ENV}"
elif [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
	source "${ROOT_DIR}/.venv/bin/activate"
elif [[ -f "${ROOT_DIR}/venv/bin/activate" ]]; then
	source "${ROOT_DIR}/venv/bin/activate"
else
	VENV_DIR="${ROOT_DIR}/.venv"
	python3 -m venv "${VENV_DIR}"
	source "${VENV_DIR}/bin/activate"
fi

python -m pip install --upgrade pip
pip install -r "${ROOT_DIR}/requirements.txt"

echo "Environment ready. Active env: ${VIRTUAL_ENV:-system}"
