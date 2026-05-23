#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hf_model_path_or_name> [llm_max_samples] [llm_max_retries]"
  exit 1
fi

MODEL_PATH="$1"
LLM_MAX_SAMPLES="${2:-500}"
LLM_MAX_RETRIES="${3:-3}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer a local cached snapshot for the known LLaMA model to avoid re-downloading.
if [[ "${MODEL_PATH}" == "unsloth/Llama-3.3-70B-Instruct" ]]; then
  LOCAL_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots"
  if [[ -d "${LOCAL_CACHE_ROOT}" ]]; then
    LOCAL_SNAPSHOT="$(find "${LOCAL_CACHE_ROOT}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [[ -n "${LOCAL_SNAPSHOT}" ]]; then
      if compgen -G "${LOCAL_SNAPSHOT}/model-*.safetensors" > /dev/null && [[ -f "${LOCAL_SNAPSHOT}/tokenizer.json" || -f "${LOCAL_SNAPSHOT}/tokenizer_config.json" ]]; then
        echo "Using local cached model snapshot: ${LOCAL_SNAPSHOT}"
        MODEL_PATH="${LOCAL_SNAPSHOT}"
      else
        echo "Found local cache snapshot but it appears incomplete; using Hugging Face model id instead."
      fi
    fi
  fi
fi

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
  --max-rows 3000 \
  --output-dir "${ROOT_DIR}/outputs_llm" \
  --test-size 0.2 \
  --random-state 42 \
  --llm-mode hf_local \
  --llm-model "${MODEL_PATH}" \
  --llm-backend unsloth \
  --llm-load-in-4bit \
  --llm-max-seq-length 2048 \
  --llm-dtype auto \
  --llm-max-samples "${LLM_MAX_SAMPLES}" \
  --llm-max-retries "${LLM_MAX_RETRIES}" \
  --llm-max-new-tokens 96 \
  --shap-background-size 200
