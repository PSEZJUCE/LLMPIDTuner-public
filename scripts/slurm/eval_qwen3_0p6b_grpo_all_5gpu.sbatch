#!/usr/bin/env bash
# Evaluate the validation-selected GRPO LoRA adapter on the frozen paper 100+100 cases.
#SBATCH --job-name=eval-qwen3-0p6b-grpo-5gpu
#SBATCH --partition=cluster
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/eval-qwen3-0p6b-grpo-5gpu-%j.out
#SBATCH --error=outputs/slurm/eval-qwen3-0p6b-grpo-5gpu-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"
GRPO_OUTPUT="${GRPO_OUTPUT:-${PROJECT_ROOT}/outputs/grpo/qwen3_0p6b_pid}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ -z "${LORA_MODULE_PATH:-}" ]]; then
  "${PYTHON_BIN}" - "${GRPO_OUTPUT}/training_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if not manifest_path.is_file():
    raise SystemExit(f"Missing GRPO training manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("selected_model_source") != "grpo":
    raise SystemExit(
        "No GRPO checkpoint exceeded the step-0 SFT baseline. "
        "Reuse the SFT evaluation instead of labeling it as GRPO."
    )
selected = manifest_path.parent / str(manifest["selected_model_path"])
if not selected.is_dir():
    raise SystemExit(f"Selected GRPO adapter does not exist: {selected}")
print(f"Selected GRPO adapter preflight passed: {selected}")
PY
  export LORA_MODULE_PATH="${GRPO_OUTPUT}/best-validation-checkpoint"
fi

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/outputs/sft/qwen3_0p6b_pid}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b-sft}"
export LORA_MODULE_NAME="${LORA_MODULE_NAME:-qwen3-0.6b-grpo}"
export LORA_MODULE_PATH="${LORA_MODULE_PATH:-${PROJECT_ROOT}/outputs/grpo/qwen3_0p6b_pid/best-validation-checkpoint}"
export MODEL_LABEL="${MODEL_LABEL:-Qwen3-0.6B-GRPO}"
export FIRST_CASE="cases/eval/first_order_100_grpo_qwen3_0p6b.yaml"
export SECOND_CASE="cases/eval/second_order_100_grpo_qwen3_0p6b.yaml"
export FIRST_RUN="runs/first_order_100_grpo_qwen3_0p6b"
export SECOND_RUN="runs/second_order_100_grpo_qwen3_0p6b"
export SUMMARY_STEM="grpo_qwen3_0p6b"
export PROFILE_BASE_URL_ENV="VLLM_QWEN3_0_6B_GRPO_BASE_URL"
export EVAL_LOG_TAG="qwen3-0p6b-grpo"
export PORT_BASE="${PORT_BASE:-8200}"

bash scripts/slurm/eval_qwen3_0p6b_sft_all_5gpu.sbatch
