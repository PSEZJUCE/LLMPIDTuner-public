#!/usr/bin/env bash
# Online GRPO on five A100 GPUs, with prompts sampled from the shared simulator.
# The policy and frozen reference both start from the configured SFT checkpoint.
#SBATCH --job-name=qwen3-0p6b-grpo
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --output=outputs/slurm/qwen3-0p6b-grpo-%j.out
#SBATCH --error=outputs/slurm/qwen3-0p6b-grpo-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"
mkdir -p outputs/slurm outputs/grpo

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${VENV_DIR}/bin/accelerate}"
for exe in "${PYTHON_BIN}" "${ACCELERATE_BIN}"; do
  if [[ ! -x "${exe}" ]]; then
    echo "Required executable is not available: ${exe}" >&2
    echo "Run uv sync --extra training --extra training-stack before submitting." >&2
    exit 2
  fi
done


GRPO_CONFIG="${GRPO_CONFIG:-configs/grpo/qwen3_0p6b_pid.yaml}"
"${PYTHON_BIN}" - "${GRPO_CONFIG}" <<'PY'
import sys
from pathlib import Path

from llmpidtuner.training.config import load_grpo_train_config
from llmpidtuner.training.grpo_control import resolve_resume_checkpoint

config = load_grpo_train_config(sys.argv[1])
output_dir = Path(config.output_dir)
model_dir = Path(config.model_name_or_path)
required_model_files = (
    model_dir / "config.json",
    model_dir / "training_manifest.json",
)
missing_model_files = [str(path) for path in required_model_files if not path.is_file()]
if missing_model_files:
    raise SystemExit(
        "Configured SFT model is incomplete. Missing: " + ", ".join(missing_model_files)
    )

resume_checkpoint = resolve_resume_checkpoint(config)
if (
    resume_checkpoint is None
    and output_dir.exists()
    and (not output_dir.is_dir() or any(output_dir.iterdir()))
):
    raise SystemExit(
        f"GRPO output directory is not empty: {output_dir}. "
        "Move it for a new run or configure resume_from_checkpoint."
    )
print(f"Configured SFT model preflight passed: {model_dir}")
print(f"GRPO output preflight passed: {output_dir}; resume={resume_checkpoint}")
PY

export NPROC_PER_NODE="${NPROC_PER_NODE:-5}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((30500 + SLURM_JOB_ID % 1000))}"

"${PYTHON_BIN}" - <<'PY'
import os
import torch

expected = int(os.environ["NPROC_PER_NODE"])
visible = torch.cuda.device_count()
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, visible_gpus={visible}")
if not torch.cuda.is_available() or visible < expected:
    raise SystemExit(f"Expected at least {expected} visible CUDA devices, found {visible}.")
PY

"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes="${NPROC_PER_NODE}" \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_port="${MAIN_PROCESS_PORT}" \
  "${VENV_DIR}/bin/llmpidtuner" train-grpo "${GRPO_CONFIG}"
