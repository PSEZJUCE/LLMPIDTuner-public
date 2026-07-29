#!/usr/bin/env bash
# Generate 20k FOPDT + 20k SOPDT SFT messages on CPU.
#SBATCH --job-name=pid-data-40k
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/pid-data-40k-%j.out
#SBATCH --error=outputs/slurm/pid-data-40k-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"
mkdir -p outputs/slurm data/sft

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
LLMPIDTUNER_BIN="${LLMPIDTUNER_BIN:-${VENV_DIR}/bin/llmpidtuner}"
if [[ ! -x "${LLMPIDTUNER_BIN}" ]]; then
  echo "LLMPIDTUNER_BIN is not executable: ${LLMPIDTUNER_BIN}" >&2
  exit 2
fi

SFT_DATA="data/sft/pid_sft_messages_40000.jsonl"
if [[ "${OVERWRITE_DATA:-0}" != "1" ]] && [[ -e "${SFT_DATA}" ]]; then
  echo "Refusing to overwrite existing SFT data: ${SFT_DATA}" >&2
  echo "Archive it first, or submit with OVERWRITE_DATA=1 intentionally." >&2
  exit 3
fi

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
"${LLMPIDTUNER_BIN}" make-sft-data configs/data/pid_sft_messages_40k.yaml
