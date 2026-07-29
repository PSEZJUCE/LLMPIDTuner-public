#!/usr/bin/env bash
#SBATCH --job-name=eval-qwen3-0p6b-base
#SBATCH --partition=cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/eval-qwen3-0p6b-base-%j.out
#SBATCH --error=outputs/slurm/eval-qwen3-0p6b-base-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"
mkdir -p outputs/slurm runs

if [[ -n "${FORCE_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${FORCE_CUDA_VISIBLE_DEVICES}"
fi
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
fi

for lib_dir in   /usr/lib/x86_64-linux-gnu   /lib/x86_64-linux-gnu; do
  if [[ -d "${lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
  fi
done

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
VLLM_BIN="${VLLM_BIN:-${VENV_DIR}/bin/vllm}"
LLMPIDTUNER_BIN="${LLMPIDTUNER_BIN:-${VENV_DIR}/bin/llmpidtuner}"
for exe in "${VLLM_BIN}" "${LLMPIDTUNER_BIN}"; do
  if [[ ! -x "${exe}" ]]; then
    echo "Required executable is not available: ${exe}" >&2
    echo "Run uv sync --extra training --extra training-stack on the server before submitting, or override VENV_DIR/VLLM_BIN/LLMPIDTUNER_BIN." >&2
    exit 2
  fi
done
echo "Using vLLM: ${VLLM_BIN}"
echo "Using LLMPIDTuner: ${LLMPIDTUNER_BIN}"
echo "Host: $(hostname)"
"${VENV_DIR}/bin/python" --version
"${VENV_DIR}/bin/python" - <<'PY'
import sys
if sys.version_info >= (3, 13):
    raise SystemExit(
        "Python >=3.13 is not supported for the server vLLM/PyTorch stack. "
        "Rebuild with: uv sync --extra training --extra training-stack"
    )
PY
"${VENV_DIR}/bin/python" - <<'PY'
import numpy
import torch
import vllm
print(f"Import smoke ok: numpy={numpy.__version__}, torch={torch.__version__}, vllm={vllm.__version__}")
PY

# Submit with a specific base-model path if the default is not correct:
#   sbatch --export=ALL,MODEL_PATH=/path/to/Qwen3-0.6B scripts/slurm/eval_qwen3_0p6b_base_all.sbatch
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b-base}"
PORT="${PORT:-8000}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export VLLM_TEMPERATURE="${VLLM_TEMPERATURE:-0.1}"
export VLLM_TOP_P="${VLLM_TOP_P:-0.1}"
export VLLM_ENABLE_THINKING="false"
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  echo "Submit with: sbatch --export=ALL,MODEL_PATH=/path/to/Qwen3-0.6B $0" >&2
  exit 2
fi

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting vLLM for ${SERVED_MODEL_NAME} from ${MODEL_PATH} on port ${PORT}"
"${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --trust-remote-code \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  > "outputs/slurm/vllm-qwen3-0p6b-base-${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM health endpoint..."
READY=0
for _ in $(seq 1 180); do
  if curl -fsS -H "Authorization: Bearer ${VLLM_API_KEY}" "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready. Last log lines:" >&2
    tail -80 "outputs/slurm/vllm-qwen3-0p6b-base-${SLURM_JOB_ID}.log" >&2 || true
    exit 3
  fi
  sleep 5
done

if [[ "${READY}" != "1" ]]; then
  echo "Timed out waiting for vLLM." >&2
  tail -80 "outputs/slurm/vllm-qwen3-0p6b-base-${SLURM_JOB_ID}.log" >&2 || true
  exit 4
fi

curl -fsS -H "Authorization: Bearer ${VLLM_API_KEY}" "http://127.0.0.1:${PORT}/v1/models"

"${LLMPIDTUNER_BIN}" run cases/eval/first_order_100_base_qwen3_0p6b.yaml --mode llm --resume
"${LLMPIDTUNER_BIN}" run cases/eval/second_order_100_base_qwen3_0p6b.yaml --mode llm --resume

"${LLMPIDTUNER_BIN}" analyze-batches \
  runs/first_order_100_base_qwen3_0p6b \
  --labels Qwen3-0.6B-Base \
  --output runs/first_order_100_base_qwen3_0p6b_summary.png \
  --summary-csv runs/first_order_100_base_qwen3_0p6b_summary.csv \
  --case-csv runs/first_order_100_base_qwen3_0p6b_cases.csv

"${LLMPIDTUNER_BIN}" analyze-batches \
  runs/second_order_100_base_qwen3_0p6b \
  --labels Qwen3-0.6B-Base \
  --output runs/second_order_100_base_qwen3_0p6b_summary.png \
  --summary-csv runs/second_order_100_base_qwen3_0p6b_summary.csv \
  --case-csv runs/second_order_100_base_qwen3_0p6b_cases.csv

echo "Evaluation complete."
