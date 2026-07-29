#!/usr/bin/env bash
#SBATCH --job-name=eval-qwen3-0p6b-sft-5gpu
#SBATCH --partition=cluster
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/eval-qwen3-0p6b-sft-5gpu-%j.out
#SBATCH --error=outputs/slurm/eval-qwen3-0p6b-sft-5gpu-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"
mkdir -p outputs/slurm runs

for lib_dir in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu; do
  if [[ -d "${lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
  fi
done

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
VLLM_BIN="${VLLM_BIN:-${VENV_DIR}/bin/vllm}"
LLMPIDTUNER_BIN="${LLMPIDTUNER_BIN:-${VENV_DIR}/bin/llmpidtuner}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/outputs/sft/qwen3_0p6b_pid}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b-sft}"
MODEL_LABEL="${MODEL_LABEL:-Qwen3-0.6B-SFT}"
PORT_BASE="${PORT_BASE:-8100}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"

FIRST_CASE="${FIRST_CASE:-cases/eval/first_order_100_sft_qwen3_0p6b.yaml}"
SECOND_CASE="${SECOND_CASE:-cases/eval/second_order_100_sft_qwen3_0p6b.yaml}"
FIRST_RUN="${FIRST_RUN:-runs/first_order_100_sft_qwen3_0p6b}"
SECOND_RUN="${SECOND_RUN:-runs/second_order_100_sft_qwen3_0p6b}"
SUMMARY_STEM="${SUMMARY_STEM:-sft_qwen3_0p6b}"
PROFILE_BASE_URL_ENV="${PROFILE_BASE_URL_ENV:-VLLM_QWEN3_0_6B_SFT_BASE_URL}"
EVAL_LOG_TAG="${EVAL_LOG_TAG:-qwen3-0p6b-sft}"

for exe in "${VLLM_BIN}" "${LLMPIDTUNER_BIN}"; do
  if [[ ! -x "${exe}" ]]; then
    echo "Required executable is not available: ${exe}" >&2
    exit 2
  fi
done
if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ -n "${LORA_MODULE_PATH:-}" ]] && [[ ! -f "${LORA_MODULE_PATH}/adapter_config.json" ]]; then
  echo "LoRA adapter is incomplete: ${LORA_MODULE_PATH}" >&2
  exit 2
fi

# Slurm normally provides exactly the five GPUs assigned to this job.
echo "Slurm CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
GROUP_SPECS=("1-20" "21-40" "41-60" "61-80" "81-100")

export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_TEMPERATURE="${VLLM_TEMPERATURE:-0.1}"
export VLLM_TOP_P="${VLLM_TOP_P:-0.1}"
export VLLM_ENABLE_THINKING="false"

cleanup() {
  if [[ -n "${VLLM_SRUN_PID:-}" ]]; then
    kill "${VLLM_SRUN_PID}" 2>/dev/null || true
    wait "${VLLM_SRUN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

declare -a RUN_PIDS=()

# Let Slurm bind one visible GPU to each task. The helper staggers startup,
# avoiding vLLM 0.10's free-memory profiling race during dummy-forward setup.
export VLLM_BIN MODEL_PATH SERVED_MODEL_NAME VLLM_GPU_MEMORY_UTILIZATION
export LORA_MODULE_NAME="${LORA_MODULE_NAME:-}" LORA_MODULE_PATH="${LORA_MODULE_PATH:-}"
export VLLM_PORT_BASE="${PORT_BASE}"
export VLLM_STARTUP_STAGGER_SECONDS="${VLLM_STARTUP_STAGGER_SECONDS:-90}"
export VLLM_LOG_PREFIX="outputs/slurm/vllm-${EVAL_LOG_TAG}-${SLURM_JOB_ID}-worker"
echo "Launching five Slurm GPU-bound vLLM tasks with ${VLLM_STARTUP_STAGGER_SECONDS}s startup stagger."
srun --nodes=1 --ntasks=5 --gpus-per-task=1 --cpus-per-task=8 --kill-on-bad-exit=1 \
  bash scripts/slurm/launch_vllm_eval_worker.sh &
VLLM_SRUN_PID=$!

for worker in 0 1 2 3 4; do
  port="$((PORT_BASE + worker))"
  log="${VLLM_LOG_PREFIX}${worker}.log"
  ready=0
  for _ in $(seq 1 240); do
    if curl -fsS -H "Authorization: Bearer ${VLLM_API_KEY}" "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "${VLLM_SRUN_PID}" 2>/dev/null; then
      echo "Slurm vLLM task step exited before worker ${worker} became ready." >&2
      tail -80 "${log}" >&2 || true
      exit 3
    fi
    sleep 5
  done
  if [[ "${ready}" != "1" ]]; then
    echo "Timed out waiting for vLLM worker ${worker}." >&2
    tail -80 "${log}" >&2 || true
    exit 4
  fi
done

for worker in 0 1 2 3 4; do
  port="$((PORT_BASE + worker))"
  groups="${GROUP_SPECS[$worker]}"
  worker_log="outputs/slurm/eval-${EVAL_LOG_TAG}-${SLURM_JOB_ID}-worker${worker}.log"
  (
    export "${PROFILE_BASE_URL_ENV}=http://127.0.0.1:${port}/v1"
    export VLLM_QWEN3_0_6B_SFT_API_KEY="${VLLM_API_KEY}"
    "${LLMPIDTUNER_BIN}" run "${FIRST_CASE}" --mode llm --resume --groups "${groups}" --no-batch-excel
    "${LLMPIDTUNER_BIN}" run "${SECOND_CASE}" --mode llm --resume --groups "${groups}" --no-batch-excel
  ) > "${worker_log}" 2>&1 &
  RUN_PIDS[$worker]=$!
done

for worker in 0 1 2 3 4; do
  wait "${RUN_PIDS[$worker]}"
done

"${LLMPIDTUNER_BIN}" collect-batch "${FIRST_CASE}"
"${LLMPIDTUNER_BIN}" collect-batch "${SECOND_CASE}"

"${LLMPIDTUNER_BIN}" analyze-batches \
  "${FIRST_RUN}" \
  --labels "${MODEL_LABEL}" \
  --output "runs/first_order_100_${SUMMARY_STEM}_summary.png" \
  --summary-csv "runs/first_order_100_${SUMMARY_STEM}_summary.csv" \
  --case-csv "runs/first_order_100_${SUMMARY_STEM}_cases.csv"

"${LLMPIDTUNER_BIN}" analyze-batches \
  "${SECOND_RUN}" \
  --labels "${MODEL_LABEL}" \
  --output "runs/second_order_100_${SUMMARY_STEM}_summary.png" \
  --summary-csv "runs/second_order_100_${SUMMARY_STEM}_summary.csv" \
  --case-csv "runs/second_order_100_${SUMMARY_STEM}_cases.csv"

echo "Five-GPU ${MODEL_LABEL} evaluation complete."
