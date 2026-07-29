#!/usr/bin/env bash
set -euo pipefail

worker="${SLURM_PROCID:?SLURM_PROCID is required}"
port="$((VLLM_PORT_BASE + worker))"
log="${VLLM_LOG_PREFIX}${worker}.log"
stagger_seconds="${VLLM_STARTUP_STAGGER_SECONDS:-90}"

sleep "$((worker * stagger_seconds))"
{
  echo "Slurm task=${worker} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
  declare -a lora_args=()
  if [[ -n "${LORA_MODULE_PATH:-}" ]]; then
    if [[ -z "${LORA_MODULE_NAME:-}" ]]; then
      echo "LORA_MODULE_NAME is required when LORA_MODULE_PATH is set." >&2
      exit 2
    fi
    lora_args+=(--enable-lora --lora-modules "${LORA_MODULE_NAME}=${LORA_MODULE_PATH}")
  fi
  exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --trust-remote-code \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    "${lora_args[@]}"
} > "${log}" 2>&1
