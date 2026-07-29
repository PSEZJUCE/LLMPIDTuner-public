#!/usr/bin/env bash
#SBATCH --job-name=eval-base-prompt-ablation
#SBATCH --partition=cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/eval-base-prompt-ablation-%j.out
#SBATCH --error=outputs/slurm/eval-base-prompt-ablation-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"
mkdir -p outputs/slurm runs

if [[ -n "${FORCE_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${FORCE_CUDA_VISIBLE_DEVICES}"
fi
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
VLLM_BIN="${VLLM_BIN:-${VENV_DIR}/bin/vllm}"
LLMPIDTUNER_BIN="${LLMPIDTUNER_BIN:-${VENV_DIR}/bin/llmpidtuner}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
for exe in "${VLLM_BIN}" "${LLMPIDTUNER_BIN}" "${PYTHON_BIN}"; do
  if [[ ! -x "${exe}" ]]; then
    echo "Required executable is not available: ${exe}" >&2
    echo "Run uv sync --extra training --extra training-stack before submitting." >&2
    exit 2
  fi
done

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b-base}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-10}"
EVAL_SCOPE="${EVAL_SCOPE:-all}"
REQUEST_SEED="${REQUEST_SEED:-42}"
VLLM_ENGINE_SEED="${VLLM_ENGINE_SEED:-42}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"

if [[ "${REQUEST_SEED}" != "42" ]]; then
  echo "The committed Base evaluation YAML files require REQUEST_SEED=42." >&2
  exit 2
fi
if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 2
fi

FULL_FIRST="cases/eval/first_order_100_base_qwen3_0p6b.yaml"
FULL_SECOND="cases/eval/second_order_100_base_qwen3_0p6b.yaml"
KPI3_FIRST="cases/eval/first_order_100_base_qwen3_0p6b_kpi3.yaml"
KPI3_SECOND="cases/eval/second_order_100_base_qwen3_0p6b_kpi3.yaml"
NUMERIC8_FIRST="cases/eval/first_order_100_base_qwen3_0p6b_numeric8.yaml"
NUMERIC8_SECOND="cases/eval/second_order_100_base_qwen3_0p6b_numeric8.yaml"

case "${EVAL_SCOPE}" in
  full-first-order)
    CASE_FILES=("${FULL_FIRST}")
    ;;
  ablation-first-order)
    CASE_FILES=("${KPI3_FIRST}" "${NUMERIC8_FIRST}")
    ;;
  ablation-second-order)
    CASE_FILES=("${KPI3_SECOND}" "${NUMERIC8_SECOND}")
    ;;
  ablation-only)
    CASE_FILES=(
      "${KPI3_FIRST}"
      "${KPI3_SECOND}"
      "${NUMERIC8_FIRST}"
      "${NUMERIC8_SECOND}"
    )
    ;;
  all)
    CASE_FILES=(
      "${FULL_FIRST}"
      "${FULL_SECOND}"
      "${KPI3_FIRST}"
      "${KPI3_SECOND}"
      "${NUMERIC8_FIRST}"
      "${NUMERIC8_SECOND}"
    )
    ;;
  *)
    echo "Unsupported EVAL_SCOPE=${EVAL_SCOPE}." >&2
    echo "Use full-first-order, ablation-first-order, ablation-second-order, ablation-only, or all." >&2
    exit 2
    ;;
esac

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Model: ${MODEL_PATH}"
echo "Scope: ${EVAL_SCOPE}"
echo "Workers: ${WORKERS}"
echo "Request seed: ${REQUEST_SEED}"
echo "vLLM engine seed: ${VLLM_ENGINE_SEED}"
"${PYTHON_BIN}" --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
fi

# Refuse to resume into a directory produced with a different model or prompt protocol.
"${PYTHON_BIN}" - "${CASE_FILES[@]}" <<'PY'
from pathlib import Path
import sys
import yaml


def read_metadata(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


for case_arg in sys.argv[1:]:
    case_path = Path(case_arg)
    config = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    output_root = Path(config.get("output_dir", "runs")) / case_path.stem
    if not output_root.exists():
        continue

    expected_variant = str(config.get("prompt_variant", "full"))
    expected_protocol = str(config["demonstration"]["protocol_id"])
    expected = {
        "profile": "vLLM_qwen3-0.6b-base",
        "model": "qwen3-0.6b-base",
        "temperature": "0.1",
        "top_p": "0.1",
        "enable_thinking": "False",
        "thinking_request_body": (
            '{"chat_template_kwargs": {"enable_thinking": false}}'
        ),
        "seed": "42",
        "prompt_variant": expected_variant,
    }

    completed = sorted(output_root.glob("group_*/run_status.yaml"))
    root_metadata = output_root / "llm_metadata.txt"
    if completed and not root_metadata.exists():
        raise SystemExit(f"Cannot safely resume {output_root}: llm_metadata.txt is missing")
    if root_metadata.exists():
        actual = read_metadata(root_metadata)
        mismatches = {
            key: (actual.get(key), value)
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise SystemExit(
                f"Cannot safely resume {output_root}: LLM metadata mismatch {mismatches}"
            )

    for status_path in completed:
        group_dir = status_path.parent
        demo_path = group_dir / "demonstration_metadata.yaml"
        if not demo_path.exists():
            raise SystemExit(
                f"Cannot safely resume {group_dir}: demonstration_metadata.yaml is missing"
            )
        demo = yaml.safe_load(demo_path.read_text(encoding="utf-8"))
        if str(demo.get("demonstration_protocol")) != expected_protocol:
            raise SystemExit(
                f"Cannot safely resume {group_dir}: demonstration protocol "
                f"{demo.get('demonstration_protocol')!r} != {expected_protocol!r}"
            )

print("Resume protocol preflight passed.")
PY

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export VLLM_ENABLE_THINKING="false"
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

VLLM_LOG="outputs/slurm/vllm-base-prompt-ablation-${SLURM_JOB_ID}.log"
echo "Starting vLLM on port ${PORT}; log: ${VLLM_LOG}"
"${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --trust-remote-code \
  --seed "${VLLM_ENGINE_SEED}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  > "${VLLM_LOG}" 2>&1 &
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
    tail -80 "${VLLM_LOG}" >&2 || true
    exit 3
  fi
  sleep 5
done
if [[ "${READY}" != "1" ]]; then
  echo "Timed out waiting for vLLM." >&2
  tail -80 "${VLLM_LOG}" >&2 || true
  exit 4
fi

"${LLMPIDTUNER_BIN}" run-api-parallel \
  --workers "${WORKERS}" \
  --resume \
  "${CASE_FILES[@]}"

echo "Base prompt evaluation complete for scope ${EVAL_SCOPE}."
