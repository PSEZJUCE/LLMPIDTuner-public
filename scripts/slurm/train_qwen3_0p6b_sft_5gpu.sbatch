#!/usr/bin/env bash
# Full-parameter SFT of Qwen3-0.6B on the five A100 GPUs allocated to the job.
# Submit only after generate_qwen3_0p6b_training_data_40k.sbatch has completed.
#SBATCH --job-name=qwen3-0p6b-sft
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/slurm/qwen3-0p6b-sft-%j.out
#SBATCH --error=outputs/slurm/qwen3-0p6b-sft-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"
mkdir -p outputs/slurm outputs/sft

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${VENV_DIR}/bin/accelerate}"
LLMPIDTUNER_BIN="${LLMPIDTUNER_BIN:-${VENV_DIR}/bin/llmpidtuner}"
for exe in "${PYTHON_BIN}" "${ACCELERATE_BIN}" "${LLMPIDTUNER_BIN}"; do
  if [[ ! -x "${exe}" ]]; then
    echo "Required executable is not available: ${exe}" >&2
    echo "Run uv sync --extra training --extra training-stack before submitting." >&2
    exit 2
  fi
done

SFT_DATA="data/sft/pid_sft_messages_40000.jsonl"
SFT_MANIFEST="${SFT_DATA}.manifest.json"
SFT_OUTPUT="outputs/sft/qwen3_0p6b_pid"
if [[ ! -f "${SFT_DATA}" ]]; then
  echo "Missing SFT data: ${SFT_DATA}. Submit the data-preparation job first." >&2
  exit 3
fi
if [[ -e "${SFT_OUTPUT}" ]]; then
  echo "SFT output already exists: ${SFT_OUTPUT}" >&2
  echo "Move the existing experiment or choose a new output_dir." >&2
  exit 4
fi

if [[ ! -f "${SFT_MANIFEST}" ]]; then
  echo "Missing SFT data manifest: ${SFT_MANIFEST}." >&2
  exit 5
fi
export SFT_DATA
export SFT_MANIFEST
export EXPECTED_SFT_ROWS="${EXPECTED_SFT_ROWS:-40000}"
export EXPECTED_CONTROL_STYLE="${EXPECTED_CONTROL_STYLE:-balanced}"
"${PYTHON_BIN}" - <<'PY'
import json
import hashlib
import os
import re
from collections import Counter
from pathlib import Path
from llmpidtuner.training.artifacts import source_tree_sha256


path = Path(os.environ["SFT_DATA"])
manifest_path = Path(os.environ["SFT_MANIFEST"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
digest = hashlib.sha256()
with path.open("rb") as data_handle:
    for chunk in iter(lambda: data_handle.read(1024 * 1024), b""):
        digest.update(chunk)
recorded_dataset = manifest.get("dataset") or {}
if digest.hexdigest() != recorded_dataset.get("sha256"):
    raise SystemExit("SFT data SHA-256 does not match its generation manifest.")
if manifest.get("source_tree_sha256") != source_tree_sha256():
    raise SystemExit(
        "Current source tree differs from the code that generated the SFT data."
    )
expected = int(os.environ["EXPECTED_SFT_ROWS"])
expected_style = os.environ["EXPECTED_CONTROL_STYLE"]
styles = Counter()
generator_style = (manifest.get("generator_config") or {}).get("control_style")
if generator_style != expected_style:
    raise SystemExit(
        f"SFT manifest control style mismatch: {generator_style!r} != {expected_style!r}."
    )
if recorded_dataset.get("control_styles") != {expected_style: expected}:
    raise SystemExit("SFT manifest does not describe a balanced-only data set.")

kinds = Counter()
plant_types = Counter()
count = 0
protocol = re.compile(
    r"^P:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?; "
    r"I:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?; "
    r"D:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages") or []
        metadata = row.get("metadata") or {}
        if not messages or messages[-1].get("role") != "assistant":
            raise SystemExit(f"Row {line_number} has no final assistant answer.")
        answer = messages[-1]
        if "reasoning_content" in answer:
            raise SystemExit(f"Row {line_number} still contains reasoning_content.")
        if not protocol.fullmatch(str(answer.get("content", ""))):
            raise SystemExit(f"Row {line_number} violates the PID output protocol.")
        if metadata.get("schema_version") != 3:
            raise SystemExit(f"Row {line_number} is not SFT schema version 3.")
        if "target_metrics" not in metadata:
            raise SystemExit(f"Row {line_number} has no target-response audit metrics.")
        if metadata.get("target_method") != "imc_style":
            raise SystemExit(f"Row {line_number} has an unexpected target method.")
        style = str(metadata.get("control_style", ""))
        if style != expected_style:
            raise SystemExit(
                f"Row {line_number} control style {style!r} != {expected_style!r}."
            )
        expected_style_line = f"Desired control style: {expected_style}."
        user_messages = [
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "user"
        ]
        if not user_messages or any(
            expected_style_line not in content for content in user_messages
        ):
            raise SystemExit(f"Row {line_number} has inconsistent prompt control style.")
        styles[style] += 1
        kinds[str(metadata.get("sample_kind"))] += 1
        plant_types[str((metadata.get("plant") or {}).get("plant_type"))] += 1
        count += 1

if count != expected:
    raise SystemExit(f"Expected {expected} SFT rows, found {count}.")
if expected % 2:
    raise SystemExit("EXPECTED_SFT_ROWS must be even for the balanced SFT data set.")
expected_per_type = expected // 2
if plant_types != Counter({"first_order": expected_per_type, "second_order": expected_per_type}):
    raise SystemExit(
        f"Expected {expected_per_type} rows per plant type; found {dict(plant_types)}."
    )
if not kinds["initial"] or not kinds["feedback"]:
    raise SystemExit(f"Both initial and feedback rows are required; found {dict(kinds)}.")
if styles != Counter({expected_style: expected}):
    raise SystemExit(
        f"Expected only {expected_style} rows; found control styles {dict(styles)}."
    )
print(
    f"SFT data validation passed: rows={count}, control_styles={dict(styles)}, "
    f"plant_types={dict(plant_types)}, sample_kinds={dict(kinds)}"
)
PY

export TOKENIZERS_PARALLELISM=false
"${LLMPIDTUNER_BIN}" validate-sft-data configs/sft/qwen3_0p6b_pid.yaml
export LLMPIDTUNER_SFT_PREFLIGHT_DONE=1

export NPROC_PER_NODE="${NPROC_PER_NODE:-5}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((29500 + SLURM_JOB_ID % 1000))}"

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
  "${VENV_DIR}/bin/llmpidtuner" train-sft configs/sft/qwen3_0p6b_pid.yaml
