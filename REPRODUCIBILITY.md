# Reproducibility Guide

This guide separates four reproducibility levels so that readers can reproduce the numerical protocol without requiring the authors' private infrastructure.

## 1. Environment and Protocol Check

```bash
uv sync --group dev
uv run pytest -q
uv run llmpidtuner build-protocol-assets --check
```

The last command regenerates protocol assets in memory and verifies the committed source cases, demonstrations, and manifests. It should not modify the working tree.

## 2. Deterministic Simulation and IMC Baselines

The FOPDT and SOPDT evaluation YAMLs under `cases/eval/` reference the same frozen 100+100 process cases. IMC does not require an LLM:

```bash
uv run llmpidtuner run cases/eval/first_order_100_imc.yaml
uv run llmpidtuner run cases/eval/second_order_100_imc.yaml
```

These commands recreate the simulated trajectories and metrics under the canonical `4000 s / 40001 point` protocol.

## 3. API Model Evaluation

Create `.env` from `.env.example`, add provider endpoints and API keys, and run:

```bash
uv run llmpidtuner run-api-parallel --workers 10 --resume \
  cases/eval/first_order_100_deepseek_v4_flash.yaml \
  cases/eval/second_order_100_deepseek_v4_flash.yaml \
  cases/eval/first_order_100_qwen3_7_plus.yaml \
  cases/eval/second_order_100_qwen3_7_plus.yaml
```

API providers may update hosted model implementations. Preserve each run's `llm_metadata.txt`, request seed, endpoint model identifier, date, and case-level status files.

## 4. Small-Model Training

Install the CUDA stack on the training server:

```bash
uv sync --extra training --extra training-stack
```

Generate and audit the balanced-only 40,000-row SFT dataset:

```bash
uv run llmpidtuner make-sft-data configs/data/pid_sft_messages_40k.yaml
uv run llmpidtuner validate-sft-data configs/sft/qwen3_0p6b_pid.yaml
```

On Slurm, use the committed job templates:

```bash
sbatch scripts/slurm/generate_qwen3_0p6b_training_data_40k.sbatch
sbatch scripts/slurm/train_qwen3_0p6b_sft_5gpu.sbatch
sbatch scripts/slurm/train_qwen3_0p6b_grpo_from_sft_5gpu.sbatch
```

The templates expect five CUDA devices for SFT and GRPO. Cluster partition names and wall-time/resource directives are site-specific. The base model may be supplied as the Hugging Face identifier `Qwen/Qwen3-0.6B` or as an equivalent local snapshot.

## 5. Evaluation and Figures

Evaluate base, SFT, and validation-selected GRPO models with the Slurm templates in `scripts/slurm/`. Then regenerate all paper figures:

```bash
uv run python scripts/paper/generate_paper_figures.py all
```

Compact manuscript-facing figures, companion CSV/JSON files, and training-curve inputs are stored in `paper_artifacts/`. Their checksums are in `paper_artifacts/SHA256SUMS`.

## 6. External Artifacts

The following large artifacts are intentionally not stored in Git:

- complete case-level raw runs;
- the 40,000-row generated SFT JSONL;
- full SFT checkpoints;
- GRPO checkpoints and optimizer state;
- vLLM and Slurm logs not needed for plotted training curves.

The versioned data archive and its checksums are available in the [Zenodo reproducibility dataset](https://doi.org/10.5281/zenodo.21669697). Fine-tuned weights are not part of the initial release; direct inference reproduction therefore requires retraining from the released code/configuration/data generator.

## 7. Numerical Compatibility

The shared simulator, IAE, overshoot, SSE, and convergence implementations are used by API evaluation, SFT data generation, GRPO reward evaluation, and analysis. Do not alter these definitions when comparing against the published benchmark. Protocol changes require a new protocol identifier and new frozen assets.