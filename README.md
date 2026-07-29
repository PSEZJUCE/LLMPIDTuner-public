# LLMPIDTuner

> **Associated arXiv paper:** The code and released results in this repository correspond to *A Physics-Informed Framework for PID Tuning of Chemical Processes Using Large Language Model Agents*. The arXiv link will be updated after submission: [arXiv preprint](https://arxiv.org/abs/XXXX.XXXXX).
>
> **Relationship to the master's thesis:** This repository is an improved and extended implementation developed on the basis of Zhoupeng Shou's master's thesis work. Its code architecture, experiment protocol, generated cases, demonstrations, SFT/GRPO pipeline, stability treatment, and reported results have been revised; it is therefore not identical to, and should not be interpreted as a complete reproduction of, the original master's thesis.
>
> **Reproducibility dataset:** Full raw evaluation runs, the balanced 40,000-row SFT dataset, training records, and frozen protocol materials are archived at [Zenodo](https://doi.org/10.5281/zenodo.21669697).

LLMPIDTuner is a reproducible research codebase for response-diagnostic PID tuning of FOPDT and SOPDT processes. It supports four comparable routes:

- OpenAI-compatible API models, including DeepSeek V4 Flash and Qwen3.7 Plus;
- Qwen3-0.6B after supervised fine-tuning (SFT);
- the SFT model after online group relative policy optimization (GRPO);
- internal model control (IMC) as a non-LLM reference.

The frozen paper protocol is `perturbed_imc_delay_stratified_v1`. It uses variable dead time, case-specific perturbed initial PID settings, independent demonstration/evaluation/GRPO-validation sources, and a shared simulator and metric implementation.

## Repository Contents

```text
src/llmpidtuner/   simulation, metrics, prompting, API evaluation, SFT, and GRPO
cases/eval/        frozen 100-case evaluation configurations
cases/protocol/    frozen demonstration/evaluation/validation source cases
cases/demonstrations/
                   frozen prompt artifacts and SHA-256 manifests
configs/data/      SFT data generation configuration
configs/sft/       SFT training configuration
configs/grpo/      GRPO training and validation configuration
scripts/slurm/     portable Slurm job templates
scripts/paper/     manuscript figure generation
docs/              protocol and method records
paper_artifacts/   compact paper figures, tables, and training-curve inputs
tests/             unit and protocol tests
```

Generated `runs/`, training `outputs/`, API secrets, large datasets, and model checkpoints are intentionally excluded from Git.

## Installation

Python is pinned to 3.12.11 in `.python-version`; `pyproject.toml` accepts Python `>=3.11,<3.13`, and `uv.lock` fixes the dependency graph.

For simulation, API evaluation, plotting, and development:

```bash
uv sync --group dev
uv run pytest -q
```

For SFT, GRPO, and vLLM on a CUDA server:

```bash
uv sync --extra training --extra training-stack
```

Copy `.env.example` to `.env` and provide only local endpoints and secrets. Provider/model selection is defined by `llm_profile` in case YAML; `.env` stores API keys and base URLs. Never commit `.env`.

## Frozen Experiment Protocol

Canonical simulation and convergence settings are:

- horizon `4000 s`, `40001` points, and `dt=0.1 s`;
- fractional dead time implemented by linear interpolation;
- no derivative kick at the setpoint step;
- nonnegative overshoot, tail-100-point SSE, and full-horizon IAE;
- convergence requires settling within the 5% band, overshoot below 15%, and SSE below 1%.

The committed source sets are independent:

| Purpose | FOPDT seed/count | SOPDT seed/count |
| --- | ---: | ---: |
| Demonstrations | 51001 / 10 | 52001 / 10 |
| Paper evaluation | 61001 / 100 | 62001 / 100 |
| GRPO validation | 71001 / 100 | 72001 / 100 |

Verify that all generated protocol assets match their committed hashes:

```bash
uv run llmpidtuner build-protocol-assets --check
```

To intentionally rebuild them:

```bash
uv run llmpidtuner build-protocol-assets --force
uv run llmpidtuner build-protocol-assets --check
```

Do not rebuild frozen assets between compared experiments. Detailed ranges, fault schedules, seeds, exclusions, and generation equations are documented in [docs/benchmark_and_demonstration_provenance_zh.md](docs/benchmark_and_demonstration_provenance_zh.md) and [DATA_CARD.md](DATA_CARD.md).

## API Evaluation

Evaluation YAMLs default to `dry_run`. `run-api-parallel` switches them to LLM mode; single-case execution requires `--mode llm`.

```powershell
uv run llmpidtuner run-api-parallel --workers 10 `
  cases/eval/first_order_100_deepseek_v4_flash.yaml `
  cases/eval/second_order_100_deepseek_v4_flash.yaml `
  cases/eval/first_order_100_qwen3_7_plus.yaml `
  cases/eval/second_order_100_qwen3_7_plus.yaml
```

Resume an interrupted batch by adding `--resume`. A/B prompt ablations use `_kpi3` and `_numeric8`; style variants use `_aggressive` and `_conservative`. All variants reuse the same frozen 100+100 evaluation plants and initial conditions.

Run the IMC reference:

```bash
uv run llmpidtuner run cases/eval/first_order_100_imc.yaml
uv run llmpidtuner run cases/eval/second_order_100_imc.yaml
```

## SFT and GRPO

The committed Slurm files are templates. Adjust `#SBATCH --partition`, resources, and wall time for the target cluster. Submit from the repository root or set `PROJECT_ROOT`; set `MODEL_PATH` when a script requires a local model directory.

```bash
sbatch scripts/slurm/generate_qwen3_0p6b_training_data_40k.sbatch
sbatch scripts/slurm/train_qwen3_0p6b_sft_5gpu.sbatch
sbatch scripts/slurm/train_qwen3_0p6b_grpo_from_sft_5gpu.sbatch
```

The SFT dataset contains 20,000 FOPDT and 20,000 SOPDT rows, all using the frozen balanced/full demonstrations and convergent balanced-IMC targets. Demonstration, evaluation, and GRPO-validation plants are excluded. SFT uses a 95/5 split and best-checkpoint loading.

GRPO starts from the selected SFT model, samples balanced first-turn prompts online, validates against an independent frozen 100+100 set, and records the selected checkpoint in its training manifest. Training and checkpoint behavior are documented in [docs/sft_technical_route.md](docs/sft_technical_route.md), [docs/grpo_training_protocol.md](docs/grpo_training_protocol.md), and [MODEL_CARD.md](MODEL_CARD.md).

Evaluate local models:

```bash
sbatch scripts/slurm/eval_qwen3_0p6b_base_prompt_ablation.sbatch
sbatch scripts/slurm/eval_qwen3_0p6b_sft_all_5gpu.sbatch
sbatch scripts/slurm/eval_qwen3_0p6b_grpo_all_5gpu.sbatch
```

All Qwen evaluations disable thinking and constrain the output to `P:<value>; I:<value>; D:<value>`.

## Analysis and Paper Figures

```bash
uv run llmpidtuner analyze-batches \
  runs/first_order_100_deepseek_v4_flash \
  runs/first_order_100_qwen3_7_plus \
  runs/first_order_100_base_qwen3_0p6b \
  --labels DeepSeek-V4-Flash Qwen3.7-Plus Qwen3-0.6B \
  --output runs/first_order_full_statistics.png \
  --summary-csv runs/first_order_full_summary.csv \
  --case-csv runs/first_order_full_cases.csv
```

Generate manuscript figures from completed result directories:

```bash
uv run python scripts/paper/generate_paper_figures.py all
```

The compact figure/table snapshot used for manuscript preparation is included under `paper_artifacts/`. Full raw run directories and the generated SFT dataset are archived in the [Zenodo reproducibility dataset](https://doi.org/10.5281/zenodo.21669697) because they are too large and file-heavy for a normal Git repository. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Reproducibility Rules

- Never use the 100+100 paper evaluation cases for SFT, GRPO training, or checkpoint selection.
- Never use paper evaluation outcomes to select SFT epochs or a GRPO checkpoint.
- Preserve case sources, frozen prompts, manifests, configs, code version, model checkpoint identity, and run metadata together.
- Do not combine results generated under different simulation or prompt protocols without explicitly labeling the difference.

## Citation and License

Citation metadata is provided in [CITATION.cff](CITATION.cff). The reproducibility dataset is available at [Zenodo](https://doi.org/10.5281/zenodo.21669697); the arXiv identifier and software archive DOI will be added when available.

The software is released under the [PolyForm Noncommercial License 1.0.0](LICENSE). Frozen cases, demonstrations, documentation, figures, and result tables are released under [CC BY-NC 4.0](LICENSE-DATA). Academic and other noncommercial use is permitted under those terms; commercial use requires a separate written license from the relevant rights holders. See [LICENSES.md](LICENSES.md) for the exact scope.

Because commercial use is restricted, this is a **source-available noncommercial research release**, not OSI-approved open-source software. The associated arXiv paper uses the separate arXiv.org perpetual, non-exclusive license.