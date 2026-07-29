# Qwen3-0.6B SFT Technical Route

Chinese version: [sft_technical_route_zh.md](sft_technical_route_zh.md)

This document describes the implemented supervised-fine-tuning data, objective, split, training configuration, and artifacts.

## 1. Learning task

The model maps a frozen demonstration, current PID, response diagnostics, observed dead time, and requested balanced style to one concise response:

```text
P:<Kp>; I:<Ki>; D:<Kd>
```

Transfer-function parameters are not included in the prompt. Assistant labels contain only the three PID values and no reasoning text. Thinking is disabled for Base, SFT, and GRPO during formal evaluation.

## 2. Data and isolation

`configs/data/pid_sft_messages_40k.yaml` generates `data/sft/pid_sft_messages_40000.jsonl`:

| Family | Records |
|---|---:|
| FOPDT | 20,000 |
| SOPDT | 20,000 |
| Total | 40,000 |

All records use balanced control, seed 81001, and the frozen balanced/full demonstration. Demonstration, paper-evaluation, and GRPO-validation case hashes are excluded. Every label is the balanced IMC PID calculated independently for that process and dead time and is accepted only when its simulated response meets the same observable success criteria.

The demonstration is prompt context, not 20 additional training rows.

## 3. Initial and feedback records

Every generated case begins as a first-turn prompt. With probability 0.5, the writer attempts to add a feedback history. Requested depths 1, 2, and 3 have conditional probabilities 0.5, 0.3, and 0.2. Intermediate PIDs are sampled between the current and target controllers in log space with log-normal jitter. Each intermediate response must be finite, unsuccessful, and bounded by $|y|\le3$.

If no acceptable intermediate PID is found, feedback construction stops early. Consequently, the realized feedback count and depth distribution are recorded in the data manifest rather than assumed from nominal probabilities. The final assistant message is always the balanced IMC target.

## 4. Prompt length and metadata

Every JSONL row uses OpenAI message structure and metadata schema version 3. Metadata records sample kind, realized feedback depth, style, process, dead time, current PID and metrics, target PID and metrics, and `target_method=imc_style`.

The maximum sequence length is 6144 tokens. Preflight tokenization rejects any longer row; training does not silently truncate the prompt or label.

## 5. Objective

SFT minimizes assistant-token autoregressive cross entropy:

$$
\mathcal L_{SFT}=-\sum_{t=1}^{|y|}\log p_\theta(y_t\mid x,y_{<t}).
$$

This is token prediction loss, not PID-value MSE and not a direct IAE or stability loss. Physical behavior enters through validated IMC labels and the later PI-GRPO reward.

## 6. Training configuration

The authoritative config is `configs/sft/qwen3_0p6b_pid.yaml`.

| Parameter | Value |
|---|---:|
| base model | `Qwen/Qwen3-0.6B` |
| precision | bf16, full parameter |
| max length | 6144 |
| learning rate | $10^{-5}$ |
| epochs | 3 |
| batch per GPU | 1 sequence |
| gradient accumulation | 8 |
| GPU processes | 5 |
| global optimizer batch | 40 sequences |
| validation | 5%, stratified by process family and sample kind |
| warmup ratio | 0.03 |
| weight decay | 0.01 |
| max gradient norm | 1.0 |
| evaluation/save interval | 50 optimizer steps |
| seed | 42 |

With 38,000 training records and global batch 40, one epoch contains 950 optimizer steps and three epochs contain 2,850 steps. `load_best_model_at_end` restores the checkpoint with the lowest validation loss. The frozen paper evaluation set does not select an epoch.

## 7. Commands and artifacts

```bash
uv sync --extra training --extra training-stack
sbatch scripts/slurm/generate_qwen3_0p6b_training_data_40k.sbatch
sbatch scripts/slurm/train_qwen3_0p6b_sft_5gpu.sbatch
```

The default output is `outputs/sft/qwen3_0p6b_pid`. Preserve the JSONL and manifest, SFT config, Slurm logs, `training_manifest.json`, best exported model, train/validation loss history, Git commit, and `uv.lock`.

## 8. Boundary with PI-GRPO

SFT learns the balanced IMC response-to-PID mapping. PI-GRPO starts from the exported balanced SFT model, generates balanced first-turn prompts online, and applies simulation, Padé-Routh, exact-delay phase-margin, performance, IAE, format, and gain-regularization rewards. It does not read a pre-generated GRPO training JSONL or reuse the paper evaluation set.

See [grpo_training_protocol.md](grpo_training_protocol.md) for the current reward and checkpoint protocol.
