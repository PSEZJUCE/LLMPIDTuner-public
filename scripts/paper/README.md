# Paper figure generation

This directory contains the read-only manuscript figure pipeline. It only consumes completed
experiment artifacts: no LLM call, PID tuning run, or result-directory modification is performed.
Figures and companion CSV/JSON files are written below `runs/paper_figures/` by default.

## Recommended manuscript layout

- Section 3.1: a 2x2 evaluation-case coverage figure showing FOPDT and SOPDT process spaces,
  relative-delay strata, and PID fault/severity coverage. Companion CSV files report exact
  parameter ranges and case counts.
- Section 4.1.1: four independent typical-case response figures, plus FOPDT and SOPDT batch
  statistics comparing DeepSeek V4 Flash, Qwen3.7 Plus, and Base-0.6B. The typical-case
  figures additionally include the IMC baseline.
- Section 4.1.2: DeepSeek full/KPI-3/numeric-8 statistics in the main text; matching Qwen figures
  in the appendix. Normalized per-case IAE-improvement box plots and CSVs are also produced,
  because convergence, overshoot, and SSE bins can hide ablation effects. Outlier markers are
  omitted from the box plots for readability, while every value remains in the companion CSVs.
- Section 4.1.3: four DeepSeek style-response figures in the main text; matching Qwen figures in
  the appendix.
- Section 4.2: SFT train/validation loss, GRPO reward/KL/independent-validation curves, and
  one-step FOPDT/SOPDT comparisons across DeepSeek, Qwen, Base, SFT, and GRPO. A compact 1x2
  Pass@1 count figure is produced for the main text, while the detailed statistics are retained.
  A second chart reports the combined Pass@1 rate over all 200 cases. The SFT curve and three
  GRPO curves are saved as four independent, equally sized figures.
- Section 4.3: the four typical cases with SFT and GRPO added, plus exact-delay Bode/Nyquist
  comparisons for FOPDT group 018 and SOPDT group 014.

All plots omit figure titles. Add captions and panel identifiers in the manuscript layout.

## Commands

Run sections independently from the repository root:

```powershell
uv run python scripts/paper/generate_paper_figures.py 3.1
uv run python scripts/paper/generate_paper_figures.py 4.1.1
uv run python scripts/paper/generate_paper_figures.py 4.1.2
uv run python scripts/paper/generate_paper_figures.py 4.1.3
uv run python scripts/paper/generate_paper_figures.py 4.3
```

Before regenerating Section 4.1.3, complete the missing FOPDT group 018 aggressive and
conservative API runs. The balanced group 018 result already exists:

```powershell
uv run llmpidtuner run cases/eval/first_order_100_deepseek_v4_flash_aggressive.yaml --mode llm --groups 18 --resume
uv run llmpidtuner run cases/eval/first_order_100_deepseek_v4_flash_conservative.yaml --mode llm --groups 18 --resume
uv run llmpidtuner run cases/eval/first_order_100_qwen3_7_plus_aggressive.yaml --mode llm --groups 18 --resume
uv run llmpidtuner run cases/eval/first_order_100_qwen3_7_plus_conservative.yaml --mode llm --groups 18 --resume
```

Then regenerate the style figures:

```powershell
uv run python scripts/paper/generate_paper_figures.py 4.1.3
```

The SOPDT group 052 figures in Sections 4.1.1, 4.1.3, and 4.3 use a broken time axis. They
retain the transient response over 0--500 s and the final steady-state segment over 3800--4000 s
while omitting the visually redundant interval between them.

Section 4.2 can generate the one-step model comparison without training logs:

```powershell
uv run python scripts/paper/generate_paper_figures.py 4.2
```

For complete SFT/GRPO training figures, copy only these lightweight server artifacts:

```text
SFT output directory:
  trainer_state.json

GRPO output directory:
  trainer_log.jsonl
  validation_log.jsonl
  training_manifest.json
```

Example download commands from the current server output directories:

```powershell
New-Item -ItemType Directory -Force runs/paper_inputs/sft, runs/paper_inputs/grpo
scp user@server:/path/to/LLMPIDTuner/outputs/sft/qwen3_0p6b_pid/trainer_state.json runs/paper_inputs/sft/
scp user@server:/path/to/LLMPIDTuner/outputs/grpo/qwen3_0p6b_pid/trainer_log.jsonl runs/paper_inputs/grpo/
scp user@server:/path/to/LLMPIDTuner/outputs/grpo/qwen3_0p6b_pid/validation_log.jsonl runs/paper_inputs/grpo/
scp user@server:/path/to/LLMPIDTuner/outputs/grpo/qwen3_0p6b_pid/training_manifest.json runs/paper_inputs/grpo/
```

Then run:

```powershell
uv run python scripts/paper/generate_paper_figures.py 4.2 `
  --sft-dir runs/paper_inputs/sft `
  --grpo-dir runs/paper_inputs/grpo
```

Generate every section in one command:

```powershell
uv run python scripts/paper/generate_paper_figures.py all `
  --sft-dir runs/paper_inputs/sft `
  --grpo-dir runs/paper_inputs/grpo
```

Sections 4.1.1, 4.1.3, and 4.3 consistently use FOPDT groups 018 and 055 and SOPDT groups
014 and 052. Section 4.3 uses groups 018 and 014 for stability-margin analysis.
