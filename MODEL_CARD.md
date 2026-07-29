# Model Card

## Model Family

The small-model experiments start from `Qwen/Qwen3-0.6B` and produce:

1. a balanced-response SFT model;
2. a LoRA GRPO policy initialized from the selected SFT model.

Qwen thinking is disabled during benchmark inference. The required output is exactly:

```text
P:<value>; I:<value>; D:<value>
```

## Intended Use

The models are research prototypes for recommending PID parameters from structured response diagnostics and frozen in-context demonstrations. They are intended for simulation studies and method comparison, not direct autonomous deployment on industrial equipment.

## Training

SFT uses 40,000 generated balanced-style samples (20,000 FOPDT and 20,000 SOPDT), a 95/5 train-validation split, and best-checkpoint loading. The data generator excludes all frozen demonstration, paper-evaluation, and GRPO-validation plants.

GRPO samples balanced first-turn prompts online. The reward combines:

- Pad茅/Routh-Hurwitz stability screening;
- exact-delay frequency-domain stability margin;
- time-domain performance;
- IAE improvement;
- format compliance;
- dimensionless gain regularization.

The independent frozen GRPO-validation set is evaluated before training and every 50 steps. Checkpoint selection prioritizes Pass@1, bounded mean IAE improvement, mean reward, and lower KL.

## Released Material

The initial public release includes model code, training configs, data generation, reward implementation, frozen validation assets, evaluation scripts, training-curve inputs, and manifests. Fine-tuned weights are not included. Users can retrain the models using the released pipeline; a later model deposit may add immutable weight files and checksums.

## Evaluation

Base, SFT, and GRPO models use the same 100 FOPDT and 100 SOPDT held-out cases, prompt protocol, simulator, request settings, and success criteria. Paper evaluation cases are not used for training or checkpoint selection.

## Limitations and Safety

- Recommendations can be invalid, unstable, or poorly damped.
- Stability screening relies on the documented model family and does not establish robust stability for an unknown physical plant.
- Synthetic simulation does not capture sensor noise, saturation, nonlinearities, disturbances, or implementation constraints.
- A qualified control engineer must validate every controller before hardware use.
- The released software must not be treated as a safety controller.