# Data Card

## Scope

LLMPIDTuner uses synthetic closed-loop PID-control data for first-order-plus-dead-time (FOPDT) and second-order-plus-dead-time (SOPDT) processes. No human-subject, personal, proprietary plant, or safety-sensitive operational data are included.

## Frozen Protocol

Protocol identifier: `perturbed_imc_delay_stratified_v1`.

Each accepted case contains:

- a process model and dead time;
- a balanced-IMC reference controller;
- a deliberately perturbed initial PID controller;
- the simulated initial response and response diagnostics;
- acceptance and provenance metadata.

The initial controller must produce a finite and bounded response, fail the benchmark convergence criteria, and have IAE at least 1.5 times the balanced-IMC reference. Consequently every benchmark case requires at least one tuning action.

## Independent Source Sets

| Split | FOPDT | SOPDT | Purpose |
| --- | ---: | ---: | --- |
| Demonstration | seed 51001, 10 cases | seed 52001, 10 cases | frozen in-context examples |
| Paper evaluation | seed 61001, 100 cases | seed 62001, 100 cases | final held-out comparison |
| GRPO validation | seed 71001, 100 cases | seed 72001, 100 cases | checkpoint selection only |

These source sets are disjoint by construction. Their committed YAMLs and prompt manifests live under `cases/protocol/` and `cases/demonstrations/`.

## SFT Dataset

The generated SFT dataset contains 40,000 JSONL rows:

- 20,000 FOPDT rows;
- 20,000 SOPDT rows;
- balanced control style only;
- both initial-recommendation and feedback-recommendation samples;
- strict assistant output format `P:<value>; I:<value>; D:<value>`;
- no `reasoning_content`.

The generator excludes every frozen demonstration, paper-evaluation, and GRPO-validation plant. Targets are generated with the balanced IMC-style rule and audited by re-simulation. The JSONL is generated rather than committed because of size; its manifest records row counts, hashes, source-tree hash, style, and generator configuration.

## GRPO Prompts

GRPO generates balanced, first-turn prompts online from the shared process sampler and simulator. It does not train from a pre-generated prompt JSONL. Frozen paper-evaluation cases are excluded, and checkpoint selection uses only the independent GRPO-validation set.

## Simulation Protocol

- Setpoint: 1.0
- Simulation horizon: 4000 s
- Number of points: 40001
- Sampling interval: 0.1 s
- Dead-time implementation: fractional delay with linear interpolation
- Output safety bound for case acceptance: `|y| <= 3`
- Success criteria: settled inside the 5% band, overshoot below 15%, SSE below 1%

Detailed equations, parameter ranges, delay strata, fault schedules, and deterministic generation rules are recorded in `docs/benchmark_and_demonstration_provenance.md` and its Chinese counterpart.

## Distribution

Frozen source cases and demonstrations are included in Git. Compact paper tables and figures are in `paper_artifacts/`. These research artifacts are licensed under CC BY-NC 4.0. Full raw evaluation runs and generated training data will be deposited separately with checksums, the same data-license scope, and a DOI.

## Limitations

The data are synthetic and restricted to the documented FOPDT/SOPDT families, setpoint response, PID structure, disturbance-free simulator, and acceptance filters. Performance on unmodeled nonlinearities, noise, actuator constraints, plant/model mismatch, multivariable processes, or safety-critical hardware is not established.