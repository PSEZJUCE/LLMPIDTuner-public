# Benchmark and Demonstration Provenance

Chinese version: [benchmark_and_demonstration_provenance_zh.md](benchmark_and_demonstration_provenance_zh.md)

This document records the current `perturbed_imc_delay_stratified_v1` experiment protocol. It is the compact provenance summary for manuscript auditing. Detailed generation equations are in [case_generation_reproducibility.md](case_generation_reproducibility.md).

## 1. Protocol scope

The protocol removes four confounders from the historical workflow:

1. Initial PID gains are process-specific perturbations of balanced IMC gains, not a fixed `(1, 0.1, 0.01)` controller.
2. Demonstration, paper-evaluation, GRPO-validation, and training cases are isolated.
3. Dead time varies with process scale, and every prompt diagnosis is recomputed for its own process, delay, and initial PID.
4. All methods share the same simulator, metric definitions, initial cases, and success criteria.

## 2. Process and delay ranges

FOPDT uses $K\in[0.2,0.9]$ and $T\in[100,600]$ s. SOPDT uses $K\in[0.1,3.0]$, $\tau_{slow}\in[10,100]$ s, and $\tau_{fast}/\tau_{slow}\in[0.1,1]$.

$$
T_c=\begin{cases}T,&\mathrm{FOPDT},\\ \tau_1+\tau_2,&\mathrm{SOPDT},\end{cases}
\qquad \rho=\theta/T_c.
$$

$1\le\theta\le200$ s and $\rho\le0.8$. Every ten-case block contains four low-, three medium-, two delay-dominant-, and one strong-delay case.

## 3. IMC references and initial faults

Style-specific IMC uses

$$
\lambda=m\max(\theta,0.1T_c),
$$

where $m=0.8$, 2.0, and 5.0 for aggressive, balanced, and conservative control. The balanced IMC controller supplies the perturbation reference and SFT target; it is not the initial controller.

Ten deterministic PID fault directions cover high/low proportional and integral action and selected coupled P/I/D faults. Perturbation strength is searched on a 160-point geometric grid from 1.01 to 100. A case is accepted only when its initial response is finite, unsuccessful, bounded by $|y|\le3$, and has at least 1.5 times the balanced-reference IAE.

Severity uses $IAE_{initial}/IAE_{balanced}$: $[1.5,2)$ mild, $[2,5)$ moderate, and $[5,\infty)$ severe. Every frozen 100-case family contains 30 mild, 50 moderate, and 20 severe cases and ten cases of every fault type.

## 4. Simulation and success

All current protocol cases use setpoint 1, 4000 s, 40,001 points, $\Delta t=0.1$ s, and `max_abs_output=3`. Fractional dead time is linearly interpolated from controller-output history. FOPDT uses forward Euler; SOPDT uses the documented sequential Euler update in which the second lag reads the newly updated first-lag state.

IAE is rectangular integration over the complete horizon. Steady-state error uses the mean of the last 100 samples. Settling requires entering the $\pm5\%$ band and never leaving it. Success requires finite simulation, finite settling time, overshoot below 15%, and steady-state error below 1%, with strict inequalities.

## 5. Frozen sets

| Set | FOPDT | SOPDT | Purpose |
|---|---:|---:|---|
| demonstration | seed 51001, 10 | seed 52001, 10 | frozen few-shot context |
| paper evaluation | seed 61001, 100 | seed 62001, 100 | final comparison |
| GRPO validation | seed 71001, 100 | seed 72001, 100 | checkpoint selection |

Source files are under `cases/protocol/perturbed_imc_delay_stratified/sources/`. Their manifests record seeds, case hashes, schedules, and asset hashes. SFT and GRPO explicitly exclude all frozen sources. Paper evaluation is observation-only and cannot select epochs, checkpoints, or hyperparameters.

## 6. Frozen demonstrations

The same ten source cases per process family are rendered as:

```text
balanced/full
balanced/kpi3
balanced/numeric8
aggressive/full
conservative/full
```

Prompt variants change only the response information presented. Style variants retain the same initial cases but recompute the recommended PID using the requested IMC style. Runtime validates both source and prompt hashes through the companion manifest.

## 7. Training isolation and audit

SFT contains 20,000 FOPDT and 20,000 SOPDT balanced records, generated with seed 81001. GRPO starts from that balanced SFT model and samples balanced first-turn prompts online with seed 91001. Frozen 100+100 GRPO validation is evaluated before training and every 50 steps but receives no gradients.

Archive the Git commit, `uv.lock`, source YAML files, demonstration manifests, run configs, model/data manifests, checkpoints, logs, and final curves together. Results with different protocol IDs, prompt hashes, simulation grids, or metric definitions must not be pooled.
