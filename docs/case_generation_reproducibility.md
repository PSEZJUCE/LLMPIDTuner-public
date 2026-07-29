# Reproducibility of PID Cases and Training Data

Chinese version: [case_generation_reproducibility_zh.md](case_generation_reproducibility_zh.md)

This document describes the implemented `perturbed_imc_delay_stratified_v1` protocol. Protocol code, frozen source YAML files, and their manifests are authoritative. Older protocols and `Archive/` results must not be pooled with the current benchmark.

## 1. Data sets and isolation

| Set | FOPDT | SOPDT | Purpose |
|---|---:|---:|---|
| demonstration | 10, seed 51001 | 10, seed 52001 | frozen few-shot prompts |
| paper evaluation | 100, seed 61001 | 100, seed 62001 | final comparison |
| GRPO validation | 100, seed 71001 | 100, seed 72001 | checkpoint selection |
| SFT | 20,000 | 20,000 | balanced supervision |
| GRPO | online | online | balanced first-turn optimization |

SFT and online GRPO exclude the case hashes of all frozen demonstration, evaluation, and GRPO-validation sources. Paper evaluation cases are not used for training, tuning, or checkpoint selection.

## 2. Models, sampling, and IMC references

$$
G_f(s)=\frac{K e^{-\theta s}}{Ts+1},\quad K\in[0.2,0.9],\quad T\in[100,600]\ \mathrm{s},
$$

$$
G_s(s)=\frac{K e^{-\theta s}}{(\tau_1s+1)(\tau_2s+1)}.
$$

For SOPDT, $K\in[0.1,3]$, $\tau_{slow}\in[10,100]$ s, and $\tau_{fast}/\tau_{slow}\in[0.1,1]$. Seeded SciPy Latin-hypercube sequences generate continuous variables. Define

$$
T_c=\begin{cases}T,&\mathrm{FOPDT},\\ \tau_1+\tau_2,&\mathrm{SOPDT},\end{cases}
\qquad \rho=\theta/T_c.
$$

$1\le\theta\le200$ s and $\rho\le0.8$. Each ten-case block contains four low-, three medium-, two delay-dominant-, and one strong-delay case, using $[0,0.05)$, $[0.05,0.2)$, $[0.2,0.5)$, and $[0.5,0.8]$.

The style parameter is $\lambda=m\max(\theta,0.1T_c)$, with $m=0.8$, 2.0, and 5.0 for aggressive, balanced, and conservative control. FOPDT uses

$$
K_p=\frac{T+\theta/2}{K(\lambda+\theta/2)},\quad T_i=T+\theta/2,\quad T_d=\frac{T\theta}{2T+\theta},
$$

and SOPDT uses

$$
K_p=\frac{\tau_1+\tau_2}{K(\lambda+\theta)},\quad T_i=\tau_1+\tau_2,\quad T_d=\frac{\tau_1\tau_2}{\tau_1+\tau_2}.
$$

Both are converted through $K_i=K_p/T_i$ and $K_d=K_pT_d$. The balanced controller is the perturbation reference and SFT label, not a common initial PID.

## 3. Abnormal initial PID and acceptance

Balanced gains are multiplied by deterministic factors on a 160-point geometric strength grid from 1.01 to 100. Ten fault directions are used: `p_high`, `p_low`, `i_high`, `i_low_or_off`, `i_high_d_high`, `p_high_d_low`, `p_high_i_high`, `p_low_i_high`, `i_high_d_low`, and `p_low_i_high_d_low`. The exact schedules are defined in `src/llmpidtuner/experiment_protocol.py` and recorded in source manifests.

Severity is based on

$$
q_{IAE}=\frac{IAE_{initial}}{IAE_{balanced}}:
\quad [1.5,2)\ \text{mild},\ [2,5)\ \text{moderate},\ [5,\infty)\ \text{severe}.
$$

Each frozen 100-case set contains ten cases per fault, 30/50/20 by severity, and 40/30/20/10 by delay band. A case is accepted only when initial and reference trajectories are finite, reference IAE is positive, the initial response is unsuccessful, $q_{IAE}\ge1.5$, the initial-output absolute extremum is at most 3, assigned slots match, and the case hash is neither excluded nor duplicated.

## 4. Discrete simulation

All current experiments use $r=1$, 4000 s, 40,001 samples, $\Delta t=0.1$ s, and `max_abs_output=3`. The positional parallel PID is

$$
I_k=I_{k-1}+e_k\Delta t,\qquad
u_k=K_pe_k+K_iI_k+K_d\frac{e_k-e_{k-1}}{\Delta t}.
$$

$u_0=K_pe_0$, so initialization adds no derivative kick. A fractional delay is linearly interpolated from stored controller outputs. With $d=\theta/\Delta t$, $z=(k-1)-d$, $j=\lfloor z\rfloor$, and $\eta=z-j$,

$$
u_{d,k-1}=(1-\eta)u_j+\eta u_{j+1},
$$

while $u_{d,k-1}=0$ for $z<0$.

FOPDT uses forward Euler:

$$
y_k=y_{k-1}+\frac{\Delta t}{T}(Ku_{d,k-1}-y_{k-1}).
$$

SOPDT uses a fixed sequential Euler update:

$$
x_{1,k}=x_{1,k-1}+\frac{\Delta t}{\tau_1}(Ku_{d,k-1}-x_{1,k-1}),
$$

$$
x_{2,k}=x_{2,k-1}+\frac{\Delta t}{\tau_2}(x_{1,k}-x_{2,k-1}),\qquad y_k=x_{2,k}.
$$

The second state uses the newly updated $x_{1,k}$; this is not simultaneous forward Euler on the state vector. This order is fixed in every experiment.

## 5. Metrics and response features

$$
IAE=\Delta t\sum_{k=0}^{N-1}|e_k|,
$$

$$
M_p=100\max\left(0,\frac{\max_k y_k-r}{\max(|r|,10^{-12})}\right),
$$

$$
e_{ss}=100\frac{|\operatorname{mean}(y_{N-m:N})-r|}{\max(|r|,10^{-12})},\quad m=\min(100,N).
$$

Settling time is the earliest sample after which output remains inside the $\pm5\%$ band. Success requires finite simulation and settling, $M_p<15\%$, and $e_{ss}<1\%$, with strict inequalities.

Let $s_r=\max(|r|,10^{-12})$. The additional prompt features are:

1. **Significant crossings:** discard $|y_k-r|\le0.05s_r$ samples and count sign changes in adjacent retained deviations; return 0 if fewer than two remain.
2. **Attenuation ratio:** detect output maxima with `scipy.signal.find_peaks(prominence=0.01*s_r)`. For the first two peaks $p_1,p_2$, $A_r=|y_{p_2}-r|/\max(|y_{p_1}-r|,10^{-12})$; return 0 if fewer than two peaks exist.
3. **Oscillation period:** $T_{osc}=t_{p_2}-t_{p_1}$; return 0 if fewer than two peaks exist.
4. **Time to 63.2% after dead time:** for $r=1$, report the first $t_k-\theta$ satisfying $t_k\ge\theta$ and $y_k\ge0.632r$. Return `N/A` if absent. No inter-sample interpolation is used.

## 6. SFT, GRPO, and audit

`configs/data/pid_sft_messages_40k.yaml` generates 20,000 cases of each process family. Every row has a 0.5 probability of attempting a feedback history; requested depths 1/2/3 have conditional probabilities 0.5/0.3/0.2. Each layer tries at most 20 finite unsuccessful intermediate controllers. The final label is always the process-specific balanced IMC PID and uses `P:<value>; I:<value>; D:<value>`.

GRPO uses base seed 91001 and seed $91001+100003r$ on distributed process $r$. With five processes, 800 steps, four prompts per process per step, and four completions per prompt, a complete run evaluates 16,000 online prompts and 64,000 completions. Frozen GRPO validation is evaluated every 50 steps and receives no gradients.

```bash
uv run llmpidtuner build-protocol-assets --check
uv run llmpidtuner make-sft-data configs/data/pid_sft_messages_40k.yaml
uv run llmpidtuner validate-sft-data configs/sft/qwen3_0p6b_pid.yaml
```

Archive the Git commit, `uv.lock`, configs, source YAML files, manifests, checkpoints, logs, and run outputs together. Hashes in the corresponding manifests are authoritative and are intentionally not duplicated here.
