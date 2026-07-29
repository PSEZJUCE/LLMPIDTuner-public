# PI-GRPO Training, Reward, and Checkpoint Protocol

Chinese version: [grpo_training_protocol_zh.md](grpo_training_protocol_zh.md)

## 1. Scope

PI-GRPO optimizes the first PID recommendation for balanced control. It starts from the balanced SFT model and generates training prompts online. The frozen paper 100+100 evaluation set is excluded from training, hyperparameter selection, and checkpoint selection. A separate frozen 100+100 validation set selects checkpoints.

## 2. Policy update

For prompt $q$, the rollout policy generates $G=4$ candidates $o_i$ with $T_i$ valid completion tokens. Candidate reward is normalized within the prompt group to obtain advantage $A_i$. The token-level clipped objective uses

$$
\rho_{i,t}(\Theta)=
\frac{\pi_\Theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{old}(o_{i,t}\mid q,o_{i,<t})},
$$

and the non-negative reference-policy KL estimator

$$
D_{KL,i,t}=\exp(r^{ref}_{i,t}-r^\Theta_{i,t})-(r^{ref}_{i,t}-r^\Theta_{i,t})-1.
$$

The implementation averages over all valid completion tokens:

$$
J(q)=\frac{1}{\sum_iT_i}\sum_{i=1}^{G}\sum_{t=1}^{T_i}
\left[
\min\left(\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\right)
-\beta_kD_{KL,i,t}
\right],
$$

with $\epsilon=0.2$. The reference policy is the frozen SFT model; the rollout policy is the policy before the current update.

## 3. Safety and task success

A completion receives the unsafe reward $-1$ if it cannot be parsed; violates $K_p>0$, $K_i\ge0$, $K_d\ge0$; yields a non-finite or over-limit simulation; does not pass the first-order Padé Routh test; or has a detected unity-gain crossover with non-positive exact-delay phase margin.

A Routh-stable case with no crossover in the configured search range remains safe but receives zero stability component. Task success additionally requires finite settling, $M_p<15\%$, and $e_{ss}<1\%$.

## 4. Quality score and three reward branches

For safe candidates,

$$
Q=0.35R_{stab}+0.35R_{perf}+0.15R_{IAE}+0.10R_{gain}+0.05R_{format}.
$$

The performance component contains no IAE:

$$
R_{perf}=\operatorname{clip}\left(\frac{5}{12}R_{os}+\frac13R_{set}+\frac14R_{ss},-1,1\right).
$$

With $q_s=t_s/t_{s,ref}$,

$$
t_{s,ref}=\theta-\ln(0.05)\lambda_{balanced},\qquad
\lambda_{balanced}=2\max(\theta,0.1T_c).
$$

The code uses the piecewise terms documented in the manuscript SI: overshoot breakpoints 5/15/30%, settling-ratio breakpoints 1/2, and steady-state-error breakpoints 0.1/1/2%.

IAE improvement is

$$
\Delta_{IAE}=\frac{IAE_{before}-IAE_{after}}{\max(|IAE_{before}|,10^{-8})},
\qquad R_{IAE}=\tanh(2\Delta_{IAE}).
$$

Gain regularization uses

$$
g=(KK_p,\;KK_iT_c,\;KK_d/T_c),
$$

and a per-process balanced-IMC reference $g_{ref}$, with every reference dimension floored at 0.01. Limits are $(4g_{p,ref},4g_{i,ref},6g_{d,ref})$. If $e_j=\max[0,\ln((g_j+10^{-12})/(g_{j,lim}+10^{-12}))]$, then

$$
R_{gain}=-\min\left(1,\sum_je_j^2\right).
$$

The final reward is

$$
R=\begin{cases}
-1,&\text{unsafe},\\[2mm]
-0.5+0.25(Q+1),&\text{safe but unsuccessful},\\[2mm]
0.5+0.25(Q+1),&\text{safe and successful}.
\end{cases}
$$

Thus the branches are $-1$, $[-0.5,0]$, and $[0.5,1]$.

## 5. Training configuration

The authoritative config is `configs/grpo/qwen3_0p6b_pid.yaml`.

| Parameter | Value |
|---|---:|
| QLoRA | NF4 4-bit, double quantization |
| LoRA rank / alpha / dropout | 16 / 32 / 0 |
| cumulative steps | 800 |
| distributed processes | 5 |
| prompts/process/step | 4 |
| candidates/prompt | 4 |
| policy epochs | 1 |
| micro-batch | 4 |
| prompt/completion limit | 6144 / 64 tokens |
| sampling | temperature 0.9, top-p 0.95, top-k 50 |
| learning rate | $10^{-5}$ after 50-step warmup; cosine to $10^{-6}$ at step 800 |
| max gradient norm | 1.0 |

A complete run evaluates 16,000 online prompts and 64,000 candidates.

## 6. Adaptive KL

With target 0.05, EMA coefficient 0.1, initial $\beta=0.10$, and update interval 10,

$$
\bar D_k=0.1D_k+0.9\bar D_{k-1}.
$$

$\beta$ is multiplied by 1.5 when $\bar D>0.10$, divided by 1.5 when $\bar D<0.025$, and clipped to $[0.02,1.0]$. If $\bar D>0.20$ for 25 consecutive steps, training saves a complete checkpoint and stops.

## 7. Validation and deployment

Before the first update and every 50 steps, the main process evaluates the frozen GRPO-validation set with greedy decoding. Checkpoints are ranked lexicographically by

$$
(Pass@1,\ \overline{R_{IAE}},\ \overline R,\ -\bar D_{KL}).
$$

If no GRPO checkpoint beats step 0, the manifest selects SFT. The current completed run selected step 100.

Internal checkpoint validation is performed in the 4-bit Transformers training stack. Final SFT and GRPO benchmarks use the same vLLM path: SFT loads the exported bf16 SFT model, while GRPO loads that same model plus the selected LoRA adapter. Therefore absolute internal-validation Pass@1 values are for within-run checkpoint ordering and must not be compared directly with the bf16/vLLM paper benchmark.

## 8. Resume and audit

Every `checkpoints/checkpoint-N/` stores Accelerate policy/optimizer state, framework RNG, adaptive-KL state, and the prompt-generator state for every distributed process. Exact resume requires the same process count and unchanged optimization configuration; only the cumulative target step and logging/checkpoint intervals may change.

`training_manifest.json`, `reward_metadata.json`, `trainer_log.jsonl`, `validation_log.jsonl`, rollout logs, complete checkpoints, the selected adapter, source SFT manifest, Git commit, and `uv.lock` form the audit record.
