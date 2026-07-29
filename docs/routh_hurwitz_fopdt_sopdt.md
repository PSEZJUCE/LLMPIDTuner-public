# Routh–Hurwitz and Exact-Delay Phase-Margin Checks

Chinese version: [routh_hurwitz_fopdt_sopdt_zh.md](routh_hurwitz_fopdt_sopdt_zh.md)

## 1. Scope

This document describes the stability screen implemented in `src/llmpidtuner/training/rewards.py`. It is used by the PI-GRPO reward and is not the time-domain simulator itself. The simulator retains the exact stored-history delay; only the Routh–Hurwitz check approximates a nonzero delay.

The plant models are

$$
G_1(s)=\frac{K e^{-\theta s}}{Ts+1},\qquad
G_2(s)=\frac{K e^{-\theta s}}{(\tau_1s+1)(\tau_2s+1)},
$$

with the parallel PID controller

$$
C(s)=K_p+\frac{K_i}{s}+K_ds
=\frac{K_ds^2+K_ps+K_i}{s}.
$$

## 2. Characteristic polynomials

For $\theta>10^{-12}$ s, the Routh check uses the first-order Padé approximation

$$
e^{-\theta s}\approx\frac{1-ds}{1+ds},\qquad d=\frac{\theta}{2}.
$$

### 2.1 FOPDT with active integral action

The characteristic polynomial is

$$
a_3s^3+a_2s^2+a_1s+a_0=0,
$$

where

$$
\begin{aligned}
a_3&=Td-KdK_d,\\
a_2&=T+d+K(K_d-dK_p),\\
a_1&=1+K(K_p-dK_i),\\
a_0&=KK_i.
\end{aligned}
$$

Here the product $KdK_d$ means process gain $K$ times $d$ times derivative gain $K_d$.

### 2.2 SOPDT with active integral action

Let $b_2=\tau_1\tau_2$ and $b_1=\tau_1+\tau_2$. The characteristic polynomial is

$$
a_4s^4+a_3s^3+a_2s^2+a_1s+a_0=0,
$$

where

$$
\begin{aligned}
a_4&=b_2d,\\
a_3&=b_2+b_1d-KdK_d,\\
a_2&=b_1+d+K(K_d-dK_p),\\
a_1&=1+K(K_p-dK_i),\\
a_0&=KK_i.
\end{aligned}
$$

### 2.3 Exact lower-order cases

When $K_i\le10^{-12}$, the common controller pole at the origin is cancelled before constructing the polynomial, so the check uses the corresponding PD or P polynomial rather than an artificial zero root. When $\theta\le10^{-12}$ s, no Padé factor is introduced and the exact zero-delay polynomial is used. These branches reduce the polynomial degree by one where appropriate.

## 3. Numerical Routh test

The coefficient vector is normalized by its leading coefficient. The relative tolerance is $10^{-9}$, scaled by the largest coefficient magnitude before and after normalization. A positive first column above this tolerance is classified as stable; a sign change is unstable. A near-zero first-column entry is classified as marginal, and an all-zero row or non-finite table is classified as indeterminate. Marginal and indeterminate results are conservatively rejected. The implementation does not substitute an epsilon or construct an auxiliary polynomial for these exceptional cases.

## 4. Exact-delay phase margin

The phase-margin check does not use Padé delay. It evaluates the exact factor $e^{-j\omega\theta}$ and searches all detected unity-gain crossovers over

$$
\omega\in\left[10^{-5}/T_c,\ 10^5/T_c\right],
$$

using 2,048 logarithmically spaced points followed by 64 bisection iterations. Here $T_c=T$ for FOPDT and $T_c=\tau_1+\tau_2$ for SOPDT. If multiple crossovers exist, the minimum phase margin is used.

A detected phase margin $\le0^\circ$ is unsafe. If no crossover is detected in the configured range, the candidate is not rejected by this check, but its stability reward is zero.

For a safe candidate with detected phase margin $PM$, the stability component is

$$
R_{stab}=\begin{cases}
PM/60,&0<PM\le30^\circ,\\
0.5+(PM-30)/30,&30^\circ<PM\le45^\circ,\\
1,&PM>45^\circ.
\end{cases}
$$

## 5. Role in the PI-GRPO reward

A candidate passes the frequency-domain safety screen only when its Routh status is stable and every detected exact-delay crossover has positive phase margin. Parsing, PID-sign, and time-domain finite/divergence checks are applied separately. Any failed safety condition receives reward $-1$; a safe candidate is then scored by stability, response performance, IAE improvement, gain regularization, and output format as described in [grpo_training_protocol.md](grpo_training_protocol.md).

The Routh polynomial and table, exact-delay crossover analysis, phase margin, and rejection reason are retained in the reward metadata for audit.
