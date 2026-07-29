# Routh–Hurwitz 与精确时滞相位裕度检查

English version: [routh_hurwitz_fopdt_sopdt.md](routh_hurwitz_fopdt_sopdt.md)

## 1. 适用范围

本文说明 `src/llmpidtuner/training/rewards.py` 中实际实现的稳定性筛查。它用于 PI-GRPO 奖励，但不等同于时域仿真器。时域仿真仍使用控制器输出历史表示真实时滞；只有 Routh–Hurwitz 检查对非零时滞作近似。

过程模型为

$$
G_1(s)=\frac{K e^{-\theta s}}{Ts+1},\qquad
G_2(s)=\frac{K e^{-\theta s}}{(\tau_1s+1)(\tau_2s+1)},
$$

并采用并联式 PID：

$$
C(s)=K_p+\frac{K_i}{s}+K_ds
=\frac{K_ds^2+K_ps+K_i}{s}.
$$

## 2. 特征多项式

当 $\theta>10^{-12}$ s 时，Routh 检查使用一阶 Padé 近似

$$
e^{-\theta s}\approx\frac{1-ds}{1+ds},\qquad d=\frac{\theta}{2}.
$$

### 2.1 积分项有效的 FOPDT

特征多项式为

$$
a_3s^3+a_2s^2+a_1s+a_0=0,
$$

其中

$$
\begin{aligned}
a_3&=Td-KdK_d,\\
a_2&=T+d+K(K_d-dK_p),\\
a_1&=1+K(K_p-dK_i),\\
a_0&=KK_i.
\end{aligned}
$$

式中的 $KdK_d$ 表示过程增益 $K$、$d$ 与微分增益 $K_d$ 三者的乘积。

### 2.2 积分项有效的 SOPDT

令 $b_2=\tau_1\tau_2$、$b_1=\tau_1+\tau_2$，特征多项式为

$$
a_4s^4+a_3s^3+a_2s^2+a_1s+a_0=0,
$$

其中

$$
\begin{aligned}
a_4&=b_2d,\\
a_3&=b_2+b_1d-KdK_d,\\
a_2&=b_1+d+K(K_d-dK_p),\\
a_1&=1+K(K_p-dK_i),\\
a_0&=KK_i.
\end{aligned}
$$

### 2.3 精确的低阶分支

当 $K_i\le10^{-12}$ 时，代码先约去控制器在原点的公共极点，再构造相应的 PD 或 P 特征多项式，避免人为引入零根。当 $\theta\le10^{-12}$ s 时，代码不引入 Padé 因子，而使用精确的无时滞多项式。这些分支会在适用时降低多项式阶次。

## 3. 数值 Routh 检查

代码先用最高次项系数归一化系数向量。相对容差为 $10^{-9}$，并分别按归一化前后最大的系数幅值缩放。Routh 表第一列均大于容差时判为稳定；出现符号变化时判为不稳定。第一列元素接近零时判为临界稳定，全零行或非有限 Routh 表判为无法确定。临界稳定和无法确定均作保守拒绝。当前实现不会对异常行代入小量 $\epsilon$，也不会构造辅助多项式继续判别。

## 4. 精确时滞相位裕度

相位裕度检查不使用 Padé 时滞，而直接计算精确因子 $e^{-j\omega\theta}$。搜索区间为

$$
\omega\in\left[10^{-5}/T_c,\ 10^5/T_c\right],
$$

先使用 2,048 个对数等距点检测全部单位增益交越区间，再进行 64 次二分细化。FOPDT 取 $T_c=T$，SOPDT 取 $T_c=\tau_1+\tau_2$。若存在多个交越点，采用其中最小的相位裕度。

检测到的相位裕度 $\le0^\circ$ 时，候选不安全。若配置区间内没有检测到交越点，该项不直接拒绝候选，但稳定性奖励为零。

对具有交越点的安全候选，稳定性分量为

$$
R_{stab}=\begin{cases}
PM/60,&0<PM\le30^\circ,\\
0.5+(PM-30)/30,&30^\circ<PM\le45^\circ,\\
1,&PM>45^\circ.
\end{cases}
$$

## 5. 在 PI-GRPO 奖励中的作用

只有 Routh 状态为稳定，且所有检测到的精确时滞单位增益交越点均具有正相位裕度时，候选才通过频域安全筛查。解析、PID 参数符号以及时域仿真的有限性和发散检查独立执行。任一安全条件失败都会获得奖励 $-1$；安全候选再按照稳定性、响应性能、IAE 改善、增益正则化和输出格式评分，详见 [grpo_training_protocol_zh.md](grpo_training_protocol_zh.md)。

奖励元数据会保留 Routh 特征多项式与表格、精确时滞交越分析、相位裕度和拒绝原因，便于审计。
