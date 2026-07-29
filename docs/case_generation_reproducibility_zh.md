# PID 案例与训练数据生成复现说明

英文版：[case_generation_reproducibility.md](case_generation_reproducibility.md)

本文档描述 `perturbed_imc_delay_stratified_v1` 的当前实现。协议代码、冻结 source YAML 和各自 manifest 是最终依据；旧协议与 `Archive/` 结果不得并入当前统计。

## 1. 数据集合与隔离

| 集合 | FOPDT | SOPDT | 用途 |
|---|---:|---:|---|
| demonstration | 10，seed 51001 | 10，seed 52001 | 冻结 few-shot 提示 |
| paper evaluation | 100，seed 61001 | 100，seed 62001 | 最终方法比较 |
| GRPO validation | 100，seed 71001 | 100，seed 72001 | checkpoint 选择 |
| SFT | 20,000 | 20,000 | balanced 监督训练 |
| GRPO | 在线采样 | 在线采样 | balanced first-turn 优化 |

SFT 和 GRPO 在线生成器排除全部冻结 demonstration、evaluation 和 GRPO-validation source 中的 case hash。论文 evaluation 不参与训练、超参数调整或 checkpoint 选择。

## 2. 模型、采样与 IMC 参考

$$
G_f(s)=\frac{K e^{-\theta s}}{Ts+1},\quad K\in[0.2,0.9],\quad T\in[100,600]\ \mathrm{s},
$$

$$
G_s(s)=\frac{K e^{-\theta s}}{(\tau_1s+1)(\tau_2s+1)}.
$$

SOPDT 使用 $K\in[0.1,3]$、$\tau_{slow}\in[10,100]$ s、$\tau_{fast}/\tau_{slow}\in[0.1,1]$。连续变量来自固定种子的 SciPy Latin-hypercube 序列。定义

$$
T_c=\begin{cases}T,&\mathrm{FOPDT},\\ \tau_1+\tau_2,&\mathrm{SOPDT},\end{cases}
\qquad \rho=\theta/T_c.
$$

$1\le\theta\le200$ s 且 $\rho\le0.8$。每 10 个同类案例包含 4 low、3 medium、2 delay-dominant 和 1 strong-delay，区间分别为 $[0,0.05)$、$[0.05,0.2)$、$[0.2,0.5)$、$[0.5,0.8]$。

风格参数为 $\lambda=m\max(\theta,0.1T_c)$，aggressive、balanced、conservative 的 $m$ 分别为 0.8、2.0、5.0。FOPDT 使用

$$
K_p=\frac{T+\theta/2}{K(\lambda+\theta/2)},\quad T_i=T+\theta/2,\quad T_d=\frac{T\theta}{2T+\theta},
$$

SOPDT 使用

$$
K_p=\frac{\tau_1+\tau_2}{K(\lambda+\theta)},\quad T_i=\tau_1+\tau_2,\quad T_d=\frac{\tau_1\tau_2}{\tau_1+\tau_2}.
$$

两者均通过 $K_i=K_p/T_i$、$K_d=K_pT_d$ 转为并联增益。balanced 控制器是扰动参考与 SFT 标签，不是统一初始 PID。

## 3. 异常初始 PID 与验收

balanced 增益乘以 1.01 到 100 的 160 点确定性几何扰动。十类方向为：`p_high`、`p_low`、`i_high`、`i_low_or_off`、`i_high_d_high`、`p_high_d_low`、`p_high_i_high`、`p_low_i_high`、`i_high_d_low`、`p_low_i_high_d_low`。精确日程定义于 `src/llmpidtuner/experiment_protocol.py`，并写入 source manifest。

严重度由

$$
q_{IAE}=\frac{IAE_{initial}}{IAE_{balanced}}
$$

划分：$[1.5,2)$ 为 mild，$[2,5)$ 为 moderate，$[5,\infty)$ 为 severe。每个冻结 100-case 集合包含每类故障 10 个、30/50/20 的严重度和 40/30/20/10 的时滞层。

案例仅在下列条件全部满足时接受：初始和参考轨迹有限；参考 IAE 大于 0；初始响应未成功；$q_{IAE}\ge1.5$；初始输出绝对极值不超过 3；槽位匹配；case hash 未被排除且未重复。

## 4. 离散仿真

所有当前实验使用 $r=1$、4000 s、40,001 个采样点、$\Delta t=0.1$ s 和 `max_abs_output=3`。位置式并联 PID 为

$$
I_k=I_{k-1}+e_k\Delta t,\qquad
u_k=K_pe_k+K_iI_k+K_d\frac{e_k-e_{k-1}}{\Delta t}.
$$

$u_0=K_pe_0$，因此初始点没有微分冲击。非整数时滞由控制历史线性插值。令 $d=\theta/\Delta t$、$z=(k-1)-d$、$j=\lfloor z\rfloor$、$\eta=z-j$，则

$$
u_{d,k-1}=(1-\eta)u_j+\eta u_{j+1},
$$

$z<0$ 时取 $u_{d,k-1}=0$。

FOPDT 使用标准前向 Euler：

$$
y_k=y_{k-1}+\frac{\Delta t}{T}(Ku_{d,k-1}-y_{k-1}).
$$

SOPDT 使用固定的顺序 Euler：

$$
x_{1,k}=x_{1,k-1}+\frac{\Delta t}{\tau_1}(Ku_{d,k-1}-x_{1,k-1}),
$$

$$
x_{2,k}=x_{2,k-1}+\frac{\Delta t}{\tau_2}(x_{1,k}-x_{2,k-1}),\qquad y_k=x_{2,k}.
$$

第二状态使用本步已更新的 $x_{1,k}$，因此不是状态向量的同步 forward Euler。该顺序在全部实验中固定。

## 5. 指标与响应特征

$$
IAE=\Delta t\sum_{k=0}^{N-1}|e_k|,
$$

$$
M_p=100\max\left(0,\frac{\max_k y_k-r}{\max(|r|,10^{-12})}\right),
$$

$$
e_{ss}=100\frac{|\operatorname{mean}(y_{N-m:N})-r|}{\max(|r|,10^{-12})},\quad m=\min(100,N).
$$

调节时间是此后始终留在 $\pm5\%$ 带内的最早采样时刻。成功要求仿真有限、已调节、$M_p<15\%$ 且 $e_{ss}<1\%$，均使用严格小于号。

令 $s_r=\max(|r|,10^{-12})$，其余提示特征为：

1. **显著穿越次数**：删除 $|y_k-r|\le0.05s_r$ 的点，对剩余相邻偏差统计符号变化；不足两个点时为 0。
2. **衰减比**：在完整输出上用 `scipy.signal.find_peaks(prominence=0.01*s_r)` 找峰。前两峰为 $p_1,p_2$ 时，$A_r=|y_{p_2}-r|/\max(|y_{p_1}-r|,10^{-12})$；不足两峰时为 0。
3. **振荡周期**：$T_{osc}=t_{p_2}-t_{p_1}$；不足两峰时为 0。
4. **死区后的 63.2% 时间**：本文 $r=1$，报告首次满足 $t_k\ge\theta$、$y_k\ge0.632r$ 的 $t_k-\theta$；未达到时为 `N/A`，不做采样间插值。

## 6. SFT、GRPO 与审计

`configs/data/pid_sft_messages_40k.yaml` 固定生成两类过程各 20,000 条。每行以 0.5 概率尝试 feedback 历史；请求深度 1/2/3 的条件概率为 0.5/0.3/0.2。每层最多尝试 20 个有限且未成功的中间控制器。最终标签始终为该过程的 balanced IMC PID，格式为 `P:<value>; I:<value>; D:<value>`。

GRPO 基础 seed 为 91001，第 $r$ 个分布式进程使用 $91001+100003r$。五进程、800 steps、每进程每步 4 prompts、每 prompt 4 completions 的完整运行评估 16,000 个在线 prompts 和 64,000 个 completions。冻结 GRPO validation 每 50 steps 评估一次且不参与梯度更新。

```bash
uv run llmpidtuner build-protocol-assets --check
uv run llmpidtuner make-sft-data configs/data/pid_sft_messages_40k.yaml
uv run llmpidtuner validate-sft-data configs/sft/qwen3_0p6b_pid.yaml
```

归档时共同保存 Git commit、`uv.lock`、配置、source YAML、manifest、checkpoint、日志和运行结果。哈希以对应 manifest 为准，本文不重复维护静态副本。
