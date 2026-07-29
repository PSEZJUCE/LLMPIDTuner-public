# PI-GRPO 训练、奖励与检查点协议

English version: [grpo_training_protocol.md](grpo_training_protocol.md)

## 1. 适用范围

PI-GRPO 从 balanced SFT 模型出发，在线生成训练 prompt，优化模型给出的第一组 PID 参数。论文使用的固定 100 个 FOPDT 和 100 个 SOPDT 测试案例不参与训练、超参数选择或检查点选择；检查点由另一组固定的 100+100 验证案例选择。

## 2. 策略更新

对每个 prompt $q$，rollout policy 生成 $G=4$ 个候选 $o_i$，其有效 completion token 数为 $T_i$。候选奖励在同一 prompt 组内标准化，得到相对优势 $A_i$。token 级概率比为

$$
\rho_{i,t}(\Theta)=
\frac{\pi_\Theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{old}(o_{i,t}\mid q,o_{i,<t})},
$$

相对于冻结 SFT reference policy 的非负 KL 估计为

$$
D_{KL,i,t}=\exp(r^{ref}_{i,t}-r^\Theta_{i,t})-(r^{ref}_{i,t}-r^\Theta_{i,t})-1.
$$

实现对所有候选的有效 completion token 作整体平均：

$$
J(q)=\frac{1}{\sum_iT_i}\sum_{i=1}^{G}\sum_{t=1}^{T_i}
\left[
\min\left(\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\right)
-\beta_kD_{KL,i,t}
\right],
$$

其中 $\epsilon=0.2$。$\pi_{old}$ 是本次更新前的 rollout policy，$\pi_{ref}$ 是冻结的 SFT 模型。

## 3. 安全性与任务成功

以下任一情况使候选获得不安全奖励 $-1$：无法解析；不满足 $K_p>0$、$K_i\ge0$、$K_d\ge0$；仿真结果非有限或超过发散阈值；未通过基于一阶 Padé 时滞近似的 Routh–Hurwitz 检验；或存在单位增益交越点且精确时滞相位裕度不大于零。

若 Routh 稳定但在配置的搜索区间内未检测到单位增益交越点，候选仍视为安全，但稳定性分量为零。任务成功还要求响应能够进入并保持在 5% 稳态带内、$M_p<15\%$ 且 $e_{ss}<1\%$。

## 4. 质量分数与三段奖励

对安全候选，

$$
Q=0.35R_{stab}+0.35R_{perf}+0.15R_{IAE}+0.10R_{gain}+0.05R_{format}.
$$

性能分量不重复包含 IAE：

$$
R_{perf}=\operatorname{clip}\left(\frac{5}{12}R_{os}+\frac13R_{set}+\frac14R_{ss},-1,1\right).
$$

令 $q_s=t_s/t_{s,ref}$，其中

$$
t_{s,ref}=\theta-\ln(0.05)\lambda_{balanced},\qquad
\lambda_{balanced}=2\max(\theta,0.1T_c).
$$

代码中的分段点与论文 SI 一致：超调量为 5/15/30%，调节时间比为 1/2，稳态误差为 0.1/1/2%。

IAE 改善项为

$$
\Delta_{IAE}=\frac{IAE_{before}-IAE_{after}}{\max(|IAE_{before}|,10^{-8})},
\qquad R_{IAE}=\tanh(2\Delta_{IAE}).
$$

增益正则化使用无量纲向量

$$
g=(KK_p,\;KK_iT_c,\;KK_d/T_c),
$$

其中 FOPDT 取 $T_c=T$，SOPDT 取 $T_c=\tau_1+\tau_2$。参考值 $g_{ref}$ 由每个过程自己的 balanced IMC 控制器独立计算，各维下限为 0.01；上限为 $(4g_{p,ref},4g_{i,ref},6g_{d,ref})$。若

$$
e_j=\max\left[0,\ln\frac{g_j+10^{-12}}{g_{j,lim}+10^{-12}}\right],
$$

则

$$
R_{gain}=-\min\left(1,\sum_je_j^2\right).
$$

最终奖励为

$$
R=\begin{cases}
-1,&\text{不安全},\\[2mm]
-0.5+0.25(Q+1),&\text{安全但未成功},\\[2mm]
0.5+0.25(Q+1),&\text{安全且成功}.
\end{cases}
$$

三个分支的值域分别为 $-1$、$[-0.5,0]$ 和 $[0.5,1]$。

## 5. 训练配置

权威配置文件为 `configs/grpo/qwen3_0p6b_pid.yaml`。

| 参数 | 数值 |
|---|---:|
| QLoRA | NF4 4-bit，double quantization |
| LoRA rank / alpha / dropout | 16 / 32 / 0 |
| 累计更新步数 | 800 |
| 分布式进程数 | 5 |
| 每进程每步 prompt 数 | 4 |
| 每个 prompt 的候选数 | 4 |
| policy epoch | 1 |
| micro-batch | 4 |
| prompt/completion 长度上限 | 6144 / 64 tokens |
| 采样参数 | temperature 0.9，top-p 0.95，top-k 50 |
| 学习率 | 50 步 warmup 后为 $10^{-5}$；至第 800 步余弦衰减到 $10^{-6}$ |
| 最大梯度范数 | 1.0 |

完整训练共评估 16,000 个在线 prompt 和 64,000 个候选 completion。

## 6. 自适应 KL

KL 目标为 0.05，EMA 系数为 0.1，初始 $\beta=0.10$，每 10 步更新一次：

$$
\bar D_k=0.1D_k+0.9\bar D_{k-1}.
$$

当 $\bar D>0.10$ 时，$\beta$ 乘以 1.5；当 $\bar D<0.025$ 时，$\beta$ 除以 1.5；随后限制在 $[0.02,1.0]$。若 $\bar D>0.20$ 连续维持 25 步，训练会保存完整检查点并停止。

## 7. 验证与最终评估

主进程在第 0 步以及之后每 50 步，使用 greedy decoding 在固定 GRPO 验证集上评估。检查点按以下指标作字典序排序：

$$
(Pass@1,\ \overline{R_{IAE}},\ \overline R,\ -\bar D_{KL}).
$$

若没有 GRPO 检查点优于第 0 步，manifest 会选择 SFT。当前已完成实验选择的是第 100 步检查点。

内部检查点验证使用训练栈中的 4-bit Transformers 推理；最终 SFT 与 GRPO 基准均使用同一 vLLM 路径：SFT 加载导出的 bf16 模型，GRPO 加载该 bf16 模型和选定的 LoRA adapter。因此，内部验证的 Pass@1 只用于同一训练过程内的检查点排序，不能与论文中 bf16/vLLM 基准的绝对通过率直接比较。

## 8. 恢复训练与审计

每个 `checkpoints/checkpoint-N/` 保存 Accelerate policy/optimizer 状态、框架随机数状态、自适应 KL 状态，以及每个分布式进程的 prompt 生成器状态。精确恢复要求进程数和优化配置不变；只允许调整累计目标步数及日志、保存间隔。

`training_manifest.json`、`reward_metadata.json`、`trainer_log.jsonl`、`validation_log.jsonl`、rollout 日志、完整检查点、选定 adapter、源 SFT manifest、Git commit 和 `uv.lock` 构成审计记录。
