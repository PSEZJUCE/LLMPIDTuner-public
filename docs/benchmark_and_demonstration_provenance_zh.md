# 基准测试与 Demonstration 来源记录

English version: [benchmark_and_demonstration_provenance.md](benchmark_and_demonstration_provenance.md)

本文档简要记录当前 `perturbed_imc_delay_stratified_v1` 实验协议，供论文复现与结果审计使用。详细的案例生成规则见 [case_generation_reproducibility_zh.md](case_generation_reproducibility_zh.md)。

## 1. 协议范围

当前协议消除了早期流程中的四类混杂因素：

1. 初始 PID 是 balanced IMC 参数的过程特异性扰动，不再固定为 `(1, 0.1, 0.01)`。
2. demonstration、论文评估、GRPO 验证和训练案例彼此隔离。
3. 时滞随过程时间尺度变化，每条 prompt 的诊断均根据本案例的过程、时滞和初始 PID 重新计算。
4. 所有方法共享仿真器、指标定义、初始案例和成功判据。

## 2. 过程与时滞范围

FOPDT 取 $K\in[0.2,0.9]$、$T\in[100,600]$ s。SOPDT 取 $K\in[0.1,3.0]$、$\tau_{slow}\in[10,100]$ s、$\tau_{fast}/\tau_{slow}\in[0.1,1]$。

$$
T_c=\begin{cases}T,&\mathrm{FOPDT},\\ \tau_1+\tau_2,&\mathrm{SOPDT},\end{cases}
\qquad \rho=\theta/T_c.
$$

$1\le\theta\le200$ s 且 $\rho\le0.8$。每十个同类案例包括 4 个 low、3 个 medium、2 个 delay-dominant 和 1 个 strong-delay 案例。

## 3. IMC 参考与初始故障

不同控制风格的 IMC 使用

$$
\lambda=m\max(\theta,0.1T_c),
$$

aggressive、balanced、conservative 分别取 $m=0.8$、2.0、5.0。balanced IMC 控制器提供扰动参考和 SFT 标签，但不是初始控制器。

十类确定性 PID 故障方向覆盖比例和积分作用的过强/过弱，以及若干 P/I/D 组合故障。扰动强度在 1.01 至 100 的 160 点几何网格上搜索。案例仅在初始响应有限、未成功、满足 $|y|\le3$ 且 IAE 至少为 balanced 参考的 1.5 倍时接受。

严重度按 $IAE_{initial}/IAE_{balanced}$ 定义：$[1.5,2)$ 为 mild，$[2,5)$ 为 moderate，$[5,\infty)$ 为 severe。每个冻结的 100 案例集合包括 30 个 mild、50 个 moderate、20 个 severe 案例，并包含每类故障各 10 个。

## 4. 仿真与成功判据

当前协议统一使用设定值 1、仿真时长 4000 s、40,001 个采样点、$\Delta t=0.1$ s 和 `max_abs_output=3`。非整数时滞由控制器输出历史线性插值。FOPDT 使用显式 Euler；SOPDT 使用顺序 Euler，即第二惯性环节读取本步刚更新的第一状态。

IAE 在完整时域上作矩形积分。稳态误差使用最后 100 个采样点的均值。调节完成要求进入 $\pm5\%$ 带后不再离开。成功要求仿真有限、调节时间有限、超调量严格小于 15%、稳态误差严格小于 1%。

## 5. 冻结集合

| 集合 | FOPDT | SOPDT | 用途 |
|---|---:|---:|---|
| demonstration | seed 51001，10 | seed 52001，10 | 冻结 few-shot 上下文 |
| paper evaluation | seed 61001，100 | seed 62001，100 | 最终方法比较 |
| GRPO validation | seed 71001，100 | seed 72001，100 | 检查点选择 |

source 文件位于 `cases/protocol/perturbed_imc_delay_stratified/sources/`。manifest 记录种子、case hash、调度表和资产 hash。SFT 与 GRPO 显式排除所有冻结 source。论文评估集只用于最终观察，不能选择 epoch、checkpoint 或超参数。

## 6. 冻结 Demonstration

每类过程使用相同的十个 source 案例渲染为：

```text
balanced/full
balanced/kpi3
balanced/numeric8
aggressive/full
conservative/full
```

prompt 变体只改变所呈现的响应信息。风格变体保留相同初始案例，但按所请求的 IMC 风格重新计算推荐 PID。运行时通过配套 manifest 校验 source 与 prompt hash。

## 7. 训练隔离与审计

SFT 包含 20,000 个 FOPDT 和 20,000 个 SOPDT balanced 样本，生成种子为 81001。GRPO 从该 balanced SFT 模型出发，以种子 91001 在线采样 balanced first-turn prompt。冻结的 100+100 GRPO 验证集在训练前及之后每 50 步评估一次，但不参与梯度更新。

应共同归档 Git commit、`uv.lock`、source YAML、demonstration manifest、运行配置、模型与数据 manifest、检查点、日志和最终曲线。protocol ID、prompt hash、仿真网格或指标定义不同的结果不得合并统计。
