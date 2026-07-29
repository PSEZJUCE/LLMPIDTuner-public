# Qwen3-0.6B SFT 技术路线

English version: [sft_technical_route.md](sft_technical_route.md)

本文档说明当前实现的监督微调数据、目标函数、数据划分、训练配置和审计产物。

## 1. 学习任务

模型根据冻结 demonstration、当前 PID、响应诊断、观测到的时滞和 requested balanced style，输出一组简洁参数：

```text
P:<Kp>; I:<Ki>; D:<Kd>
```

prompt 不包含传递函数参数。assistant 标签只包含 PID 三个数值，不含推理文本。Base、SFT 和 GRPO 模型在正式评估时均关闭 thinking。

## 2. 数据与隔离

`configs/data/pid_sft_messages_40k.yaml` 生成 `data/sft/pid_sft_messages_40000.jsonl`：

| 过程类型 | 样本数 |
|---|---:|
| FOPDT | 20,000 |
| SOPDT | 20,000 |
| 合计 | 40,000 |

所有记录均使用 balanced 风格、种子 81001 和冻结的 balanced/full demonstration。生成器排除 demonstration、论文评估和 GRPO 验证案例的 hash。每个标签均由该过程及其时滞独立计算的 balanced IMC PID 给出，并要求其仿真响应满足相同的可观测成功判据。

demonstration 是 prompt 上下文，不是额外的 20 条训练记录。

## 3. Initial 与 feedback 样本

每个生成案例首先形成 first-turn prompt。生成器以 0.5 的概率尝试加入 feedback 历史；请求深度 1、2、3 的条件概率分别为 0.5、0.3、0.2。中间 PID 在当前与目标控制器之间按对数尺度插值，并加入对数正态扰动。每个中间响应必须有限、未成功且满足 $|y|\le3$。

若无法找到合格的中间 PID，feedback 构造会提前停止。因此，实际 feedback 数量与深度分布以数据 manifest 的记录为准，不能直接由名义概率推断。最终 assistant 消息始终是 balanced IMC 目标。

## 4. Prompt 长度与 metadata

每条 JSONL 记录采用 OpenAI messages 结构和 metadata schema version 3。metadata 记录 sample kind、实际 feedback depth、风格、过程、时滞、当前 PID 与指标、目标 PID 与指标，以及 `target_method=imc_style`。

最大序列长度为 6144 tokens。预检 tokenization 会拒绝任何超长记录；训练不会静默截断 prompt 或标签。

## 5. 目标函数

SFT 最小化仅作用于 assistant token 的自回归交叉熵：

$$
\mathcal L_{SFT}=-\sum_{t=1}^{|y|}\log p_\theta(y_t\mid x,y_{<t}).
$$

该 loss 是 token 预测误差，而不是 PID 数值 MSE，也不直接包含 IAE 或稳定性损失。物理行为通过经过验证的 IMC 标签和后续 PI-GRPO 奖励进入训练流程。

## 6. 训练配置

权威配置为 `configs/sft/qwen3_0p6b_pid.yaml`。

| 参数 | 数值 |
|---|---:|
| base model | `Qwen/Qwen3-0.6B` |
| 精度与训练范围 | bf16，全参数训练 |
| max length | 6144 |
| learning rate | $10^{-5}$ |
| epochs | 3 |
| 每 GPU batch | 1 sequence |
| gradient accumulation | 8 |
| GPU processes | 5 |
| global optimizer batch | 40 sequences |
| validation | 5%，按过程类型和 sample kind 分层 |
| warmup ratio | 0.03 |
| weight decay | 0.01 |
| max gradient norm | 1.0 |
| evaluation/save interval | 50 optimizer steps |
| seed | 42 |

38,000 条训练记录、global batch 40 对应每个 epoch 950 个 optimizer steps，三个 epoch 共 2,850 步。`load_best_model_at_end` 恢复验证 loss 最低的检查点。冻结的论文评估集不用于选择 epoch。

## 7. 命令与产物

```bash
uv sync --extra training --extra training-stack
sbatch scripts/slurm/generate_qwen3_0p6b_training_data_40k.sbatch
sbatch scripts/slurm/train_qwen3_0p6b_sft_5gpu.sbatch
```

默认输出目录为 `outputs/sft/qwen3_0p6b_pid`。应保留 JSONL 与 manifest、SFT 配置、Slurm 日志、`training_manifest.json`、最佳导出模型、训练/验证 loss 历史、Git commit 和 `uv.lock`。

## 8. 与 PI-GRPO 的边界

SFT 学习 balanced IMC 的响应到 PID 映射。PI-GRPO 从导出的 balanced SFT 模型开始，在线生成 balanced first-turn prompt，并使用仿真、Padé–Routh、精确时滞相位裕度、性能、IAE、格式和增益正则化奖励。它不读取预生成的 GRPO 训练 JSONL，也不复用论文评估集。

当前奖励与检查点协议见 [grpo_training_protocol_zh.md](grpo_training_protocol_zh.md)。
