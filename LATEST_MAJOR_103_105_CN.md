# 103-105 大方向实验记录：源节点分桶专家门控与硬切换

日期：2026-05-15

## 背景

98、99、100 连续明显回退后，停止继续做 MLP 共识接管、rank transplant、topK 移植。

101-102 切到“多教师一致伪在线历史重放”：

```text
101: 严格一致伪在线重放
102: 97 + 101 rank 融合
```

本轮 103-105 继续沿新方向推进，但不再做统一权重融合，而是做按源节点和教师一致性的分桶专家门控。

## 新增脚本

```text
blend_adaptive_replay_gate.py
blend_hard_expert_switch.py
```

共同输入：

```text
base = outputs/website_submission_97_major_dual_mlp_strong_boost
replay = outputs/website_submission_101_major_consensus_replay_strict
stable teacher A = outputs/website_submission_89_major_teacher88_online_top4q80_40
stable teacher B = outputs/website_submission_94_major_combo_mlp_consensus_boost
```

分桶特征：

```text
source_total_count: 当前 src 在 test 中总出现次数
source_seen_before: 当前 src 在当前时间之前已出现次数
teacher_top1: 89/94/97/101 的 top1 一致性
base_margin / replay_margin: 行内 top1 与 top2 差距
```

源节点热度阈值：

```text
p50 = 21
p75 = 83
p90 = 200
```

## 103：大胆自适应 replay gate

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_adaptive_replay_gate.py \
  --base-dir outputs/website_submission_97_major_dual_mlp_strong_boost \
  --replay-dir outputs/website_submission_101_major_consensus_replay_strict \
  --stable-dirs outputs/website_submission_89_major_teacher88_online_top4q80_40 outputs/website_submission_94_major_combo_mlp_consensus_boost \
  --data-root data/official_raw \
  --output-dir outputs/website_submission_103_major_adaptive_replay_gate_bold \
  --mode bold \
  --chunksize 4096
```

提交包：

```text
submissions/103_major_adaptive_replay_gate_bold/result.zip
```

统计：

```text
avg_weight_to_101 = 0.530866
top1_change_vs_97 = 0.057926
bucket_replay_majority_hot = 90063
bucket_replay_majority_active = 15690
bucket_protect_base_majority = 21556
```

相对差异：

```text
103 vs 89:  d2_mad=0.01768952, top1_change=0.113355
103 vs 94:  d2_mad=0.01462018, top1_change=0.060266
103 vs 97:  d2_mad=0.02348846, top1_change=0.057926
103 vs 101: d2_mad=0.01741115, top1_change=0.183979
103 vs 102: d2_mad=0.01327116, top1_change=0.112019
```

## 104：稳健自适应 replay gate

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_adaptive_replay_gate.py \
  --base-dir outputs/website_submission_97_major_dual_mlp_strong_boost \
  --replay-dir outputs/website_submission_101_major_consensus_replay_strict \
  --stable-dirs outputs/website_submission_89_major_teacher88_online_top4q80_40 outputs/website_submission_94_major_combo_mlp_consensus_boost \
  --data-root data/official_raw \
  --output-dir outputs/website_submission_104_major_adaptive_replay_gate_stable \
  --mode stable \
  --chunksize 4096
```

提交包：

```text
submissions/104_major_adaptive_replay_gate_stable/result.zip
```

统计：

```text
avg_weight_to_101 = 0.363346
top1_change_vs_97 = 0.034643
bucket_replay_majority_hot = 90063
bucket_replay_majority_active = 15690
bucket_protect_base_majority = 22538
```

相对差异：

```text
104 vs 89:  d2_mad=0.02405207, top1_change=0.135888
104 vs 94:  d2_mad=0.01980268, top1_change=0.078503
104 vs 97:  d2_mad=0.01631638, top1_change=0.033953
104 vs 101: d2_mad=0.02472174, top1_change=0.207339
104 vs 102: d2_mad=0.00728051, top1_change=0.111850
```

## 105：硬专家切换

105 是本轮最激进版本，不再只做连续权重融合，而是按分桶直接切换专家：

```text
热源 + replay 与多数教师一致：直接用 101
base 与多数教师一致且 replay 冲突：保留 97
冷源：回退 94
中间分桶：使用 replay/base/stable 混合
```

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_hard_expert_switch.py \
  --base-dir outputs/website_submission_97_major_dual_mlp_strong_boost \
  --replay-dir outputs/website_submission_101_major_consensus_replay_strict \
  --stable-a-dir outputs/website_submission_89_major_teacher88_online_top4q80_40 \
  --stable-b-dir outputs/website_submission_94_major_combo_mlp_consensus_boost \
  --data-root data/official_raw \
  --output-dir outputs/website_submission_105_major_hard_expert_switch \
  --chunksize 4096
```

提交包：

```text
submissions/105_major_hard_expert_switch/result.zip
```

统计：

```text
top1_change_vs_97 = 0.087798
bucket_hard_replay_majority_hot = 90063
bucket_hard_base_protect = 22566
bucket_hard_stable_cold = 18212
```

相对差异：

```text
105 vs 89:  d2_mad=0.01089360, top1_change=0.083353
105 vs 94:  d2_mad=0.01320538, top1_change=0.050860
105 vs 97:  d2_mad=0.03258484, top1_change=0.087798
105 vs 101: d2_mad=0.00836951, top1_change=0.174221
105 vs 103: d2_mad=0.01089753, top1_change=0.037003
105 vs 104: d2_mad=0.01738568, top1_change=0.057815
```

## 格式校验

103、104、105 均已检查：

```text
result.zip 内含 dataset1.csv 和 dataset2.csv
dataset1.csv: 61051 x 100
dataset2.csv: 153420 x 100
概率范围在 [0, 1]
NaN = 0
```

## 提交建议

如果今天提交次数紧张，建议顺序：

```text
1. 104_major_adaptive_replay_gate_stable
2. 103_major_adaptive_replay_gate_bold
3. 105_major_hard_expert_switch
```

理由：

```text
104 风险最低，最接近 102/97，但已经加入分桶门控。
103 更大胆，101 权重更高，适合验证动态重放是否能提升。
105 是硬切换，变化最大，若 103/104 没有明显突破再提交它。
```

如果 103/104/105 都回退，说明“101 动态重放专家”本身可能在线无效；下一阶段应转向完全不同方向，例如候选位置先验、源节点分组概率校准、或按公开 leaderboard 反馈做专家族贝叶斯选择。
