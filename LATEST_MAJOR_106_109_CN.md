# 106-109 大方向实验记录：候选集合 Meta 专家与分桶校准

日期：2026-05-15

## 背景

101-105 主要围绕“多教师一致伪在线历史重放”和源节点分桶专家门控。

本轮换一个更独立的方向：候选集合自身的无监督 meta 信号。

核心假设：

```text
官方候选集不是随机列表。
候选节点在 test 候选集合中的重复出现、源内重复出现、候选列均值、重复候选、与强教师 top1 的一致性，可能携带候选生成器先验。
```

注意：

```text
该方向不使用测试标签。
不把未来真实交互写入历史。
只使用公开 test.csv 的候选集合结构和已有提交分数作为教师信号。
```

## 新增脚本

```text
make_candidate_meta_expert.py
blend_candidate_meta_gate.py
```

## 106：纯候选集合 Meta 专家

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python make_candidate_meta_expert.py \
  --data-root data/official_raw \
  --teacher-dirs outputs/website_submission_89_major_teacher88_online_top4q80_40 outputs/website_submission_94_major_combo_mlp_consensus_boost outputs/website_submission_97_major_dual_mlp_strong_boost \
  --output-dir outputs/website_submission_106_major_candidate_meta_expert \
  --mode rank \
  --chunk-size 5000 \
  --recent-limit 24 \
  --w-srcfreq 2.8 \
  --w-globalfreq 0.20 \
  --w-mean-col 0.50 \
  --w-col-prior 0.05 \
  --w-recent 1.5 \
  --w-teacher-top 1.25 \
  --w-row-dup 2.5
```

提交包：

```text
submissions/106_major_candidate_meta_expert/result.zip
```

统计：

```text
dataset1 teacher_hit_rate = 0.975234
dataset2 teacher_hit_rate = 0.964314
dataset2 duplicate_row_rate = 0.043312
dataset2 unique_src = 2180
dataset2 src_rows_p50 = 21
dataset2 src_rows_p90 = 200
```

相对差异：

```text
106 vs 89: d2_mad=0.28588065, top1_change=0.087629
106 vs 94: d2_mad=0.28444630, top1_change=0.044042
106 vs 97: d2_mad=0.27308913, top1_change=0.122083
```

解释：106 的分数分布与主线差异极大，但 top1 并没有完全偏离，说明它更像候选生成器校准专家，不适合直接作为稳健提交，适合作为强专家参与融合。

## 107：97 与 106 的 rank 融合

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_rank_consensus.py \
  --base-dir outputs/website_submission_97_major_dual_mlp_strong_boost \
  --expert-dirs outputs/website_submission_106_major_candidate_meta_expert \
  --output-dir outputs/website_submission_107_major_candidate_meta_rank_fusion \
  --weight 0.30 \
  --sharpen 0.0 \
  --chunksize 4096
```

提交包：

```text
submissions/107_major_candidate_meta_rank_fusion/result.zip
```

相对差异：

```text
107 vs 89:  d2_mad=0.10324745, top1_change=0.103852
107 vs 94:  d2_mad=0.10015653, top1_change=0.046669
107 vs 97:  d2_mad=0.08192674, top1_change=0.065709
107 vs 104: d2_mad=0.08840565, top1_change=0.063069
106 vs 107: d2_mad=0.19116239, top1_change=0.068596
```

## 108：候选 Meta 稳健门控

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_candidate_meta_gate.py \
  --base-dir outputs/website_submission_97_major_dual_mlp_strong_boost \
  --meta-dir outputs/website_submission_106_major_candidate_meta_expert \
  --replay-gate-dir outputs/website_submission_104_major_adaptive_replay_gate_stable \
  --data-root data/official_raw \
  --output-dir outputs/website_submission_108_major_candidate_meta_gate_stable \
  --mode stable \
  --chunksize 4096
```

提交包：

```text
submissions/108_major_candidate_meta_gate_stable/result.zip
```

统计：

```text
top1_change_vs_97 = 0.029273
bucket_hot_agree = 66792
bucket_same_top = 37547
bucket_cold_agree = 27946
bucket_duplicate_agree = 4896
bucket_protect = 16239
```

相对差异：

```text
108 vs 89:  d2_mad=0.09353169, top1_change=0.137713
108 vs 94:  d2_mad=0.09035374, top1_change=0.079462
108 vs 97:  d2_mad=0.07232345, top1_change=0.029273
108 vs 104: d2_mad=0.07805841, top1_change=0.031078
108 vs 106: d2_mad=0.20117015, top1_change=0.097543
```

解释：108 是校准型大改，分数分布变化明显，但 top1 基本保护 97。适合优先提交验证。

## 109：候选 Meta 大胆门控

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_candidate_meta_gate.py \
  --base-dir outputs/website_submission_97_major_dual_mlp_strong_boost \
  --meta-dir outputs/website_submission_106_major_candidate_meta_expert \
  --replay-gate-dir outputs/website_submission_103_major_adaptive_replay_gate_bold \
  --data-root data/official_raw \
  --output-dir outputs/website_submission_109_major_candidate_meta_gate_bold \
  --mode bold \
  --chunksize 4096
```

提交包：

```text
submissions/109_major_candidate_meta_gate_bold/result.zip
```

统计：

```text
top1_change_vs_97 = 0.068589
bucket_hot_meta_replay_agree = 70081
bucket_same_top_meta_smooth = 68027
bucket_base_protect = 11325
bucket_cold_meta = 2205
bucket_duplicate_meta_replay = 1782
```

相对差异：

```text
109 vs 89:  d2_mad=0.11766058, top1_change=0.100587
109 vs 94:  d2_mad=0.11519237, top1_change=0.042752
109 vs 97:  d2_mad=0.10105684, top1_change=0.068583
109 vs 104: d2_mad=0.10498346, top1_change=0.054282
109 vs 106: d2_mad=0.17308152, top1_change=0.065480
```

## 格式校验

106、107、108、109 均已检查：

```text
result.zip 内含 dataset1.csv 和 dataset2.csv
dataset1.csv: 61051 x 100
dataset2.csv: 153420 x 100
概率范围在 [0, 1]
NaN = 0
```

## 提交建议

如果只提交一个，优先：

```text
submissions/108_major_candidate_meta_gate_stable/result.zip
```

推荐顺序：

```text
1. 108_major_candidate_meta_gate_stable
2. 109_major_candidate_meta_gate_bold
3. 107_major_candidate_meta_rank_fusion
4. 106_major_candidate_meta_expert
```

理由：

```text
108 是最稳的候选集合校准包，top1 变化小但分数分布大改。
109 更大胆，适合验证 meta 专家是否能带来线上突破。
107 是简单 rank 融合，作为中间对照。
106 纯专家风险最大，不建议优先提交，除非需要快速判断候选 meta 方向上限。
```

如果 108/109 有明显提升，下一阶段应继续候选集合 meta 方向：加入源节点分组温度、重复候选专门规则、dataset1/dataset2 分开权重。
