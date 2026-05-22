# 101-102 大方向实验记录：多教师一致伪在线历史重放

日期：2026-05-15

## 为什么停止 98-100 方向

线上反馈：

```text
98 约 1.173982430668677
99 约 1.1757092317721356
100 1.1788037969609755
```

连续三轮明显低于 94-97，说明问题不是参数没调好，而是方向本身不稳：

1. 98/99 的 rank transplant 过强，直接把 MLP/专家排序移植到最终输出，破坏了已在线验证有效的 94/97 顺序。
2. 100 的 triple consensus MLP 虽然更保守，但仍然把 MLP 共识当作主要可靠信号；线上结果显示这个共识和隐藏测试分布不匹配。
3. 继续在 MLP 共识、topK 移植、全行接管上微调，大概率只是小幅波动或继续回退，不适合作为冲 1.3 的主线。

结论：MLP 可以继续作为弱专家或辅助信号，但不再作为主排序接管器。

## 新方向

本轮转向“多教师一致伪在线历史重放”：

1. 使用线上确认较强的提交作为教师：80、89、94、97。
2. 排除已证明回退的 98、99、100。
3. 对 dataset2 按测试时间顺序处理。
4. 对每一行候选，只在多个教师对 top 候选高度一致，且教师 margin 足够高时，才把该候选当作伪发生边。
5. 伪边只在时间切换后写入历史统计，避免同一时间片内互相泄漏。
6. 更新后仍使用官方 baseline 派生的 Jittor combo 特征和线性模型重新打分。

这个方向不直接训练测试标签，不使用验证/测试标签，不把未来交互泄漏到当前行。

## 新增脚本

```text
make_consensus_online_replay_result_zip.py
```

核心输入：

```text
model_dir = outputs/combo_jittor
baseline_dir = outputs/website_submission_97_major_dual_mlp_strong_boost
teachers =
  submissions/80_major_top4q85_35/result.zip
  submissions/89_major_teacher88_online_top4q80_40/result.zip
  submissions/94_major_combo_mlp_consensus_boost/result.zip
  submissions/97_major_dual_mlp_strong_boost/result.zip
```

## 101：严格一致伪在线重放

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
export nvcc_path=
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
/mnt/e/Jitter/.venv_wsl_cpu/bin/python make_consensus_online_replay_result_zip.py \
  --data-root data/official_raw \
  --model-dir outputs/combo_jittor \
  --baseline-dir outputs/website_submission_97_major_dual_mlp_strong_boost \
  --teacher submissions/80_major_top4q85_35/result.zip \
  --teacher submissions/89_major_teacher88_online_top4q80_40/result.zip \
  --teacher submissions/94_major_combo_mlp_consensus_boost/result.zip \
  --teacher submissions/97_major_dual_mlp_strong_boost/result.zip \
  --output-dir outputs/website_submission_101_major_consensus_replay_strict \
  --mode rank \
  --include-valid-history \
  --min-top1-votes 3 \
  --min-topk-votes 4 \
  --topk 5 \
  --require-strong-teachers 3 \
  --margin-quantile 0.75 \
  --max-updates 1
```

统计：

```text
dataset2 rows = 153420
selected_rows = 38352
pseudo_edges = 38352
```

提交包：

```text
submissions/101_major_consensus_online_replay_strict/result.zip
```

格式校验：

```text
dataset1.csv: 61051 x 100, min=0.02121212, max=1.0, NaN=0
dataset2.csv: 153420 x 100, min=0.0, max=1.0, NaN=0
```

相对差异：

```text
101 vs 89:  d2_mad=0.00466634, top1_change=0.215506
101 vs 94:  d2_mad=0.00931571, top1_change=0.216008
101 vs 97:  d2_mad=0.04062622, top1_change=0.237577
101 vs 100: d2_mad=0.03110087, top1_change=0.485634
```

解释：101 是一次明显换方法的包，输出接近 89 体系，但动态伪历史来自 80/89/94/97 多教师共识。它可能提供新的线上反馈，但风险是和 97 差异较大。

## 102：动态重放专家与 97 的 rank 融合

为了降低 101 全量重放相对 97 的偏移风险，102 将 101 作为动态重放专家，与 97 做 rank-level 融合，不做全行排序移植。

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_rank_consensus.py \
  --base-dir outputs/website_submission_97_major_dual_mlp_strong_boost \
  --expert-dirs outputs/website_submission_101_major_consensus_replay_strict \
  --output-dir outputs/website_submission_102_major_replay_rank_fusion \
  --weight 0.35 \
  --sharpen 0.0 \
  --chunksize 4096
```

提交包：

```text
submissions/102_major_replay_rank_fusion/result.zip
```

格式校验：

```text
dataset1.csv: 61051 x 100, min=0.02121212, max=1.0, NaN=0
dataset2.csv: 153420 x 100, min=0.0, max=1.0, NaN=0
```

相对差异：

```text
102 vs 89:  d2_mad=0.02522428, top1_change=0.184324
102 vs 94:  d2_mad=0.02077098, top1_change=0.155377
102 vs 97:  d2_mad=0.01421918, top1_change=0.124260
102 vs 100: d2_mad=0.03132917, top1_change=0.454152
```

解释：102 不是小参数微调；它引入了新的动态重放专家，但以 97 为锚点控制风险。若提交次数紧张，建议优先提交 102；若想验证新方向上限，再提交 101。

## 本轮清理

已清理旧提交包中的 `result.zip`，只删除过期/失败提交包，不删除源码、数据、文档或输出目录。

保留提交包：

```text
80, 88, 89, 93, 94, 95, 96, 97, 100, 101, 102
```

清理记录：

```text
submissions/deleted_obsolete_submissions_20260515.txt
```

清理后 `submissions` 目录约 549M；加入 101/102 后约 651M 左右。

## 下一步判断

如果 102 线上超过 97，继续沿“动态重放专家 + 稳健融合”做：

1. 教师组合消融：80/89/97、89/94/97、80/89/94/97。
2. 按 source 活跃度分桶融合，而不是统一 weight。
3. 对 selected_rows 再加历史热度门控，避免伪边污染冷启动段。

如果 102 也明显回退，则说明“测试期伪历史重放”不可靠，应转向完全不同的方向：按候选位置/源节点分布做无监督校准或重建官方隐藏评估分布。
