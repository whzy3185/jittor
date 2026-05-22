# 110-111 大方向实验记录：线上反馈加权 Rank Ensemble

日期：2026-05-15

## 线上反馈依据

用户反馈：

```text
101 1.2135024872359346
102 1.2148036842867738
103 1.2144142783655167
104 1.2143281923992328
105 提交失败
107 提交失败
```

102-104 按提交时间顺序映射：

```text
102 -> 2026-05-15 11:06:01 -> 1.2148036842867738
103 -> 2026-05-15 11:06:26 -> 1.2144142783655167
104 -> 2026-05-15 11:07:01 -> 1.2143281923992328
```

结论：

```text
102 是当前线上最优。
101 动态重放信号有效，但 103/104 的复杂门控没有超过简单 rank 融合。
105/107 提交失败，不纳入下一轮融合。
```

## 新增脚本

```text
blend_online_score_rank_ensemble.py
```

## 110：线上分数加权 rank ensemble

只使用已知有效专家：

```text
102: 当前最佳锚点
101: 动态重放专家
97: 旧强基线
94: MLP consensus boost
89: teacher online combo
```

权重：

```text
102 = 0.46
101 = 0.24
97  = 0.16
94  = 0.08
89  = 0.06
```

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_online_score_rank_ensemble.py \
  --base-dir outputs/website_submission_102_major_replay_rank_fusion \
  --expert-dirs outputs/website_submission_102_major_replay_rank_fusion outputs/website_submission_101_major_consensus_replay_strict outputs/website_submission_97_major_dual_mlp_strong_boost outputs/website_submission_94_major_combo_mlp_consensus_boost outputs/website_submission_89_major_teacher88_online_top4q80_40 \
  --weights 0.46 0.24 0.16 0.08 0.06 \
  --output-dir outputs/website_submission_110_major_online_weighted_rank_ensemble \
  --preserve-base-scale 0.0 \
  --chunksize 4096
```

提交包：

```text
submissions/110_major_online_weighted_rank_ensemble/result.zip
```

统计：

```text
top1_change_vs_102 = 0.004517
zip_size = 45M
```

相对差异：

```text
110 vs 89:  d2_mad=0.01861038, top1_change=0.180439
110 vs 94:  d2_mad=0.01434358, top1_change=0.151695
110 vs 97:  d2_mad=0.02165148, top1_change=0.127643
110 vs 101: d2_mad=0.01936629, top1_change=0.126874
110 vs 102: d2_mad=0.00783259, top1_change=0.004517
```

## 111：保留 102 分数尺度的线上加权 ensemble

权重：

```text
102 = 0.52
101 = 0.22
97  = 0.14
94  = 0.07
89  = 0.05
```

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_online_score_rank_ensemble.py \
  --base-dir outputs/website_submission_102_major_replay_rank_fusion \
  --expert-dirs outputs/website_submission_102_major_replay_rank_fusion outputs/website_submission_101_major_consensus_replay_strict outputs/website_submission_97_major_dual_mlp_strong_boost outputs/website_submission_94_major_combo_mlp_consensus_boost outputs/website_submission_89_major_teacher88_online_top4q80_40 \
  --weights 0.52 0.22 0.14 0.07 0.05 \
  --output-dir outputs/website_submission_111_major_online_weighted_rank_ensemble_scaled \
  --preserve-base-scale 0.55 \
  --chunksize 4096
```

提交包：

```text
submissions/111_major_online_weighted_rank_ensemble_scaled/result.zip
```

统计：

```text
top1_change_vs_102 = 0.002236
zip_size = 67M
```

相对差异：

```text
111 vs 89:  d2_mad=0.01975531, top1_change=0.182434
111 vs 94:  d2_mad=0.01541480, top1_change=0.153644
111 vs 97:  d2_mad=0.02004406, top1_change=0.126001
111 vs 101: d2_mad=0.02074130, top1_change=0.128588
111 vs 102: d2_mad=0.00607558, top1_change=0.002236
```

## 格式校验

110、111 均已检查：

```text
result.zip 内含 dataset1.csv 和 dataset2.csv
dataset1.csv: 61051 x 100
dataset2.csv: 153420 x 100
概率范围在 [0, 1]
NaN = 0
```

## 提交建议

优先：

```text
submissions/110_major_online_weighted_rank_ensemble/result.zip
```

原因：

```text
110 文件更小，变化略大，是更适合先验证的线上反馈 ensemble。
111 更保守但文件 67M，接近 107 的大包体积；在 107 提交失败后，不建议优先提交 111。
```

如果 110 提升，继续围绕线上分数加权 ensemble 做 102 锚点下的权重搜索。
如果 110 回退，则说明 102 已接近当前专家族最优，下一阶段应换全新专家，而不是继续线性/rank ensemble。
