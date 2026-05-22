# 100 号三方共识 MLP 大改动

## 线上反馈背景

用户截图反馈：95-99 中，98/99 出现明显回退。

结论：

```text
98/99 的 rank transplant 让双 MLP 过度接管排序，线上明显失败。
MLP 信号仍然有效，但必须受 base=94 的强约束，不能全行或大范围 topK 移植。
```

## 100 的方法

新增脚本：

```text
blend_dual_mlp_intersection.py
```

100 使用三方共识：

```text
base = 94
expert A = MLP1 rank
expert B = MLP2 rank
```

候选必须满足：

```text
1. 在 MLP1 topN 中；
2. 在 MLP2 topN 中；
3. 与 base=94 的 top30 至少有 2 个重合候选；
```

才允许抢 94 的高分位。这样避免 98/99 那种 MLP 全面接管。

## 生成命令

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_dual_mlp_intersection.py \
  --base-dir outputs/website_submission_94_major_combo_mlp_consensus_boost \
  --expert-a-dir outputs/website_submission_combo_mlp_split0_internal_rank \
  --expert-b-dir outputs/website_submission_combo_mlp_s3407_h192_rank \
  --output-dir outputs/website_submission_100_major_triple_consensus_mlp \
  --warm-seen 4 \
  --hot-seen 24 \
  --warm-topn 30 \
  --hot-topn 50 \
  --warm-take 6 \
  --hot-take 14 \
  --base-keep 30 \
  --require-base-overlap 2
mkdir -p submissions/100_major_triple_consensus_mlp
cp outputs/website_submission_100_major_triple_consensus_mlp/result.zip submissions/100_major_triple_consensus_mlp/result.zip
```

## 提交包

```text
submissions/100_major_triple_consensus_mlp/result.zip
```

校验：

```text
valid = True
dataset1.csv = 61051 x 100
dataset2.csv = 153420 x 100
```

统计：

```text
changed_rows = 141943
eligible_rows = 143161
avg_shared = 39.482184
d1_mad_vs94 = 0.00000000
d2_mad_vs94 = 0.02504816
d2_std = 0.29105039
```

## 建议

```text
100 是大改动，但比 98/99 更受控。
如果 97 的结果好，可以提交 100 继续验证三方共识；
如果 97 也回撤，则 100 风险仍然偏高，应回到 94 或训练更贴近 test 的新 ranker。
```
