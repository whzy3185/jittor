# 97 号大改动说明

## 背景

用户要求在到达 1.3 前只做大幅改动，不再围绕 94 做小幅 alpha 调整。因此：

```text
96_major_combo_mlp_consensus_plus/result.zip
d2_mad_vs94 = 0.00290084
```

标记为低优先级，不建议占用提交次数。

## 第二个差异化 MLP

新增训练一个和 93/94 不同的 Jittor MLP：

```text
outputs/combo_mlp_jittor_split0_internal_s3407_h192_fast/
seed = 3407
hidden_dim = 192
dropout = 0.16
internal_holdout_edges = 30000
negatives = 59
epochs = 5
```

训练约束：

```text
只使用 dataset2 split=0 内部 holdout
不使用官方 split=1 validation labels 作为训练目标
Jittor MLP pairwise ranking loss
```

训练结果：

```text
loss: 0.5367 -> 0.4687
```

第二 MLP 专家：

```text
outputs/website_submission_combo_mlp_s3407_h192_rank/result.zip
d2_mad_vs89 = 0.34454801
```

## 双 MLP 共识专家

将 93/94 使用的第一个 MLP 专家与第二 MLP 专家做 50/50 rank 平均：

```text
outputs/website_submission_combo_mlp_dualavg_rank/result.zip
MLP1 50% + MLP2 50%
d2_mad_vs89 = 0.34451265
```

这不是单一模型的小参数扰动，而是两个不同 seed / hidden / holdout / negative 设置下的 Jittor MLP 专家共识。

## 97 提交包

```text
submissions/97_major_dual_mlp_strong_boost/result.zip
```

生成方式：

```text
base = 94
expert = dual MLP average rank
保留 base 每行分数集合，只用 dual MLP 改变候选排序
```

参数：

```text
warm_seen = 2
hot_seen = 16
warm_alpha = 0.14
hot_alpha = 0.30
low_agree_scale = 0.25
topn = 10
min_top_overlap = 1
```

统计：

```text
changed_rows = 146678
agree_rows = 146270
warm_rows = 19629
hot_rows = 127049
d1_mad_vs94 = 0.00000000
d2_mad_vs94 = 0.03329532
d2_std = 0.29105039
```

格式校验：

```text
valid = True
dataset1.csv = 61051 x 100
dataset2.csv = 153420 x 100
```

## 提交建议

```text
97 是大幅多 MLP 共识重排包，风险高，但符合当前冲 1.3 前的大改动策略。
如果要稳健保分，当前线上最好仍是 94。
如果要探索突破，则提交 97。
```
