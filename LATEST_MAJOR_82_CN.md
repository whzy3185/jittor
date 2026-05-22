# 第 82 包：三路 Pseudo-Online Rank Consensus

## 背景

当前已知最强仍为：

```text
submissions/80_major_top4q85_35/result.zip
```

线上分数：

```text
1.2061364461925292
```

81 是 agreement-gated top4/q85，尚未反馈线上结果。继续设计时不再简单推进 topK，而是尝试多路 pseudo-online 共识。

## 新增脚本

```text
blend_rank_consensus.py
```

该脚本读取多个 pseudo-online 组件：

- top2/q70
- top3/q80
- top4/q85

对每个 query 的 100 个候选做行内 rank，再取多路平均 rank，形成 consensus expert。最后将 consensus expert 注入当前最强底座。

## 试错记录

第一版使用：

```text
weight = 0.40
sharpen = 4.0
```

结果相对 80 的 dataset2 平均差异达到 `0.10023`，分布改写过大，风险过高，已删除，不推荐提交。

## 最终 82

最终保留低风险共识版：

```text
submissions/82_major_rank_consensus15/result.zip
```

参数：

```text
weight = 0.15
sharpen = 0.0
```

融合方式：

```text
dataset1 = 100% 80
dataset2 = 85% 80 + 15% rank_consensus(top2/q70, top3/q80, top4/q85)
```

相对 80 的差异：

```text
82: d1_mad 0.00000000, d2_mad 0.00076358
```

## 提交建议

如果 81 没有明显提升，82 可以作为低风险共识包提交。若需要更激进，优先等 81 线上结果后再决定是否提高 consensus 权重，而不是直接提交已删除的 sharpen 版本。
