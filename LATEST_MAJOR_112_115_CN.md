# 112-115 大方向实验记录：110 锚点下的 Replay / 多样性 / 概率校准

日期：2026-05-15

## 线上反馈依据

```text
110 -> 1.2151966152453952
111 -> 1.2151332588531099
```

结论：

```text
110 是当前已知最优。
111 的 scaled 分支略低，说明后续不再优先保留 102 原分数尺度。
```

本轮以 110 为新锚点。

## 112：更偏 101 replay 的 rank ensemble

目的：验证 101 动态重放信号是否还能继续加权。

权重：

```text
110 = 0.42
101 = 0.28
102 = 0.15
97  = 0.08
94  = 0.04
89  = 0.03
```

提交包：

```text
submissions/112_major_more_replay_rank_ensemble/result.zip
```

统计：

```text
top1_change_vs_110 = 0.001819
zip_size = 46M
```

相对差异：

```text
112 vs 102: d2_mad=0.01151150, top1_change=0.005795
112 vs 110: d2_mad=0.00389673, top1_change=0.001819
112 vs 101: d2_mad=0.01571821, top1_change=0.125864
112 vs 103: d2_mad=0.00908101, top1_change=0.109301
```

## 113：纳入 103/104 少量多样性的 rank ensemble

目的：103/104 单包低于 110，但可能提供少量多样性；本包只纳入小权重，不让它们接管。

权重：

```text
110 = 0.50
102 = 0.18
101 = 0.14
103 = 0.08
104 = 0.06
97  = 0.04
```

提交包：

```text
submissions/113_major_diverse_rank_ensemble/result.zip
```

统计：

```text
top1_change_vs_110 = 0.000717
zip_size = 45M
```

相对差异：

```text
113 vs 102: d2_mad=0.00907673, top1_change=0.004641
113 vs 110: d2_mad=0.00289927, top1_change=0.000717
113 vs 101: d2_mad=0.01876386, top1_change=0.126776
113 vs 103: d2_mad=0.00960971, top1_change=0.108728
```

## 114：110 概率分布 flatten 校准

目的：不改排序，只改变概率分布，判断线上指标是否吃概率校准。

参数：

```text
gamma = 0.92
mix_original = 0.20
```

提交包：

```text
submissions/114_major_110_power_flatten/result.zip
```

统计：

```text
top1_change_vs_110 = 0.000000
zip_size = 66M
```

相对差异：

```text
114 vs 102: d2_mad=0.01799378, top1_change=0.004517
114 vs 110: d2_mad=0.01653654, top1_change=0.000000
114 vs 101: d2_mad=0.02872146, top1_change=0.126874
114 vs 103: d2_mad=0.02047525, top1_change=0.108799
```

## 115：110 概率分布 sharpen 校准

目的：与 114 对照，不改排序，只让概率更尖锐。

参数：

```text
gamma = 1.08
mix_original = 0.20
```

提交包：

```text
submissions/115_major_110_power_sharpen/result.zip
```

统计：

```text
top1_change_vs_110 = 0.000000
zip_size = 66M
```

相对差异：

```text
115 vs 102: d2_mad=0.01703582, top1_change=0.004517
115 vs 110: d2_mad=0.01527197, top1_change=0.000000
115 vs 101: d2_mad=0.02773894, top1_change=0.126874
115 vs 103: d2_mad=0.01977789, top1_change=0.108799
```

## 格式校验

112、113、114、115 均已检查：

```text
result.zip 内含 dataset1.csv 和 dataset2.csv
dataset1.csv: 61051 x 100
dataset2.csv: 153420 x 100
概率范围在 [0, 1]
NaN = 0
```

## 提交建议

优先顺序：

```text
1. 112_major_more_replay_rank_ensemble
2. 113_major_diverse_rank_ensemble
3. 115_major_110_power_sharpen
4. 114_major_110_power_flatten
```

理由：

```text
112 是最清晰的 replay 加权延伸，文件小，变化明确。
113 是多样性检验，文件小，风险较低。
114/115 排序完全不变，只测概率校准；如果平台主要看排序，则不会变化。
115 比 114 更接近常见排名指标偏好，排在 114 前。
```
