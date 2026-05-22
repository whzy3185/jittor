# 116-118 大方向实验记录：110 锚点行级专家一致性门控

日期：2026-05-15

## 背景

当前已知最优：

```text
110 1.2151966152453952
```

本轮不再简单整体加权，而是以 110 为 base，根据 101、112、113、102、97 的 top1 一致性和行内 margin 做行级组合。

## 新增脚本

```text
blend_row_consensus_gate.py
```

使用专家：

```text
base        = 110_major_online_weighted_rank_ensemble
replay      = 101_major_consensus_replay_strict
more_replay = 112_major_more_replay_rank_ensemble
diverse     = 113_major_diverse_rank_ensemble
stable      = 102_major_replay_rank_fusion
old         = 97_major_dual_mlp_strong_boost
```

## 116：保护型一致性门控

提交包：

```text
submissions/116_major_row_consensus_protect/result.zip
```

统计：

```text
top1_change_vs_110 = 0.000750
bucket_base_majority = 153371
bucket_diverse_majority = 42
bucket_replay_majority = 3
bucket_protect_fallback = 4
zip_size = 57M
```

相对差异：

```text
116 vs 110: d2_mad=0.00107491, top1_change=0.000750
116 vs 112: d2_mad=0.00328014, top1_change=0.001134
116 vs 113: d2_mad=0.00205193, top1_change=0.000124
116 vs 102: d2_mad=0.00850953, top1_change=0.004700
116 vs 101: d2_mad=0.01890154, top1_change=0.126718
```

## 117：Replay 推进型一致性门控

提交包：

```text
submissions/117_major_row_consensus_replay/result.zip
```

统计：

```text
top1_change_vs_110 = 0.001662
bucket_strong_replay = 116308
bucket_more_replay_vote = 37061
bucket_base_majority = 46
bucket_replay_fallback = 5
zip_size = 57M
```

相对差异：

```text
117 vs 110: d2_mad=0.00655839, top1_change=0.001662
117 vs 112: d2_mad=0.00358510, top1_change=0.000196
117 vs 113: d2_mad=0.00551140, top1_change=0.001049
117 vs 102: d2_mad=0.01388391, top1_change=0.005645
117 vs 101: d2_mad=0.01352705, top1_change=0.125851
```

## 118：多样性型一致性门控

提交包：

```text
submissions/118_major_row_consensus_diverse/result.zip
```

统计：

```text
top1_change_vs_110 = 0.000306
bucket_diverse_vote = 153355
bucket_stable_base = 61
bucket_replay_diverse = 2
bucket_diverse_fallback = 2
zip_size = 56M
```

相对差异：

```text
118 vs 110: d2_mad=0.00301669, top1_change=0.000306
118 vs 112: d2_mad=0.00308324, top1_change=0.001564
118 vs 113: d2_mad=0.00125497, top1_change=0.000424
118 vs 102: d2_mad=0.00953976, top1_change=0.004269
118 vs 101: d2_mad=0.01851611, top1_change=0.127076
```

## 格式校验

116、117、118 均已检查：

```text
result.zip 内含 dataset1.csv 和 dataset2.csv
dataset1.csv: 61051 x 100
dataset2.csv: 153420 x 100
概率范围在 [0, 1]
NaN = 0
```

## 提交建议

如果 112/113 尚未提交，优先仍是：

```text
112_major_more_replay_rank_ensemble
113_major_diverse_rank_ensemble
```

本轮 116-118 的推荐顺序：

```text
1. 117_major_row_consensus_replay
2. 118_major_row_consensus_diverse
3. 116_major_row_consensus_protect
```

理由：

```text
117 最有信息量，是 replay 推进方向。
118 接近 113，用于验证 diverse 分支。
116 太接近 110，若提交次数紧张可以跳过。
```
