# 第 86 包：Source-Time 双门控 Top4/Q85

## 背景

当前已知最强：

```text
submissions/80_major_top4q85_35/result.zip
```

线上分数：

```text
1.2061364461925292
```

81-85 是候选包，尚未收到线上反馈。本轮继续单包结构改动。

## 新增脚本

```text
blend_source_time_gated.py
```

核心假设：

- 高活跃 source 更适合相信 pseudo-online。
- 测试后段拥有更多伪历史积累，也更适合相信 pseudo-online。
- 同时满足高活跃和测试后段时，给 top4/q85 最高权重。
- 低活跃且测试早期时，保守回退 80。

## 新增提交包

```text
submissions/86_major_source_time_gated_top4q85/result.zip
```

融合方式：

```text
dataset1 = 100% 80
dataset2 = source-time-gated blend(80, top4/q85)
```

权重：

```text
low         = 0.12
active      = 0.38
late        = 0.42
active_late = 0.68
```

门限：

```text
src_count >= 112
time >= 1317945600
```

实际行数：

```text
low_rows         = 31410
active_rows      = 83526
late_rows        = 10517
active_late_rows = 27967
```

相对 80 的差异：

```text
86: d1_mad 0.00000000, d2_mad 0.00237746
```

## 提交建议

当前候选优先级：

```text
81 -> 84 -> 86 -> 85 -> 83 -> 82
```

86 用于验证 source 活跃度和时间进度是否可以共同决定 pseudo-online 权重。如果 84 或 85 单独有效，86 可能进一步受益。
