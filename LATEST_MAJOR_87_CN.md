# 第 87 包：Source-Time 双门控 Top5/Q90

## 背景

当前已知最强：

```text
submissions/80_major_top4q85_35/result.zip
```

线上分数：

```text
1.2061364461925292
```

81-86 是候选包，尚未收到线上反馈。本轮继续单包结构改动。

## 设计

87 沿用 86 的 source-time 双门控，但把专家组件从 top4/q85 换成 top5/q90：

```text
expert = outputs/website_submission_online_combo_jittor_top5q90
```

核心假设：

- top5/q90 写回总边数更少，但每个高置信 query 的候选覆盖更宽。
- 高活跃 source 且测试后段时，更适合使用这种宽覆盖专家。
- 低活跃且测试早期时，继续保守回退 80。

## 新增提交包

```text
submissions/87_major_source_time_gated_top5q90/result.zip
```

融合方式：

```text
dataset1 = 100% 80
dataset2 = source-time-gated blend(80, top5/q90)
```

权重：

```text
low         = 0.10
active      = 0.34
late        = 0.40
active_late = 0.64
```

门限与行数：

```text
src_count >= 112
time >= 1317945600

low_rows         = 31410
active_rows      = 83526
late_rows        = 10517
active_late_rows = 27967
```

相对 80 的差异：

```text
87: d1_mad 0.00000000, d2_mad 0.00258885
```

## 提交建议

当前候选优先级：

```text
81 -> 84 -> 86 -> 87 -> 85 -> 83 -> 82
```

87 用于验证 top5/q90 是否能在 source-time 双门控下超过 top4/q85。
