# 第 84 包：Source-Activity Gated Top4/Q85

## 背景

当前已知最强：

```text
submissions/80_major_top4q85_35/result.zip
```

线上分数：

```text
1.2061364461925292
```

之前的 81/82/83 都是候选包，尚未收到线上反馈。本轮继续遵守“一次只生成一个大胆包”的原则。

## 新增脚本

```text
blend_source_gated.py
```

核心思路：

- pseudo-online 写回对测试期重复出现的 source 更有价值。
- 低活跃 source 更容易被错误传播污染。
- 因此按 dataset2 test 中 source 的出现次数动态调整 top4/q85 专家权重。

## 新增提交包

```text
submissions/84_major_source_gated_top4q85/result.zip
```

融合方式：

```text
dataset1 = 100% 80
dataset2 = source-gated blend(80, top4/q85)
```

权重：

```text
low_activity  = 0.10
mid_activity  = 0.35
high_activity = 0.60
```

source 活跃度门限：

```text
q_mid  = 0.50 -> src_count >= 21
q_high = 0.80 -> src_count >= 112
```

实际行数：

```text
low_rows  = 6699
mid_rows  = 35228
high_rows = 111493
```

相对 80 的差异：

```text
84: d1_mad 0.00000000, d2_mad 0.00308397
```

## 提交建议

如果 81 还没有提交，优先级：

```text
81 -> 84 -> 83 -> 82
```

84 的意义是验证“高活跃 source 更应相信 pseudo-online”的假设。如果 84 提升，后续可继续做 source-level 动态解码和 source 分桶模型。
