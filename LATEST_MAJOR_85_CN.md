# 第 85 包：Time-Gated Top4/Q85

## 背景

当前已知最强：

```text
submissions/80_major_top4q85_35/result.zip
```

线上分数：

```text
1.2061364461925292
```

本轮继续单包结构改动，不做多包小扰动。

## 新增脚本

```text
blend_time_gated.py
```

核心假设：

- pseudo-online 的优势会随着测试时间推进而增强。
- 测试早期可用的伪历史少，错误传播更敏感。
- 测试后期已有更多高置信伪边积累，可以更大权重使用 top4/q85。

## 新增提交包

```text
submissions/85_major_time_gated_top4q85/result.zip
```

融合方式：

```text
dataset1 = 100% 80
dataset2 = time-gated blend(80, top4/q85)
```

权重：

```text
early = 0.15
mid   = 0.35
late  = 0.62
```

时间段统计：

```text
early_rows = 45457
mid_rows   = 69479
late_rows  = 38484
```

时间门限：

```text
early threshold = 1304726400
late threshold  = 1317945600
```

相对 80 的差异：

```text
85: d1_mad 0.00000000, d2_mad 0.00228293
```

## 提交建议

如果 81 还没有提交，优先级仍建议：

```text
81 -> 84 -> 85 -> 83 -> 82
```

85 用于验证“测试后段更应相信 pseudo-online”的假设。如果 85 提升，后续可以做同时结合 source 活跃度和时间进度的双门控。
