# 第 83 包：Top5/Q90 高置信宽覆盖

## 背景

当前已知最强：

```text
submissions/80_major_top4q85_35/result.zip
```

线上分数：

```text
1.2061364461925292
```

81 和 82 是候选包，尚未收到线上反馈。本轮继续遵守“一次只生成一个大胆包”的原则。

## 新增结构组件

```text
outputs/website_submission_online_combo_jittor_top5q90/result.zip
```

参数：

```text
update_topk = 5
margin_quantile = 0.90
mode = rank
include_valid_history = true
```

含义：

- 每个 query 最多写回 top5 预测边。
- 只保留同一时间片内 margin 位于前 10% 的高置信预测。
- 实际写回伪边约 `77515` 条。
- 相比 top4/q85，写回总数更少，但每个高置信 query 的候选覆盖更宽。

## 新增提交包

```text
submissions/83_major_top5q90_30/result.zip
```

融合方式：

```text
dataset1 = 100% 80
dataset2 = 70% 80 + 30% online_combo_top5q90
```

相对 80 的差异：

```text
83: d1_mad 0.00000000, d2_mad 0.00208577
```

## 提交建议

如果还没有提交 81，建议优先级：

```text
81 -> 83 -> 82
```

81 更激进，83 是高置信宽覆盖，82 是低风险三路 consensus。若 81 已提交且下降，可考虑 83；若提交次数很紧，83 比 82 更有冲击力。
