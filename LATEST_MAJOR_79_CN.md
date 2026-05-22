# 第 79 包：Top3/Q80 Online Combo

## 线上反馈

`78_major_top2q70_25/result.zip` 线上分数：

```text
1.188911078447078
```

当前已知最强为 78。相比 77 的 `1.1716880219377352`，说明更激进的 pseudo-online 写回明显有效。

由于每天只能提交 20 次，当前已提交 18 次，本轮只生成一个大胆结构包。

## 新增结构组件

```text
outputs/website_submission_online_combo_jittor_top3q80/result.zip
```

参数：

```text
update_topk = 3
margin_quantile = 0.80
mode = rank
include_valid_history = true
```

含义：

- 每个 query 最多写回 top3 预测边。
- 只保留同一时间片内 margin 位于前 20% 的高置信预测。
- 实际写回伪边约 `92466` 条。
- 78 的 top2/q70 组件写回约 `92220` 条；79 的写回数量接近，但候选覆盖方式更激进。

## 新增提交包

```text
submissions/79_major_top3q80_30/result.zip
```

融合方式：

```text
dataset1 = 100% 78
dataset2 = 70% 78 + 30% online_combo_top3q80
```

相对 78 的差异：

```text
79: d1_mad 0.00000000, d2_mad 0.00356242
```

## 提交建议

下一次只提交：

```text
submissions/79_major_top3q80_30/result.zip
```

如果 79 提升，说明高置信 top3 写回仍有收益；如果下降，说明当前 pseudo-online 路线可能到达错误传播边界，最后一次提交应转向完全不同的 group-level/test-candidate 特征路线。
