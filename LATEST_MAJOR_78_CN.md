# 第 78 包：更激进 Online Combo Top2/Q70

## 当前线上反馈

- `76_major_onlinecombo10/result.zip`: `1.1637751423053575`
- `77_major_onlinecombo20/result.zip`: `1.1716880219377352`

当前已知最强：

```text
submissions/77_major_onlinecombo20/result.zip
```

说明 pseudo-online combo_jittor 方向有效，而且 20% online-combo 融合明显强于 10%。

用户提醒每天只能提交 20 次，当前已提交 17 次。因此后续每次只生成一个大胆结构包，不再整理多个小扰动包。

## 新增结构组件

```text
outputs/website_submission_online_combo_jittor_top2q70/result.zip
```

相对 76/77 使用的 top1/q50 组件，本组件改为：

```text
update_topk = 2
margin_quantile = 0.70
mode = rank
include_valid_history = true
```

含义：

- 每个 query 最多写回 top2 预测边。
- 只保留同一时间片内 margin 位于前 30% 的高置信预测。
- 实际写回伪边约 `92220` 条。
- 作为对比，top1/q50 组件写回约 `76796` 条。

这是一次结构性改动，不是 0.01 级别权重扰动。

## 新增提交包

只生成一个提交包：

```text
submissions/78_major_top2q70_25/result.zip
```

融合方式：

```text
dataset1 = 100% 77
dataset2 = 75% 77 + 25% online_combo_top2q70
```

相对 77 的差异：

```text
78: d1_mad 0.00000000, d2_mad 0.00378438
```

## 提交建议

下一次优先提交：

```text
submissions/78_major_top2q70_25/result.zip
```

如果 78 提升，说明更激进的 pseudo-online 写回仍有收益；如果 78 下降，说明 top2 带来的错误传播开始超过收益，后续应转向 group-level/test-candidate 特征或重新训练 ranker。
