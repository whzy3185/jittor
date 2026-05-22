# 第 80 包：Top4/Q85 Online Combo

## 线上反馈

`79_major_top3q80_30/result.zip` 线上分数：

```text
1.2004171202584817
```

当前已知最强为 79。相比 78 的 `1.188911078447078`，说明高置信多边 pseudo-online 写回仍然有效。

如果今天只剩最后一次提交，本轮只生成一个大胆结构包。

## 新增结构组件

```text
outputs/website_submission_online_combo_jittor_top4q85/result.zip
```

参数：

```text
update_topk = 4
margin_quantile = 0.85
mode = rank
include_valid_history = true
```

含义：

- 每个 query 最多写回 top4 预测边。
- 只保留同一时间片内 margin 位于前 15% 的高置信预测。
- 实际写回伪边约 `92576` 条。
- 79 的 top3/q80 组件写回约 `92466` 条；80 的写回数量接近，但候选覆盖更宽、置信门槛更高。

## 新增提交包

```text
submissions/80_major_top4q85_35/result.zip
```

融合方式：

```text
dataset1 = 100% 79
dataset2 = 65% 79 + 35% online_combo_top4q85
```

相对 79 的差异：

```text
80: d1_mad 0.00000000, d2_mad 0.00316827
```

## 提交建议

如果今天只剩一次提交，提交：

```text
submissions/80_major_top4q85_35/result.zip
```

如果 80 继续提升，说明高置信多边 pseudo-online 仍未到上限；如果 80 下降，说明 top4 已开始引入过多错误传播。
