# 第 81 包：Agreement-Gated Top4/Q85

## 当天总结

当前已知最强：

```text
submissions/80_major_top4q85_35/result.zip
```

线上分数：

```text
1.2061364461925292
```

今日主线从 76 到 80 全部围绕 pseudo-online combo_jittor：

- 76: `1.1637751423053575`
- 77: `1.1716880219377352`
- 78: `1.188911078447078`
- 79: `1.2004171202584817`
- 80: `1.2061364461925292`

结论：pseudo-online 写回是目前最有效方向，但 80 的边际提升已经变小，不能再盲目增加 topK。

## 新增大胆操作

新增脚本：

```text
blend_agreement_gated.py
```

原计划按置信度门控，但 rank 输出的 top1-top2 margin 基本恒定，因此改为 agreement gating：

- 比较当前最强底座 80 与 top4/q85 组件的 top 候选排序。
- 如果 top1 一致或 top5 overlap 足够高，则大权重使用 top4/q85。
- 如果 top10 overlap 中等，则中等权重。
- 如果排序分歧明显，则低权重回退到底座。

实际统计：

```text
high_rows = 153234
mid_rows  = 186
low_rows  = 0
```

说明 80 与 top4/q85 的排序高度一致，top4/q85 已经成为稳定主信号。

## 新增提交包

```text
submissions/81_major_agree_gated_top4q85/result.zip
```

融合方式：

```text
dataset1 = 100% 80
dataset2 = agreement-gated blend(80, top4/q85)
```

权重：

```text
low  = 0.12
mid  = 0.35
high = 0.58
```

相对 80 的差异：

```text
81: d1_mad 0.00000000, d2_mad 0.00341019
```

## 提交建议

今天提交次数已满，不建议继续提交。明天第一发可以考虑：

```text
submissions/81_major_agree_gated_top4q85/result.zip
```

如果 81 提升，说明 top4/q85 组件可继续提高占比；如果下降，说明 80 附近已经接近 pseudo-online 误差传播边界。
