# 2026-05-12 提交与实验总结

## 当天核心结论

今天最有效的大方向是 dataset2 的 pseudo-online combo_jittor 动态写回。当前已知最强为：

```text
submissions/80_major_top4q85_35/result.zip
```

线上分数：

```text
1.2061364461925292
```

相比早期强包：

```text
43: 1.1305940693487557
56: 1.1462969419419828
69: 1.1550793918085815
77: 1.1716880219377352
78: 1.188911078447078
79: 1.2004171202584817
80: 1.2061364461925292
```

整体提升路径非常清楚：静态 full-history 融合已经接近上限，真正拉开分数的是把 combo_jittor 变成按时间顺序 pseudo-online 推理，并把高置信预测写回历史。

## 今日有效改动

### 1. Online Combo Jittor

新增脚本：

```text
make_online_combo_jittor_result_zip.py
```

它将 combo_jittor 从静态打分改为动态打分：

- dataset2 测试集按 `time, row_id` 顺序处理。
- 同一时间片内先完成打分，不把同时间片预测互相写回。
- 时间片结束后，根据 margin 置信度选择预测边写回历史。
- 写回后同步更新 sequential、temporal、assoc、pair/pop/recency 统计。
- 后续时间片使用更新后的历史重新构造 combo_jittor 特征。

### 2. 提交包结果

| 包号 | 改动 | 线上分数 | 结论 |
|---|---|---:|---|
| 76 | 69 + 10% top1/q50 online-combo | 1.1637751423053575 | online-combo 有效 |
| 77 | 69 + 20% top1/q50 online-combo | 1.1716880219377352 | 20% 明显强于 10% |
| 78 | 77 + 25% top2/q70 online-combo | 1.188911078447078 | 更激进写回显著有效 |
| 79 | 78 + 30% top3/q80 online-combo | 1.2004171202584817 | top3 高置信写回继续有效 |
| 80 | 79 + 35% top4/q85 online-combo | 1.2061364461925292 | 仍提升，但边际收益变小 |

## 当前判断

- dataset2 是主要提分来源。
- pseudo-online 写回方向确认有效。
- top1/q50 到 top3/q80 的提升明显。
- top4/q85 仍然提升，但边际收益已明显下降。
- 继续简单增加 topK 或融合比例，风险会升高。

## 明天优先方向

今天提交次数已用满，明天不应继续盲目 topK 推进。下一步应该做更有选择性的大胆改动：

```text
confidence-gated online-combo
```

思路：

- 对每个 query 计算 online-combo 组件的行内置信度。
- 高置信 query：更大权重使用 pseudo-online 组件。
- 中置信 query：维持当前 80 附近权重。
- 低置信 query：回退到 80，避免错误传播污染。

这比全局固定 `65% 79 + 35% top4/q85` 更精细，也符合今天观察到的“更激进有收益，但边际开始下降”。
