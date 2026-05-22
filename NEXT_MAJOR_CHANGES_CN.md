# 后续大规模改动方向

当前已知最强提交：

```text
submissions/69_d1_62_d2_online07_combo36_temp06/result.zip
```

线上分数：

```text
1.1550793918085815
```

第一名约 `1.37`，当前方案和第一名仍有明显差距。后续不再继续整理小幅权重扰动提交包，只在完成大规模结构性改动后再生成新的 `result.zip`。

## 已知有效方向

- dataset2 是主要增益来源。
- `combo_jittor` 从 0.22 提到 0.36 持续有效。
- `online q50` 从 0.05 提到 0.07 有效。
- `temporal=0.06` 优于 `0.04`。
- 单独调整 dataset1 online 的收益较小。
- 几何 rank 融合有一定收益，但弱于当前线性融合主线。

## 后续只做大规模改动

后续新提交包应至少包含以下一种结构性变化：

1. 重新训练更强的 Jittor ranker，而不是只调融合权重。
2. 加入新的 leakage-safe 历史特征，例如多跳转移、按 source 类型的序列模式、时间分段转移矩阵。
3. 为 dataset2 单独训练模型和特征权重，不再复用 dataset1 的策略。
4. 做 pseudo-online sequential decoding 的更强版本，例如按候选组动态更新 top-k 伪历史，而不是固定 q50。
5. 引入更强的 group-level 特征，例如同一 source 的候选共现、候选在不同 query 中的重复模式、目标节点在测试候选池内的局部热度。
6. 做模型级 ensemble，例如多 seed Jittor MLP / combo ranker，而不是只融合已有 CSV。

## 暂停的小改动

以下类型改动暂时不再单独出包：

- combo 权重从 0.36 调到 0.37/0.38。
- online 权重从 0.07 调到 0.06/0.08。
- temporal 权重做 0.01 级别微调。
- 62/69 与几何 rank 包做 3%-5% 小比例融合。
- 只调整 dataset1 online 的小步扰动。

## 已执行的大改：Online Combo Jittor

新增脚本：

```text
make_online_combo_jittor_result_zip.py
```

该脚本让 `combo_jittor` 在 dataset2 测试集上按时间顺序 pseudo-online 推理，并把高置信 top1 预测写回 sequential/temporal/assoc 历史，用后续更新后的历史影响后续 query。

生成组件：

```text
outputs/website_submission_online_combo_jittor_q50/result.zip
```

提交包：

```text
submissions/76_major_onlinecombo10/result.zip
submissions/77_major_onlinecombo20/result.zip
```

推荐先交 `76_major_onlinecombo10`。如果 76 提升，再交 77；如果 76 下降，说明该 pseudo-online combo 方向存在误差传播，需要转向重新训练 ranker 或构建更强 group-level/test-candidate 特征。

## 线上反馈与第 78 包

线上反馈：

- `76_major_onlinecombo10/result.zip`: `1.1637751423053575`
- `77_major_onlinecombo20/result.zip`: `1.1716880219377352`

当前已知最强为 77。用户提醒每天只能提交 20 次，当前已提交 17 次，因此后续每次只生成一个大胆结构包。

第 78 包采用更激进的 pseudo-online 写回：

- `update_topk=2`
- `margin_quantile=0.70`
- 实际写回伪边约 `92220` 条

提交包：

```text
submissions/78_major_top2q70_25/result.zip
```

融合方式：

```text
dataset1 = 100% 77
dataset2 = 75% 77 + 25% online_combo_top2q70
```

下一次优先提交 78。

## 线上反馈与第 79 包

线上反馈：

- `78_major_top2q70_25/result.zip`: `1.188911078447078`

当前已知最强为 78。由于每天只能提交 20 次且当前已提交 18 次，本轮只生成一个大胆结构包。

第 79 包采用 top3/q80 pseudo-online 写回：

- `update_topk=3`
- `margin_quantile=0.80`
- 实际写回伪边约 `92466` 条

提交包：

```text
submissions/79_major_top3q80_30/result.zip
```

融合方式：

```text
dataset1 = 100% 78
dataset2 = 70% 78 + 30% online_combo_top3q80
```

下一次只建议提交 79。

## 线上反馈与第 80 包

线上反馈：

- `79_major_top3q80_30/result.zip`: `1.2004171202584817`

当前已知最强为 79。如果今天只剩最后一次提交，本轮只生成一个大胆结构包。

第 80 包采用 top4/q85 pseudo-online 写回：

- `update_topk=4`
- `margin_quantile=0.85`
- 实际写回伪边约 `92576` 条

提交包：

```text
submissions/80_major_top4q85_35/result.zip
```

融合方式：

```text
dataset1 = 100% 79
dataset2 = 65% 79 + 35% online_combo_top4q85
```

如果今天只剩一次提交，建议提交 80。
