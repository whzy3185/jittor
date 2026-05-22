# 94/95 结果与后续方向

## 线上结果

```text
93 1.2113817385926295
94 1.2114108457479125
```

对比：

```text
89 1.2107659416475025
```

结论：训练式 Jittor MLP ranker 方向有效；其中 94 的共识门控版本略优于 93。

## 94 为什么更稳

94 保留 89 的行内分数分布，只在 MLP 与 89 的 top10 有至少 3 个交集时做重排：

```text
warm_seen = 8
hot_seen = 40
warm_alpha = 0.04
hot_alpha = 0.10
low_agree_scale = 0.00
topn = 10
min_top_overlap = 3
d2_mad_vs89 = 0.00560484
```

这比 93 的重排更保守：

```text
93 d2_mad_vs89 = 0.01600329
94 d2_mad_vs89 = 0.00560484
```

## 95 风险

95 已生成，但属于高风险增强：

```text
submissions/95_major_combo_mlp_stronger_boost/result.zip
warm_seen = 2
hot_seen = 16
warm_alpha = 0.12
hot_alpha = 0.26
d2_mad_vs89 = 0.03037050
```

由于 91/92 的大幅重排已经明显回撤，95 的差异过大，不建议优先提交。后续应围绕 94 的小差异共识门控继续做受控增强。

## 推荐

```text
当前最好线上包：94_major_combo_mlp_consensus_boost/result.zip
下一步方向：94 附近的小幅增强、多 seed MLP 或更强共识 gating。
```

## 96 备注

96 已生成：

```text
submissions/96_major_combo_mlp_consensus_plus/result.zip
d2_mad_vs94 = 0.00290084
```

但用户明确要求在到达 1.3 前只做大幅度改动，因此 96 标记为低优先级，不建议占用提交次数。下一步改为训练差异化 MLP 专家和多专家共识。
