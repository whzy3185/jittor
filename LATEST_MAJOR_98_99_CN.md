# 98/99 大幅排序移植说明

## 背景

用户要求在到达 1.3 前继续做大幅改动。97 已经是双 MLP 共识重排，但仍是加权扰动。98/99 改为 rank transplant：直接把双 MLP 共识排序移植到当前最好 94 的分数集合上。

## 新增脚本

```text
blend_rank_transplant.py
```

核心逻辑：

```text
1. base 使用 94；
2. expert 使用双 MLP 平均 rank；
3. 保留 base 每行 100 个分数集合；
4. 用 expert 排序决定候选拿到哪个分数；
5. 因此概率分布不塌缩，但候选排序可以大幅改变。
```

## 98

提交包：

```text
submissions/98_major_dual_mlp_rank_transplant/result.zip
```

策略：

```text
warm source: top20 移植
hot source: 全行排序移植
warm_seen = 4
hot_seen = 24
min_overlap = 1
```

统计：

```text
changed_rows = 142921
warm_rows = 24798
hot_rows = 118363
d1_mad_vs94 = 0.00000000
d2_mad_vs94 = 0.27614148
d2_std = 0.29105039
```

98 是极高风险大改动包。它可以测试双 MLP 是否足以接管大部分 hot source 排序。

## 99

提交包：

```text
submissions/99_major_dual_mlp_top_transplant/result.zip
```

策略：

```text
warm source: top30 移植
hot source: top30 移植
warm_seen = 4
hot_seen = 24
min_overlap = 1
```

统计：

```text
changed_rows = 143062
warm_rows = 24798
hot_rows = 118363
d1_mad_vs94 = 0.00000000
d2_mad_vs94 = 0.12737576
d2_std = 0.29105039
```

99 仍是大改动，但比 98 少了全行移植，风险更低。

## 校验

```text
98 valid = True
99 valid = True
dataset1.csv = 61051 x 100
dataset2.csv = 153420 x 100
```

## 建议

```text
稳健最好：94
大改动优先：99
极限大改动：98
已生成但不建议优先：96
中等大改动：97
```

如果 99 提升，后续继续围绕 dual MLP top-k transplant 做分组；如果 99/98 都明显回撤，说明 MLP 学到的排序只能小幅辅助，不能接管候选排序。

## 线上反馈补充

用户截图反馈：从下到上为 95-99，其中 98/99 出现明显回退。



结论：大幅 rank transplant 过度接管排序会严重回撤。MLP 信号有效，但不能用全行移植或大范围 topK 移植接管候选排序。后续应放弃 98/99 这种 transplant 路线，改为基于当前最好 MLP boost 包做受控多专家增强。
