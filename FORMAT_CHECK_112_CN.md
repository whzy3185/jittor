# 112 提交前格式检查

日期：2026-05-15

## 背景

121 重打包版已成功提交并得到：

```text
121 1.2156836736721401
```

用户反馈最近提交成功率较低，因此对 112 做与 121 相同的完整格式检查，并生成重打包版。

## 检查对象

原始包：

```text
submissions/112_major_more_replay_rank_ensemble/result.zip
```

重打包版：

```text
submissions/112_major_more_replay_rank_ensemble_repack_checked/result.zip
```

## 检查结果

本地未发现 112 格式问题：

```text
zip CRC 检查：通过
zip 根目录文件：dataset1.csv, dataset2.csv
dataset1.csv 行数：61051
dataset2.csv 行数：153420
每行列数：100
概率范围：[0, 1]
NaN/Inf：无
空值：无
小数位：8 位
```

## 文件大小

```text
原始 112 result.zip：47917194 bytes
重打包 result.zip：47917194 bytes
```

## 当前建议

下一次提交优先使用重打包版：

```text
submissions/112_major_more_replay_rank_ensemble_repack_checked/result.zip
```

如果 112 重打包版仍提交失败，继续按顺序尝试：

```text
submissions/117_major_row_consensus_replay/result.zip
submissions/113_major_diverse_rank_ensemble/result.zip
submissions/120_major_teacher110_replay_rank_fusion/result.zip
```

当前保底最好：

```text
submissions/121_major_teacher110_multi_replay_ensemble_repack_checked/result.zip
线上分数：1.2156836736721401
```
