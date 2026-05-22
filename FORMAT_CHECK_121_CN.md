# 121 提交失败后的格式检查

日期：2026-05-15

## 检查对象

原始提交包：

```text
submissions/121_major_teacher110_multi_replay_ensemble/result.zip
```

重打包后的提交包：

```text
submissions/121_major_teacher110_multi_replay_ensemble_repack_checked/result.zip
```

## 检查结论

本地未发现 121 的格式问题。

```text
zip CRC 检查：通过
zip 根目录文件：dataset1.csv, dataset2.csv
dataset1.csv 行数：61051
dataset2.csv 行数：153420
每行列数：100
概率范围：[0, 1]
NaN/Inf：无
小数位：8 位
```

## 文件大小

```text
原始 121 result.zip：45589117 bytes
重打包 result.zip：45589117 bytes
```

两个 zip 的 MD5 不同，但内容文件一致；重打包只是重新写入 zip 容器，便于排除平台对 zip 元数据的异常兼容问题。

## 建议

优先尝试重新提交：

```text
submissions/121_major_teacher110_multi_replay_ensemble_repack_checked/result.zip
```

如果重打包版仍然提交失败，跳过 121，按当前顺序提交：

```text
1. submissions/112_major_more_replay_rank_ensemble/result.zip
2. submissions/117_major_row_consensus_replay/result.zip
3. submissions/113_major_diverse_rank_ensemble/result.zip
4. submissions/120_major_teacher110_replay_rank_fusion/result.zip
```

保底仍为：

```text
submissions/110_major_online_weighted_rank_ensemble/result.zip
```
