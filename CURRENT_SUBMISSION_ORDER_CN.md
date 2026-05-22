# 当前提交顺序与改动总表

日期：2026-05-15

## 当前已知线上结果

当前已知最好成绩：

```text
121_major_teacher110_multi_replay_ensemble_repack_checked
线上分数：1.2156836736721401
```

已知线上反馈：

```text
101 1.2135024872359346
102 1.2148036842867738
103 1.2144142783655167
104 1.2143281923992328
105 提交失败
107 提交失败
110 1.2151966152453952
111 1.2151332588531099
121 1.2156836736721401
```

结论：

```text
121 是当前线上最优。
102 是 110 之前的最优。
101 replay 信号有效。
103/104 分桶门控未超过 102。
111 scaled 分支略低于 110。
105/107 暂不继续优先提交。
```

## 当前提交优先级

如果 112-121 尚未全部提交，建议按以下顺序：

### 第一梯队：优先提交

```text
1. submissions/112_major_more_replay_rank_ensemble_repack_checked/result.zip
2. submissions/117_major_row_consensus_replay/result.zip
3. submissions/113_major_diverse_rank_ensemble/result.zip
4. submissions/120_major_teacher110_replay_rank_fusion/result.zip
```

理由：

```text
121：最新大方向，用 110 作为 teacher 重新生成 replay，再受控融合；top1 相对 110 改约 1.08%。
112：110 主线延伸，更偏 101 replay；文件小，风险适中。
117：行级一致性 replay 推进；比 116/118 更有信息量。
113：加入少量 103/104 多样性，风险较低。
```

### 第二梯队：有提交余量再交

```text
5. submissions/118_major_row_consensus_diverse/result.zip
7. submissions/115_major_110_power_sharpen/result.zip
8. submissions/114_major_110_power_flatten/result.zip
```

理由：

```text
120：验证 119 replay 专家强度，top1 相对 110 改约 9.89%，风险高于 121。
118：接近 113，用于验证 diverse 分支。
115/114：不改排序，只测概率校准；如果平台主要看排序，收益可能有限。
```

### 第三梯队：不建议优先提交

```text
9. submissions/116_major_row_consensus_protect/result.zip
10. submissions/119_major_teacher110_online_replay/result.zip
11. submissions/108_major_candidate_meta_gate_stable/result.zip
12. submissions/109_major_candidate_meta_gate_bold/result.zip
13. submissions/106_major_candidate_meta_expert/result.zip
```

理由：

```text
116：太接近 110，信息量低。
119：纯 teacher110 replay，top1 相对 110 改约 33%，只适合测试上限。
108/109/106：候选 meta 方向尚未证明有效，且 107 已提交失败。
```

### 已知失败或不再建议提交

```text
105_major_hard_expert_switch
107_major_candidate_meta_rank_fusion
98_major_dual_mlp_rank_transplant
99_major_dual_mlp_top_transplant
100_major_triple_consensus_mlp
```

## 最近改动汇总

### 101-102：多教师一致伪在线重放

文件：

```text
make_consensus_online_replay_result_zip.py
LATEST_MAJOR_101_102_CN.md
submissions/README_101_102_CN.md
```

结论：

```text
101 证明 replay 新信号有效。
102 将 101 与 97 做 rank 融合，线上达到 1.2148036842867738。
```

### 103-105：源节点分桶 replay 门控

文件：

```text
blend_adaptive_replay_gate.py
blend_hard_expert_switch.py
LATEST_MAJOR_103_105_CN.md
submissions/README_103_105_CN.md
```

结论：

```text
103/104 低于 102。
105 提交失败。
复杂门控没有超过简单 rank 融合。
```

### 106-109：候选集合 Meta 专家

文件：

```text
make_candidate_meta_expert.py
blend_candidate_meta_gate.py
LATEST_MAJOR_106_109_CN.md
submissions/README_106_109_CN.md
```

结论：

```text
107 提交失败。
该方向目前不优先。
```

### 110-111：线上反馈加权 Rank Ensemble

文件：

```text
blend_online_score_rank_ensemble.py
ONLINE_FEEDBACK_110_111_CN.md
LATEST_MAJOR_110_111_CN.md
submissions/README_110_111_CN.md
```

结论：

```text
110 达到当前最好 1.2151966152453952。
111 低于 110，scaled 分支不继续优先。
```

### 112-115：110 锚点 Replay / 多样性 / 概率校准

文件：

```text
calibrate_submission_power.py
LATEST_MAJOR_112_115_CN.md
submissions/README_112_115_CN.md
```

结论：

```text
112/113 是当前值得优先验证的 110 后续包。
114/115 不改排序，只测概率校准。
```

### 116-118：行级专家一致性门控

文件：

```text
blend_row_consensus_gate.py
LATEST_MAJOR_116_118_CN.md
submissions/README_116_118_CN.md
```

结论：

```text
117 最值得提交。
118 次之。
116 过于接近 110。
```

### 119-121：110 Teacher Pseudo-online Replay

文件：

```text
make_teacher_online_combo_jittor_result_zip.py
LATEST_MAJOR_119_121_CN.md
submissions/README_119_121_CN.md
```

结论：

```text
119 是纯 teacher110 replay，风险高。
120 是 110 与 119 的 rank 融合。
121 是最推荐的最新包，将 119 纳入多 replay ensemble。
```

## 最短提交清单

如果只剩 3 次提交：

```text
1. submissions/112_major_more_replay_rank_ensemble_repack_checked/result.zip
2. submissions/117_major_row_consensus_replay/result.zip
3. submissions/113_major_diverse_rank_ensemble/result.zip
```

如果只剩 1 次提交：

```text
submissions/112_major_more_replay_rank_ensemble_repack_checked/result.zip
```

如果需要保底：

```text
submissions/121_major_teacher110_multi_replay_ensemble_repack_checked/result.zip
```

## 121 提交失败后的处理

121 原始包本地格式检查通过，但平台提示提交失败。已生成重打包版：

```text
submissions/121_major_teacher110_multi_replay_ensemble_repack_checked/result.zip
```

121 重打包版已成功，分数为 1.2156836736721401。
112 也已生成重打包版：

```text
submissions/112_major_more_replay_rank_ensemble_repack_checked/result.zip
```

如果 112 重打包版仍失败，跳过 112，直接提交 117。

详细检查记录：

```text
FORMAT_CHECK_121_CN.md
FORMAT_CHECK_112_CN.md
```
