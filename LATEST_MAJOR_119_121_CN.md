# 119-121 大方向实验记录：110 Teacher 的 Pseudo-online Replay

日期：2026-05-15

## 背景

当前已知线上最优：

```text
110 1.2151966152453952
```

101 的多教师 replay 已证明有效，102/110 也证明 replay 信号可以通过 rank ensemble 提升。

本轮不再只融合已有 CSV，而是用当前最强 110 作为 teacher，重新生成 pseudo-online replay 专家。

## 119：110 teacher pseudo-online replay

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
export nvcc_path=
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
/mnt/e/Jitter/.venv_wsl_cpu/bin/python make_teacher_online_combo_jittor_result_zip.py \
  --data-root data/official_raw \
  --model-dir outputs/combo_jittor \
  --baseline-dir outputs/website_submission_110_major_online_weighted_rank_ensemble \
  --teacher-zip submissions/110_major_online_weighted_rank_ensemble/result.zip \
  --output-dir outputs/website_submission_119_major_teacher110_online_replay \
  --mode rank \
  --include-valid-history \
  --update-topk 4 \
  --margin-quantile 0.78
```

统计：

```text
pseudo_edges = 194264
dataset2 rows = 153420
```

提交包：

```text
submissions/119_major_teacher110_online_replay/result.zip
```

相对差异：

```text
119 vs 110: d2_mad=0.02006172, top1_change=0.329664
119 vs 112: d2_mad=0.01682993, top1_change=0.329559
119 vs 117: d2_mad=0.01494529, top1_change=0.329514
119 vs 101: d2_mad=0.00458591, top1_change=0.332356
119 vs 102: d2_mad=0.02685440, top1_change=0.332036
```

解释：119 是纯 replay 专家，top1 改动约 33%，风险极高，不建议优先提交。

## 120：110 与 119 的 rank 融合

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_rank_consensus.py \
  --base-dir outputs/website_submission_110_major_online_weighted_rank_ensemble \
  --expert-dirs outputs/website_submission_119_major_teacher110_online_replay \
  --output-dir outputs/website_submission_120_major_teacher110_replay_rank_fusion \
  --weight 0.35 \
  --sharpen 0.0 \
  --chunksize 4096
```

提交包：

```text
submissions/120_major_teacher110_replay_rank_fusion/result.zip
```

相对差异：

```text
120 vs 110: d2_mad=0.00702160, top1_change=0.098931
120 vs 112: d2_mad=0.00435358, top1_change=0.098846
120 vs 117: d2_mad=0.00442833, top1_change=0.098788
120 vs 101: d2_mad=0.01375801, top1_change=0.144232
120 vs 102: d2_mad=0.01411472, top1_change=0.101825
```

解释：120 是中高风险验证包，判断 110 teacher replay 是否真有线上价值。

## 121：110 + 119 + 112 + 102 + 101 多 replay ensemble

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_online_score_rank_ensemble.py \
  --base-dir outputs/website_submission_110_major_online_weighted_rank_ensemble \
  --expert-dirs outputs/website_submission_110_major_online_weighted_rank_ensemble outputs/website_submission_119_major_teacher110_online_replay outputs/website_submission_112_major_more_replay_rank_ensemble outputs/website_submission_102_major_replay_rank_fusion outputs/website_submission_101_major_consensus_replay_strict \
  --weights 0.48 0.24 0.12 0.10 0.06 \
  --output-dir outputs/website_submission_121_major_teacher110_multi_replay_ensemble \
  --preserve-base-scale 0.0 \
  --chunksize 4096
```

提交包：

```text
submissions/121_major_teacher110_multi_replay_ensemble/result.zip
```

相对差异：

```text
121 vs 110: d2_mad=0.00606811, top1_change=0.010813
121 vs 112: d2_mad=0.00262094, top1_change=0.010566
121 vs 117: d2_mad=0.00343936, top1_change=0.010435
121 vs 101: d2_mad=0.01465830, top1_change=0.125238
121 vs 102: d2_mad=0.01346928, top1_change=0.014672
```

解释：121 是最适合作为本轮主提交的包。它引入 119 新 replay 专家，但仍以 110 为锚点，top1 相对 110 改约 1.08%。

## 格式校验

119、120、121 均已检查：

```text
result.zip 内含 dataset1.csv 和 dataset2.csv
dataset1.csv: 61051 x 100
dataset2.csv: 153420 x 100
概率范围在 [0, 1]
NaN = 0
```

## 提交建议

推荐顺序：

```text
1. 121_major_teacher110_multi_replay_ensemble
2. 120_major_teacher110_replay_rank_fusion
3. 119_major_teacher110_online_replay
```

理由：

```text
121 风险可控且包含新 replay 信号。
120 用于验证 119 replay 专家强度。
119 是纯 replay 专家，top1 改动约 33%，只适合测试上限，不适合优先提交。
```
