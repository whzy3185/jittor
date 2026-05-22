# 89 号大改动说明

## 背景

线上反馈：

```text
80 1.2061364461925292
88 1.2095967342783347
```

88 已经验证 teacher-forced pseudo-online 有效。因此 89 不回退到 source/time 门控，也不做小权重扰动，而是做递归 teacher：

```text
teacher: 80 -> 88
```

## 方法

继续使用：

```text
make_teacher_online_combo_jittor_result_zip.py
```

变化：

```text
1. teacher zip 从 80 换成 88。
2. 按 dataset2 test 的 time、row_id 顺序推理。
3. 同一 timestamp 内不写回。
4. 时间切换后，把 teacher=88 的高置信 top4 候选写入伪历史。
5. 输出仍由 Jittor combo 模型基于动态历史重算。
```

生成命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
export nvcc_path=""
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
/mnt/e/Jitter/.venv_wsl_cpu/bin/python make_teacher_online_combo_jittor_result_zip.py \
  --include-valid-history \
  --baseline-dir outputs/website_submission_88_major_teacher80_online_top4q80_40 \
  --teacher-zip submissions/88_major_teacher80_online_top4q80_40/result.zip \
  --output-dir outputs/website_submission_teacher88_online_top4q80 \
  --update-topk 4 \
  --margin-quantile 0.80 \
  --mode rank
```

专家输出：

```text
outputs/website_submission_teacher88_online_top4q80/result.zip
pseudo_edges = 123320
```

## 提交包

融合命令：

```bash
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_website_csv_by_dataset.py \
  --inputs outputs/website_submission_88_major_teacher80_online_top4q80_40 outputs/website_submission_teacher88_online_top4q80 \
  --weights-dataset1 1,0 \
  --weights-dataset2 0.60,0.40 \
  --output-dir outputs/website_submission_89_major_teacher88_online_top4q80_40
mkdir -p submissions/89_major_teacher88_online_top4q80_40
cp outputs/website_submission_89_major_teacher88_online_top4q80_40/result.zip submissions/89_major_teacher88_online_top4q80_40/result.zip
```

应提交：

```text
submissions/89_major_teacher88_online_top4q80_40/result.zip
```

校验：

```text
valid = True
dataset1.csv = 61051 x 100
dataset2.csv = 153420 x 100
d1_mad_vs88 = 0.00000000
d2_mad_vs88 = 0.00187324
```

## 预期

89 用 88 作为更强 teacher 进一步修正伪历史。如果 89 继续提升，说明递归 teacher 可继续做一次更激进版本；如果 89 回撤，则 88 可能是当前 teacher-forced 路线的较优平衡点。

## 线上结果

```text
89 1.2107659416475025
```

对比：

```text
88 1.2095967342783347
提升约 0.00117
```

结论：递归 teacher 继续有效，但边际收益下降。下一步不再同参数递归，而是提高伪历史写回覆盖率和最终融合比例，进行更激进探索。
