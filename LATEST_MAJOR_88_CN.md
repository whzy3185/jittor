# 88 号大改动说明

## 背景

线上反馈显示：

```text
80 1.2061364461925292
81 1.200475678432507
84 1.2025460175004976
86 1.2032310925649679
87 1.204522167055046
```

81/84/86/87 都没有超过 80，说明继续在 80 和 top4/top5 online 专家之间做 source/time/agreement 门控，收益已经不足。

## 本轮方向

本轮不再做小幅权重扰动，而是新增 teacher-forced pseudo-online 路线：

```text
make_teacher_online_combo_jittor_result_zip.py
```

核心思想：

```text
用当前最强的 80 号提交作为 teacher 控制伪历史写回。
当前行输出仍由 Jittor combo 模型重算。
teacher 只决定哪些 test 候选作为 pseudo edge 写入后续历史。
```

这样相比原始 `make_online_combo_jittor_result_zip.py` 有一个关键区别：

```text
原始 online combo：combo_jittor 自己预测、自己写回。
88 teacher-online：80 号强集成预测写回，combo_jittor 使用被 teacher 修正过的动态历史继续推理。
```

## 防泄漏处理

```text
1. 不读取 test label。
2. 不读取 validation label 训练新模型。
3. dataset2 test 按 time、row_id 排序。
4. 同一 timestamp 内只打分和缓存，不立即写回。
5. 时间切换后才把上一 timestamp 的高置信 teacher top4 写入伪历史。
```

## 生成命令

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
export nvcc_path=""
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
/mnt/e/Jitter/.venv_wsl_cpu/bin/python make_teacher_online_combo_jittor_result_zip.py \
  --include-valid-history \
  --output-dir outputs/website_submission_teacher80_online_top4q80 \
  --update-topk 4 \
  --margin-quantile 0.80 \
  --mode rank
```

专家输出：

```text
outputs/website_submission_teacher80_online_top4q80/result.zip
pseudo_edges = 123348
d2_mad_vs80 = 0.00667563
```

## 88 提交包

融合命令：

```bash
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_website_csv_by_dataset.py \
  --inputs outputs/website_submission_80_major_top4q85_35 outputs/website_submission_teacher80_online_top4q80 \
  --weights-dataset1 1,0 \
  --weights-dataset2 0.60,0.40 \
  --output-dir outputs/website_submission_88_major_teacher80_online_top4q80_40
mkdir -p submissions/88_major_teacher80_online_top4q80_40
cp outputs/website_submission_88_major_teacher80_online_top4q80_40/result.zip submissions/88_major_teacher80_online_top4q80_40/result.zip
```

应提交：

```text
submissions/88_major_teacher80_online_top4q80_40/result.zip
```

校验结果：

```text
valid = True
dataset1.csv = 61051 x 100
dataset2.csv = 153420 x 100
d1_mad_vs80 = 0.00000000
d2_mad_vs80 = 0.00267025
```

## 预期

88 的预期是：如果 80 的 top ranking 比单个 combo_jittor 更可靠，那么 teacher 控制写回能减少自举错误，并让后续时间段的历史特征更接近线上真实动态。

## 线上结果

```text
88 1.2095967342783347
```

对比：

```text
80 1.2061364461925292
提升约 0.00346
```

结论：teacher-forced pseudo-online 方向有效，优于 81/84/86/87 的门控融合。

如果 88 仍低于 80，则说明 pseudo-online 写回路线接近上限，后续应切到：

```text
1. dataset2 专用 Jittor ranker 重新训练；
2. 因果 test-candidate exposure 特征；
3. 以 source/time/group 统计为输入的二阶段校准器。
```
