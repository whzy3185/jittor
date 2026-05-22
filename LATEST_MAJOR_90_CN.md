# 90 号大改动说明

## 背景

线上反馈：

```text
88 1.2095967342783347
89 1.2107659416475025
```

89 继续提升，但边际收益变小。如果继续同参数 `top4/q80` 递归，很可能只是小幅挤分。因此 90 改为更激进的伪历史写回覆盖率实验。

## 方法

仍然使用 teacher-forced pseudo-online 框架：

```text
make_teacher_online_combo_jittor_result_zip.py
```

但参数明显放宽：

```text
teacher = 89
update_topk = 6
margin_quantile = 0.70
mode = rank
```

与 88/89 对比：

```text
88/89: top4/q80，约 12.3 万 pseudo edges
90:    top6/q70，约 27.7 万 pseudo edges
```

这会显著提高 test-time pseudo history 的覆盖面，用更强的 89 teacher 推动后续时间段历史特征重算。

## 防泄漏处理

```text
1. 不读取 test label。
2. 不使用 validation label 训练新模型。
3. dataset2 test 仍按 time、row_id 排序。
4. 同一 timestamp 内只缓存，不写回。
5. 时间切换后才写入上一 timestamp 的 teacher 高置信 top6。
```

## 生成命令

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
export nvcc_path=""
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
/mnt/e/Jitter/.venv_wsl_cpu/bin/python make_teacher_online_combo_jittor_result_zip.py \
  --include-valid-history \
  --baseline-dir outputs/website_submission_89_major_teacher88_online_top4q80_40 \
  --teacher-zip submissions/89_major_teacher88_online_top4q80_40/result.zip \
  --output-dir outputs/website_submission_teacher89_online_top6q70 \
  --update-topk 6 \
  --margin-quantile 0.70 \
  --mode rank
```

专家输出：

```text
outputs/website_submission_teacher89_online_top6q70/result.zip
pseudo_edges = 276804
d2_mad_vs89 = 0.00417210
```

## 提交包

融合命令：

```bash
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_website_csv_by_dataset.py \
  --inputs outputs/website_submission_89_major_teacher88_online_top4q80_40 outputs/website_submission_teacher89_online_top6q70 \
  --weights-dataset1 1,0 \
  --weights-dataset2 0.45,0.55 \
  --output-dir outputs/website_submission_90_major_teacher89_top6q70_55
mkdir -p submissions/90_major_teacher89_top6q70_55
cp outputs/website_submission_90_major_teacher89_top6q70_55/result.zip submissions/90_major_teacher89_top6q70_55/result.zip
```

应提交：

```text
submissions/90_major_teacher89_top6q70_55/result.zip
```

校验：

```text
valid = True
dataset1.csv = 61051 x 100
dataset2.csv = 153420 x 100
d1_mad_vs89 = 0.00000000
d2_mad_vs89 = 0.00229466
```

## 预期

90 的目标不是稳健小涨，而是测试更高覆盖 pseudo history 是否能突破 89 的递归天花板。

如果 90 继续涨，可以继续探索 top8/q65 或对高活跃 source 使用更高 topK；如果 90 回撤，则说明 89 附近的伪历史覆盖率已经接近最优，应切换到 dataset2 专用 ranker 或 test-candidate exposure 特征。

## 线上结果

```text
90 1.2064799588604114
```

对比：

```text
89 1.2107659416475025
```

结论：高覆盖率 pseudo-online 写回失败。继续在 AD/pseudo-edge 的 topK 和 margin 分位上调参，预期收益低且会消耗提交次数。下一阶段改为因果候选曝光和 source 分群方向。
