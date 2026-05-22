# 91 号大方向说明

## 背景

线上反馈：

```text
89 1.2107659416475025
90 1.2064799588604114
```

90 说明高覆盖率 AD/pseudo-online 写回失败。继续调整 `topK` 和 `margin_quantile` 不值得作为主线，因此 91 切换到新的信息源：因果 test-candidate exposure。

## 方法

新增脚本：

```text
make_causal_exposure_result_zip.py
blend_causal_exposure_gated.py
```

91 不写预测伪边，而是统计当前时间之前 test 候选本身的曝光结构。对每个候选使用以下特征打分：

```text
1. source-candidate 历史曝光次数；
2. source-candidate 曝光占 source 已出现行数比例；
3. source-candidate 上次曝光 recency；
4. source 内 candidate 平均列位置；
5. source 近期候选列表命中；
6. candidate 全局历史曝光次数；
7. candidate 全局上次曝光 recency；
8. candidate 全局平均列位置；
9. 当前列位置先验。
```

## 防泄漏

```text
1. 不读取 test label。
2. 不读取 validation label 训练。
3. dataset2 test 按 time、row_id 排序。
4. 同一 timestamp 的所有行先打分。
5. 打分完成后才把该 timestamp 的候选曝光写入统计。
```

因此 91 使用的是“当前 timestamp 之前的候选曝光”，不是全 test 未来统计。

## 生成命令

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python make_causal_exposure_result_zip.py \
  --baseline-dir outputs/website_submission_89_major_teacher88_online_top4q80_40 \
  --output-dir outputs/website_submission_causal_exposure_rank \
  --dataset dataset2 \
  --mode rank \
  --recent-limit 512
```

专家输出：

```text
outputs/website_submission_causal_exposure_rank/result.zip
time_groups = 349
global_exposed = 110368
src_groups = 2180
d2_mad_vs89 = 0.31668016
```

专家和 89 差异极大，所以 91 不做全局大比例混合，而是按 source 已经出现过的行数门控：

```bash
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_causal_exposure_gated.py \
  --base-dir outputs/website_submission_89_major_teacher88_online_top4q80_40 \
  --expert-dir outputs/website_submission_causal_exposure_rank \
  --output-dir outputs/website_submission_91_major_causal_exposure_source_gated \
  --cold-weight 0.01 \
  --warm-weight 0.06 \
  --hot-weight 0.14 \
  --warm-seen 8 \
  --hot-seen 40
mkdir -p submissions/91_major_causal_exposure_source_gated
cp outputs/website_submission_91_major_causal_exposure_source_gated/result.zip submissions/91_major_causal_exposure_source_gated/result.zip
```

## 提交包

```text
submissions/91_major_causal_exposure_source_gated/result.zip
```

校验：

```text
valid = True
dataset1.csv = 61051 x 100
dataset2.csv = 153420 x 100
```

source 分群：

```text
cold_rows = 16207
warm_rows = 33523
hot_rows  = 103690
mean_weight = 0.10878666
```

最终差异：

```text
d1_mad_vs89 = 0.00000000
d2_mad_vs89 = 0.03447949
```

## 预期

91 是高风险探索包，用来判断候选曝光结构是否是有效新信号。

如果 91 提升，后续应把 exposure 特征纳入 Jittor 二阶段校准器；
如果 91 回撤，说明手工 exposure 高权重融合不稳，应转向训练式 gating，而不是继续调 AD/pseudo-edge。

## 线上结果

```text
91 1.197548684166401
```

对比：

```text
89 1.2107659416475025
```

结论：手工 causal exposure 概率融合明显失败，后续不应继续沿该方式调权重。
