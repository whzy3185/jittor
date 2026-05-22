# 93 号大方向说明

## 背景

90 的高覆盖 AD/pseudo-online 写回失败，91/92 切到手工 exposure 信号。93 进一步切到训练式二阶段：用 Jittor 在 dataset2 的 `split=0` 内部 holdout 上训练非线性 combo MLP ranker。

## 训练约束

```text
1. 只使用 dataset2 train.csv 中 split=0 的边作为训练来源；
2. 不把官方 split=1 validation labels 作为训练目标；
3. 在 split=0 内部按时间切尾部 holdout；
4. 用内部 holdout 正样本 + test candidate pool / popularity 负样本训练 pairwise loss；
5. 全程使用 Jittor，不使用 PyTorch。
```

## 新增脚本

```text
train_combo_mlp_jittor.py
make_combo_mlp_result_zip.py
```

训练命令：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
export nvcc_path=""
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
/mnt/e/Jitter/.venv_wsl_cpu/bin/python train_combo_mlp_jittor.py \
  --dataset dataset2 \
  --output-dir outputs/combo_mlp_jittor_split0_internal \
  --max-val-events 40000 \
  --negatives 49 \
  --epochs 8 \
  --batch-size 2048 \
  --hidden-dim 128 \
  --dropout 0.12 \
  --lr 0.0012 \
  --seed 2026
```

训练结果：

```text
history_edges = 1755677
internal_holdout_edges = 40000
candidates = 2000000
negatives = 49
feature_dim = 38
loss: 0.5405 -> 0.4627
```

MLP 专家生成：

```bash
/mnt/e/Jitter/.venv_wsl_cpu/bin/python make_combo_mlp_result_zip.py \
  --include-valid-history \
  --model-dir outputs/combo_mlp_jittor_split0_internal \
  --baseline-dir outputs/website_submission_89_major_teacher88_online_top4q80_40 \
  --output-dir outputs/website_submission_combo_mlp_split0_internal_rank \
  --mode rank
```

MLP 专家：

```text
outputs/website_submission_combo_mlp_split0_internal_rank/result.zip
d2_mad_vs89 = 0.36104817
```

## 93 提交包

由于 MLP 专家和 89 差异很大，93 不直接混合概率，而是做 rank-boost：

```text
1. 保留 89 每行的 100 个分数集合；
2. 用 MLP rank 专家扰动候选排序；
3. 将 89 的分数按新排序重新分配；
4. 保持 89 的行内分布强度。
```

生成命令：

```bash
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_exposure_rank_boost.py \
  --base-dir outputs/website_submission_89_major_teacher88_online_top4q80_40 \
  --expert-dir outputs/website_submission_combo_mlp_split0_internal_rank \
  --output-dir outputs/website_submission_93_major_combo_mlp_rank_boost \
  --warm-seen 4 \
  --hot-seen 24 \
  --warm-alpha 0.08 \
  --hot-alpha 0.18 \
  --low-agree-scale 0.20 \
  --topn 10 \
  --min-top-overlap 1
mkdir -p submissions/93_major_combo_mlp_rank_boost
cp outputs/website_submission_93_major_combo_mlp_rank_boost/result.zip submissions/93_major_combo_mlp_rank_boost/result.zip
```

应提交：

```text
submissions/93_major_combo_mlp_rank_boost/result.zip
```

校验与差异：

```text
valid = True
dataset1.csv = 61051 x 100
dataset2.csv = 153420 x 100
changed_rows = 143160
agree_rows = 142802
d1_mad_vs89 = 0.00000000
d2_mad_vs89 = 0.01600329
d2_std = 0.29105039
```

## 预期

93 用训练式 MLP 信号替代纯手工 exposure。若 93 提升，下一步应继续训练多 seed / 更大 hidden / 加 exposure 特征的 Jittor 二阶段 ranker；若 93 回撤，说明目前 split=0 内部 holdout 训练分布和 test 候选分布仍有偏差。

## 线上结果

```text
93 1.2113817385926295
```

对比：

```text
89 1.2107659416475025
提升约 0.00062
```

结论：训练式 Jittor MLP ranker 方向有效。后续应继续沿 MLP 专家做更强重排或多 seed MLP，而不是继续手工 exposure。
