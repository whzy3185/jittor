# 92 号大方向说明

## 背景

91 使用因果 test-candidate exposure 做概率线性融合，风险是把 89 的概率分布压平：

```text
89 dataset2 std 约 0.29105
91 dataset2 std 约 0.26502
```

这可能会伤害官方排序/概率指标。因此 92 继续测试 exposure 信号，但改用 rank-boost 重排。

## 方法

新增脚本：

```text
blend_exposure_rank_boost.py
```

核心逻辑：

```text
1. 读取 89 的 dataset2 作为强基线；
2. 读取因果 exposure expert；
3. 对 source 已经在之前 timestamp 出现过多次的行启用 exposure rank boost；
4. 用 exposure 分数扰动排序；
5. 不直接混合概率；
6. 将 89 原本每行的 100 个分数按新排序重新分配。
```

这意味着：

```text
每行分数集合完全继承 89；
只改变哪个 candidate 拿到高分；
保持 89 的概率分布强度，避免 91 的 std 塌缩。
```

## 防泄漏

```text
1. exposure expert 已按 time、row_id 顺序生成；
2. 同一 timestamp 先全部打分，再更新曝光统计；
3. source seen count 也只统计当前 timestamp 之前出现过的 source 行数；
4. 不读取 test label。
```

## 生成命令

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_exposure_rank_boost.py \
  --base-dir outputs/website_submission_89_major_teacher88_online_top4q80_40 \
  --expert-dir outputs/website_submission_causal_exposure_rank \
  --output-dir outputs/website_submission_92_major_exposure_rank_boost \
  --warm-seen 8 \
  --hot-seen 40 \
  --warm-alpha 0.10 \
  --hot-alpha 0.24 \
  --low-agree-scale 0.25 \
  --topn 10 \
  --min-top-overlap 1
mkdir -p submissions/92_major_exposure_rank_boost
cp outputs/website_submission_92_major_exposure_rank_boost/result.zip submissions/92_major_exposure_rank_boost/result.zip
```

## 提交包

```text
submissions/92_major_exposure_rank_boost/result.zip
```

校验：

```text
valid = True
dataset1.csv = 61051 x 100
dataset2.csv = 153420 x 100
```

统计：

```text
changed_rows = 137213
agree_rows = 116305
warm_rows = 33523
hot_rows = 103690
d1_mad_vs89 = 0.00000000
d2_mad_vs89 = 0.03586162
d2_std = 0.29105039
```

## 预期

92 是 exposure 作为“排序重排信号”的验证包。

如果 92 提升而 91 不提升，说明 exposure 有用，但不能直接混概率；
如果 91 和 92 都回撤，说明因果 exposure 的手工特征方向不够强，需要转向训练式二阶段 Jittor ranker。

## 线上结果

```text
92 1.191625248905075
```

对比：

```text
89 1.2107659416475025
91 1.197548684166401
```

结论：92 比 91 更差，说明失败不只是 91 的概率分布塌缩造成的；当前手工 exposure 排序信号本身不匹配线上目标。后续暂停手工 exposure / exposure rank boost。
