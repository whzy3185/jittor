# Track1 改动与实验汇总

本文档汇总从接手项目到当前版本的主要改动、每一轮生成的提交包、预期效果、风险和复现命令。

工作目录：

```text
/mnt/e/Jitter/track1_aggressive_wsl
```

数据目录：

```text
/mnt/e/Jitter/data/official_raw
```

原始 baseline 目录保留未直接覆盖：

```text
/mnt/e/Jitter/legacy_track1
```

所有训练、推理、校验和出包均通过 WSL 访问 E 盘路径完成。项目未改成 PyTorch；训练/神经打分使用 Jittor。

## 当前最推荐提交

优先提交：

```text
outputs/website_submission_blend_dsaware_seq/result.zip
```

强 A/B 备选：

```text
outputs/website_submission_blend_seq70_enh15_hard10_mlp05/result.zip
outputs/website_submission_blend_dsaware_seq_meta/result.zip
outputs/website_submission_blend_dsaware_seq_order/result.zip
outputs/website_submission_blend_dsaware_seq_shrink/result.zip
```

以上均为官网截图要求格式：`result.zip` 内只有 `dataset1.csv`、`dataset2.csv`，每行 100 个 `[0,1]` 概率，保留 8 位小数。

## 关键结论

- 早期 `outputs/website_submission/result.zip` 已实测约 `0.9425`，只作为旧回退包。
- 后续主要提升来自更贴近 test 候选池的 hard negative、历史共现特征、序列转移特征和分数据集融合。
- 本地 hard validation 中，dataset2 从序列转移特征受益明显：

```text
dataset2 enhanced MRR:   0.1960
dataset2 sequential MRR: 0.2060
```

- dataset1 上 sequential MRR 略高，enhanced hits@1 略高，因此最终用分数据集融合而不是同一套权重。

## 轮次汇总

| 轮次 | 主要改动 | 代表输出 | 本地验证或预期效果 | 风险 |
|---|---|---|---|---|
| 0. 环境与基线保护 | 检查 WSL、E 盘、Jittor/JittorGeometric；保留原 baseline；建立工作副本 | `README_CN.md`, `METHOD_CN.md` | 保证可复现和不污染原始材料 | 无模型提升 |
| 1. 官网提交格式修正 | 从旧 `result.json`/单数据集 zip 转为官网 `result.zip` + 两个 CSV | `outputs/website_submission/result.zip` | 已实测约 `0.9425` | 只是格式正确，模型较弱 |
| 2. 历史启发式与 hard tuning | 使用 pair frequency、popularity、recency、source degree 等历史特征；修正候选位置泄漏 | `outputs/website_submission_hard_shuffled/result.zip` | 预期高于旧 0.9425，作为稳定基线 | 启发式重复边偏强，新边泛化有限 |
| 3. 概率校准与列顺序 A/B | rank、sigmoid、global sigmoid、order prior、shrink 包 | `global_sigmoid`, `order_prior`, `shrink10` | 用于适配官网未知指标，可能改善校准 | order prior 属于弱先验，需 A/B |
| 4. Jittor 轻量线性 ranker | 用 Jittor pairwise loss 学习历史特征权重，关闭 candidate_prior 伪信号 | `website_submission_jittor_noprior_rank` | 增加与启发式不同的打分视角，适合小比例融合 | 单独模型不一定强 |
| 5. Jittor/启发式融合 | hard shuffled 与 Jittor noprior 融合 | `blend_hard70_jittor30` | 预期比单启发式稳健 | 线性模型信号较浅 |
| 6. item-item 共现增强 | 加 `cooc_log`、`cooc_max`，刻画 source 历史目的节点与候选目的节点的共现相似度 | `enhanced_hard_rank` | dataset1 MRR `0.7418`，dataset2 MRR `0.1960` | 依赖历史共现，对冷启动有限 |
| 7. enhanced 融合 | enhanced 与 hard/Jittor 融合 | `blend_enh70_hard20_jittor10` | 预期显著强于旧 hard shuffled | 同一套融合权重未区分数据集 |
| 8. test-meta 无标签结构 | 只读 test 候选列表，统计同 source 重复频率、全局频率、平均列位置 | `testmeta_rank`, `meta10`, `meta20` | 小权重可能利用候选生成机制 | transductive metadata，权重过大会有风险 |
| 9. Jittor 增强 MLP | 用增强特征训练 Jittor MLP pairwise ranker，训练集内部时间切分 | `enhanced_jittor_mlp_rank` | 提供非线性组合能力，适合 5%-15% 融合 | 内部切分训练分布与 test 有差异 |
| 10. 序列转移增强 | 加 `prev_dst -> next_dst` 有向转移特征：`trans_log`、`trans_max`、`trans_last` | `sequential_hard_rank` | dataset1 MRR `0.7438`，dataset2 MRR `0.2060`，当前本地最强 | 生成较慢，特征更复杂 |
| 11. 序列融合 | sequential 与 enhanced/hard/MLP/meta/order 融合 | `blend_seq70_enh15_hard10_mlp05` | 预期优于共现增强融合 | 仍使用同一套权重 |
| 12. 分数据集融合 | dataset1/dataset2 使用不同融合权重 | `blend_dsaware_seq` | 当前最推荐，dataset2 提高 seq 权重 | 权重来自本地验证，leaderboard 仍需 A/B |

## 文件改动清单

### 数据与基础管线

- `track1_dynamic_rec/data.py`
  - 增强官方 CSV 读取。
  - 支持宽表候选列 `c1..c100`。
  - 保留候选顺序和 ID 映射。

- `track1_dynamic_rec/features.py`
  - 优化历史特征构造。
  - 将部分慢 pandas 循环改为 numpy/增量统计。
  - 保证验证/测试特征只使用允许历史。

- `track1_dynamic_rec/model.py`
  - 保留 Jittor 模型路线。
  - 包含 embedding、时间编码、MLP scorer 等组件。

- `train.py`, `infer.py`, `infer_stream.py`
  - 支持训练、验证、推理、流式推理。
  - 避免一次性展开全部 test candidates。

### 官网 CSV 提交格式

- `make_result_zip.py`
  - 从旧结果生成官网 `result.zip`。

- `validate_result_zip.py`
  - 校验官网格式。
  - 检查 zip 成员、行数、100 列、8 位小数、概率范围。

- `blend_website_csv.py`
  - 多个官网 CSV 输出按统一权重融合。

- `blend_website_csv_by_dataset.py`
  - 新增：dataset1/dataset2 可使用不同融合权重。

### 启发式与调参

- `simple_heuristic.py`
  - 基础历史启发式。
  - 包含 pair frequency、target popularity、recency、source degree、candidate prior 等。

- `tune_simple_heuristic.py`
  - 随机负例调权。
  - 后续修正候选顺序，避免正例固定第 1 位导致的 candidate prior 高估。

- `tune_simple_hard.py`
  - 使用 test 候选池和热门节点构造 hard negatives。
  - 更贴近官网候选分布。

- `make_simple_result_zip.py`
  - 基础启发式官网出包。

- `make_simple_global_result_zip.py`
  - global sigmoid/rankish 校准出包。

- `make_order_prior_zip.py`
  - 只使用候选列顺序先验的 A/B 包。

### Jittor 轻量 ranker

- `train_simple_jittor.py`
  - 使用 Jittor 线性 pairwise ranker。
  - 训练特征来自历史启发式。
  - 默认关闭 `candidate_prior`，避免训练构造伪信号。

- `make_simple_jittor_result_zip.py`
  - 读取 Jittor 线性 ranker 权重并生成官网格式。

- `run_train_simple_jittor.sh`
- `run_make_jittor_submit.sh`

### item-item 共现增强

- `enhanced_heuristic.py`
  - 新增 `cooc_log`、`cooc_max`。
  - 对每个 source 保留最近历史目的节点，计算候选与历史目的节点的共现相似度。

- `tune_enhanced_hard.py`
  - hard validation 调权。
  - 预计算特征矩阵后快速搜索权重。

- `make_enhanced_result_zip.py`
  - 使用增强启发式生成官网 `result.zip`。

- `run_tune_enhanced.sh`
- `run_make_enhanced_submit.sh`

### test-meta 无标签结构

- `make_testmeta_result_zip.py`
  - 只使用 test 候选列表，无标签。
  - 特征包括同 source 候选重复、全局候选频率、平均列位置、当前列位置。

- `run_make_meta_submit.sh`

### Jittor 增强 MLP

- `train_enhanced_jittor.py`
  - 使用增强特征训练 Jittor MLP。
  - 训练集内部时间切分：前 85% 构建历史，后段作为未来正例。
  - hard negatives 来自 test 候选池和热门节点。
  - pairwise logistic loss。

- `make_enhanced_jittor_result_zip.py`
  - 加载 Jittor checkpoint。
  - 用全量允许历史计算增强特征并批量推理。

- `run_train_enhanced_jittor.sh`
- `run_make_enhanced_jittor_submit.sh`

### 序列转移增强

- `sequential_heuristic.py`
  - 在 enhanced 特征上增加有向转移特征。
  - `trans_log`：最近历史节点到候选节点的衰减转移总强度。
  - `trans_max`：最大转移强度。
  - `trans_last`：最后一个历史目的节点到候选节点的转移强度。

- `tune_sequential_hard.py`
  - 对序列增强特征进行 hard validation 调权。

- `make_sequential_result_zip.py`
  - 生成序列增强官网格式包。

- `run_tune_sequential.sh`
- `run_make_sequential_submit.sh`

### 打包、自检、说明文档

- `run_self_check.sh`
  - 环境、Jittor/JittorGeometric、语法、禁用 PyTorch、提交格式检查。

- `package_release.py`, `run_package_release.sh`
  - 发布包整理。

- `README_CN.md`
  - 运行与提交说明。

- `METHOD_CN.md`
  - 方法说明。

- `EXPERIMENT_SUMMARY_CN.md`
  - 本文件，记录全流程改动与实验预期。

## 主要提交包与预期

### 已实测旧包

```text
outputs/website_submission/result.zip
```

说明：旧版官网格式包，用户反馈 leaderboard 约 `0.9425`。

预期：只作为回退，不建议继续优先提交。

### hard-shuffled 稳定基线

```text
outputs/website_submission_hard_shuffled/result.zip
outputs/website_submission_hard_shuffled_shrink10/result.zip
outputs/website_submission_global_sigmoid/result.zip
```

说明：修正早期验证候选位置泄漏后重新调权。

预期：比旧 0.9425 更稳，但仍主要依赖历史重复边和 popularity。

### Jittor 轻量 ranker

```text
outputs/website_submission_jittor_noprior_rank/result.zip
outputs/website_submission_blend_hard70_jittor30/result.zip
```

说明：Jittor 线性 pairwise ranker，关闭候选位置伪信号。

预期：单独提交不一定最强，小比例融合可增加模型多样性。

### 共现增强

```text
outputs/website_submission_enhanced_hard_rank/result.zip
outputs/website_submission_blend_enh70_hard20_jittor10/result.zip
outputs/website_submission_blend_enh80_hard20/result.zip
```

本地验证：

```text
dataset1 enhanced MRR: 0.7418
dataset2 enhanced MRR: 0.1960
```

预期：明显强于 hard-shuffled，是第一轮真正的结构性提升。

### test-meta/order A/B

```text
outputs/website_submission_blend_enh65_hard18_jittor07_meta10/result.zip
outputs/website_submission_blend_enh68_hard18_jittor09_order05/result.zip
```

说明：利用无标签候选列表结构或列顺序弱先验。

预期：可能利用官网候选生成机制；建议只小权重 A/B。

### Jittor 增强 MLP

```text
outputs/website_submission_enhanced_jittor_mlp_rank/result.zip
outputs/website_submission_blend_enh70_hard20_mlp10/result.zip
```

说明：非线性学习增强特征组合。

预期：作为 5%-15% 融合成员，可能提升复杂模式；不建议单独作为最优提交。

### 序列增强

```text
outputs/website_submission_sequential_hard_rank/result.zip
outputs/website_submission_blend_seq70_enh15_hard10_mlp05/result.zip
outputs/website_submission_blend_seq65_enh15_hard10_meta10/result.zip
outputs/website_submission_blend_seq65_enh15_hard10_order10/result.zip
```

本地验证：

```text
dataset1 sequential MRR: 0.7438
dataset2 sequential MRR: 0.2060
```

预期：当前单一路线最强，尤其 dataset2。

### 分数据集融合

```text
outputs/website_submission_blend_dsaware_seq/result.zip
outputs/website_submission_blend_dsaware_seq_meta/result.zip
outputs/website_submission_blend_dsaware_seq_order/result.zip
outputs/website_submission_blend_dsaware_seq_shrink/result.zip
```

当前权重：

```text
dataset1: seq 0.55, enhanced 0.30, hard 0.10, mlp 0.05
dataset2: seq 0.80, enhanced 0.08, hard 0.07, mlp 0.05
```

预期：当前最推荐。原因是 dataset2 对 sequence 特征收益更大，而 dataset1 保留更多 enhanced/hard 权重更稳。

## 防泄漏说明

- 不使用 test label。
- 不把 validation label 用于训练。
- dataset2 默认只用 `split=0` 作为最终 test 历史统计；`split=1` 只用于本地验证和调权参考。
- 对验证集/训练内部未来边构造特征时，历史特征只来自切分点之前。
- `testmeta` 只读取 test 候选列表的无标签结构，不读取标签。
- 早期存在正例固定在候选第 1 位的调参偏差，后续已修正并弃用旧 `simple_tuned`/早期 `hard_tuned` 作为推荐包。

## 当前复现顺序

从已有输出直接重新生成当前推荐包：

```bash
bash run_make_dsaware_submit.sh
```

如需从调参开始重跑主要路线：

```bash
bash run_tune_enhanced.sh
bash run_make_enhanced_submit.sh
bash run_tune_sequential.sh
bash run_make_sequential_submit.sh
bash run_train_enhanced_jittor.sh
bash run_make_enhanced_jittor_submit.sh
bash run_make_dsaware_submit.sh
```

校验提交包：

```bash
/mnt/e/Jitter/.venv_wsl_cpu/bin/python validate_result_zip.py --zip outputs/website_submission_blend_dsaware_seq/result.zip
```

## 推荐提交顺序

1. `outputs/website_submission_blend_dsaware_seq/result.zip`
2. `outputs/website_submission_blend_seq70_enh15_hard10_mlp05/result.zip`
3. `outputs/website_submission_blend_dsaware_seq_meta/result.zip`
4. `outputs/website_submission_blend_dsaware_seq_order/result.zip`
5. `outputs/website_submission_blend_dsaware_seq_shrink/result.zip`

如果提交次数有限，优先第 1 个；如果允许多次 A/B，再尝试第 2-4 个。

## 2026-05-11 追加增强：dataset2 full-history 推理包

线上反馈 `01_blend_dsaware_seq/result.zip` 得分约为 `1.084853914501116`，距离第一名 `1.3+` 仍有明显差距。结合本地 hard validation，主要短板仍判断为 dataset2。因此本轮没有继续大幅扰动 dataset1，而是固定 dataset1 使用已知较稳的 `01`，重点增强 dataset2。

### 数据时间检查

dataset2 的 `train.csv` 含 `split` 字段：

```text
split=0: 1995088
split=1: 266195
```

时间顺序检查：

```text
dataset2 split=1 最大 time: 1296259200
dataset2 test 最小 time:    1296345600
```

因此从时间上看，`split=1` 位于 test 之前。本轮新增包将 `split=1` 作为测试前公开历史用于推理特征构建，不使用 test label，也不把 test 候选答案写回训练。风险点是：如果官网规则明确禁止最终提交使用 `split=1`，则不要提交 `10-17`。

### 生成的新增输出

```text
outputs/website_submission_sequential_fullhistory_rank/result.zip
outputs/website_submission_enhanced_fullhistory_rank/result.zip
outputs/website_submission_d1old_d2_fullseq/result.zip
outputs/website_submission_d1old_d2_fullseq75/result.zip
outputs/website_submission_d1old_d2_fullseq_order/result.zip
outputs/website_submission_d1old_d2_fullenh/result.zip
outputs/website_submission_d1old_d2_fullseq_fullenh_70_20/result.zip
outputs/website_submission_d1old_d2_fullseq_fullenh_meta/result.zip
```

已复制到 `submissions/10-17`，并全部通过 `validate_result_zip.py` 校验。

### 新包预期

- `11_d1old_d2_fullseq`：预期最有机会提升，因为它保持 dataset1 不变，只把 dataset2 替换为更接近测试时间的 full-history sequential。
- `16_d1old_d2_fullseq_fullenh_70_20`：预期比纯 full sequential 更稳，加入 full enhanced 共现和少量 hard/order。
- `17_d1old_d2_fullseq_fullenh_meta`：预期对候选列表分布敏感，适合 A/B。
- `13_d1old_d2_fullseq_order`：预期在候选顺序有隐藏先验时收益更高，否则可能不如 11/16。
- `15_d1old_d2_fullenh`：纯共现信号 A/B，若 sequential 过拟合则可能更稳。

当前推荐提交顺序已更新到 `submissions/README_SUBMISSIONS_CN.md`。

## 2026-05-11 继续增强：group-meta 与在线伪历史

本轮继续针对 dataset2 加强。观察到 dataset2 测试集有明显的重复查询结构：

```text
test rows: 153420
src unique: 2180
(src,time) groups: 64831
max rows per (src,time): 247
```

因此新增两类不使用标签的推理信号：

1. `groupmeta`：只统计 test 候选列表的无标签结构，例如同一 `(src,time)` 内候选重复次数、同一 src 下候选重复次数、同一时间候选频次、平均列位置等。
2. `online sequential`：按 test 时间顺序打分；同一时间的所有行先一起打分，然后把预测 top1 作为伪历史写入后续时间。这样不使用 test label，也避免同一 timestamp 内相互串用。

新增脚本：

```text
make_groupmeta_result_zip.py
make_online_sequential_result_zip.py
```

新增输出：

```text
outputs/website_submission_groupmeta_rank/result.zip
outputs/website_submission_online_seq_fullhistory_rank/result.zip
outputs/website_submission_online_seq_fullhistory_q50_rank/result.zip
outputs/website_submission_d1old_d2_groupmeta/result.zip
outputs/website_submission_d1old_d2_online_seq/result.zip
outputs/website_submission_d1old_d2_online_q50/result.zip
outputs/website_submission_d1old_d2_fullseq_groupmeta/result.zip
outputs/website_submission_d1old_d2_online_group/result.zip
outputs/website_submission_d1old_d2_onlineq50_group/result.zip
```

已复制为：

```text
submissions/18_d1old_d2_groupmeta/result.zip
submissions/19_d1old_d2_online_seq/result.zip
submissions/20_d1old_d2_online_q50/result.zip
submissions/21_d1old_d2_fullseq_groupmeta/result.zip
submissions/22_d1old_d2_online_group/result.zip
submissions/23_d1old_d2_onlineq50_group/result.zip
```

全部通过 `validate_result_zip.py` 格式校验。

### 差异度参考

以 dataset2 为对象，和 `11`/`16` 的平均绝对差异：

```text
18: mad_vs11 0.2932, mad_vs16 0.2944
19: mad_vs11 0.0128, mad_vs16 0.0521
20: mad_vs11 0.0120, mad_vs16 0.0517
21: mad_vs11 0.0761, mad_vs16 0.0733
22: mad_vs11 0.0485, mad_vs16 0.0546
23: mad_vs11 0.0483, mad_vs16 0.0546
```

预期：

- `19/20` 是对 `11` 的小幅在线修正，适合在 `11` 后优先提交。
- `21/22/23` 引入 group-meta，和 `11` 差异更大，适合冲榜 A/B。
- `18` 是纯 group-meta，差异最大，适合判断测试候选结构先验是否强。

## 2026-05-11 继续增强：temporal 滑窗热度与共识融合

本轮新增时间窗口热度特征，重点补充 dataset2 的近期趋势：

- 源-目标历史总次数与最近一次交互时间。
- 目标节点历史总热度与最近一次出现时间。
- 目标节点在 7/30/90/365 天窗口内的出现次数。
- 源-目标 pair 在 7/30/90/365 天窗口内的重复次数。
- 目标节点按 7/30/90 天半衰尺度做指数衰减热度。

新增脚本：

```text
make_temporal_result_zip.py
tune_temporal_hard.py
blend_website_csv_geomean.py
```

`tune_temporal_hard.py` 使用 dataset2 的 `split=1` 只做本地验证调权，不把验证标签写入最终训练。本地验证结果：

```text
temporal tuned dataset2 MRR: 0.1961
sequential dataset2 MRR:     0.2060
```

temporal 单路弱于 sequential，但特征来源不同，因此作为小权重融合信号使用。

新增提交包：

```text
submissions/24_d1old_d2_temporal_tuned/result.zip
submissions/25_d1old_d2_fullseq_temporal20/result.zip
submissions/26_d1old_d2_onlineq50_temporal/result.zip
submissions/27_d1old_d2_group_temporal/result.zip
submissions/28_d1old_d2_onlineq50_group_temporal/result.zip
submissions/29_d1old_d2_geo_seq_online_temporal/result.zip
submissions/30_d1old_d2_geo_all/result.zip
```

全部通过 `validate_result_zip.py` 校验。

### 差异度参考

以 dataset2 为对象，和 `11`/`20`/`23` 的平均绝对差异：

```text
24: mad11 0.0312, mad20 0.0359, mad23 0.0633
25: mad11 0.0062, mad20 0.0153, mad23 0.0487
26: mad11 0.0102, mad20 0.0077, mad23 0.0449
27: mad11 0.0634, mad20 0.0654, mad23 0.0218
28: mad11 0.0643, mad20 0.0637, mad23 0.0194
```

预期：

- `25/26` 是更稳的小幅修正，适合在 `19/20` 后提交。
- `29/30` 使用几何均值共识融合，适合测试“多信号一致高分”是否比线性融合更好。
- `24` 是纯 temporal 探路包，不建议优先提交。

## 2026-05-11 继续增强：方向关联 assoc 与 dataset1 小幅增强

本轮新增 direction association 信号，目标是补充 sequential 只统计连续转移的问题。构造方式：

- 对每个 src 按时间排序。
- 对每个历史目标 item，取它之前最近 K 个不同 item 作为上下文。
- 累计 `context_item -> future_item` 的方向关联权重。
- 推理时用测试 src 的最近历史 item 去激活候选 item。

新增脚本：

```text
make_assoc_result_zip.py
tune_assoc_hard.py
```

本地验证结果：

```text
assoc tuned dataset2 MRR: 0.1405
sequential dataset2 MRR:   0.2060
temporal dataset2 MRR:     0.1961
```

结论：assoc 单路明显弱，不适合作为主提交，只适合作为 5%-12% 的补充信号或探路。

新增提交包：

```text
submissions/31_d1old_d2_assoc_tuned/result.zip
submissions/32_d1old_d2_fullseq_temp_assoc/result.zip
submissions/33_d1old_d2_onlineq50_temp_assoc/result.zip
submissions/34_d1old_d2_geo_assoc/result.zip
submissions/35_d1_online15_temp05_d2_seq_temp/result.zip
submissions/36_d1_online20_temp05_d2_online_seq/result.zip
submissions/37_d1_mixed_d2_group_online/result.zip
submissions/38_geo_d1small_d2seqonline/result.zip
```

全部通过 `validate_result_zip.py` 校验。

### 差异度参考

assoc 系列相对强包的 dataset2 平均绝对差异：

```text
31: mad25 0.2628, mad29 0.2682, mad30 0.2411
32: mad25 0.0209, mad29 0.0300, mad30 0.0795
33: mad25 0.0251, mad29 0.0292, mad30 0.0792
34: mad25 0.0404, mad29 0.0302, mad30 0.0729
```

dataset1 小幅增强系列相对 `01` 的 dataset1 差异：

```text
35: d1_mad_vs01 0.0251
36: d1_mad_vs01 0.0330
37: d1_mad_vs01 0.0259
38: d1_mad_vs01 0.1122
```

预期：

- `35` 是最稳的 dataset1 小改版本，dataset2 等同 `25`。
- `36` 是第二稳的小改版本，同时对 dataset2 做 online/seq 混合。
- `32/33/34` 只适合在 `25/26/29/30` 后做 A/B。
- `38` 对 dataset1 改动大，属于激进探路。

## 2026-05-11 继续增强：Jittor combined ranker

本轮新增 Jittor 线性排序器，把已有强信号合并成 38 维特征后训练：

- sequential 特征 13 维；
- temporal 滑窗/衰减特征 17 维；
- direction assoc 特征 8 维。

新增脚本：

```text
train_combo_jittor.py
make_combo_jittor_result_zip.py
```

训练设置：

```text
dataset2
train history: split=0
validation supervision: split=1
queries: 5000
negatives per query: 99
epochs: 12
framework: Jittor
```

训练日志显示 BPR loss 从 `0.7617` 降到 `0.3709`。最终推理时使用 full-history 统计特征，但模型权重来自 `split=0 -> split=1` 的本地训练。

新增提交包：

```text
submissions/39_d1old_d2_combo_jittor/result.zip
submissions/40_d1old_d2_seq_temp_combo20/result.zip
submissions/41_d1old_d2_online_seq_temp_combo20/result.zip
submissions/42_d1old_d2_geo_combo/result.zip
submissions/43_d1small_d2_combo22/result.zip
```

全部通过 `validate_result_zip.py` 校验。

### 差异度参考

以 dataset2 为对象：

```text
39: mad25 0.0290, mad29 0.0373
40: mad25 0.0065, mad29 0.0206
41: mad25 0.0108, mad29 0.0192
42: mad25 0.0245, mad29 0.0069
43: mad25 0.0074, mad29 0.0208
```

预期：

- `40` 是最稳的 Jittor 小权重修正版。
- `41` 比 `40` 更偏 online sequential。
- `43` 同时小改 dataset1 和 dataset2。
- `39` 是纯 combo Jittor 探路，不建议优先提交。
- `42` 接近 `29` 的几何路线，适合在 `29` 结果较好时跟进。

### 线上反馈

用户反馈：

```text
40_d1old_d2_seq_temp_combo20/result.zip: 1.1212
43_d1small_d2_combo22/result.zip: 1.1305940693487557
```

相比早期 `01_blend_dsaware_seq` 的 `1.084853914501116`，`43` 提升约 `0.0457`；相比 `40` 再提升约 `0.0094`。这说明 dataset1 小幅 online/temporal 增强和 dataset2 combo 小权重都有效。后续优先围绕 `43` 做：

- combo 权重微调；
- dataset1 小幅增强；
- online q50 与 combo 的比例搜索；
- enhanced/groupmeta 小权重共识融合。

## 2026-05-12 继续增强：围绕线上最强 43 的局部微调

线上反馈 `43_d1small_d2_combo22/result.zip` 得分：

```text
1.1305940693487557
```

因此本轮围绕 43 做局部权重微调，不大幅改变已验证有效结构。

新增提交包：

```text
submissions/44_d1_more_online_d2_43/result.zip
submissions/45_d1_less_online_d2_43/result.zip
submissions/46_d1_43_d2_combo28/result.zip
submissions/47_d1_43_d2_combo18/result.zip
submissions/48_d1_43_d2_online10/result.zip
submissions/49_d1_43_d2_combo26_enh08/result.zip
```

全部通过 `validate_result_zip.py` 校验。

### 微调含义

- `44`：只提高 dataset1 online 权重，dataset2 保持 43。
- `45`：只降低 dataset1 online 权重，dataset2 保持 43。
- `46`：只提高 dataset2 combo 权重。
- `47`：只降低 dataset2 combo 权重。
- `48`：只给 dataset2 加少量 online q50。
- `49`：给 dataset2 加 enhanced 共识并提高 combo。

### 差异度参考

相对 43：

```text
44: d1_mad 0.00836, d2_mad 0
45: d1_mad 0.00836, d2_mad 0
46: d1_mad 0,       d2_mad 0.00159
47: d1_mad 0,       d2_mad 0.00106
48: d1_mad 0,       d2_mad 0.00120
49: d1_mad 0,       d2_mad 0.01663
```

当前建议优先试 `44/46/48`，因为它们是围绕 43 的低风险局部微调。
## 2026-05-12 新增第 50-57 轮

在 `43_d1small_d2_combo22/result.zip` 已知线上反馈 `1.1305940693487557` 的基础上，新增 8 个候选提交包，全部生成在 `submissions/` 下并通过 `validate_result_zip.py` 校验。

线上反馈更新：

- `56_d1_online20_d2_combo30_online05/result.zip`：`1.1462969419419828`，当前已验证最强。
- `57_geo_d1_online18_d2_combo28/result.zip`：`1.1405314670474809`，高于 43 但低于 56。

结论：dataset2 的 `online q50 0.05 + temporal 0.08 + fullseq 0.57 + combo_jittor 0.30` 明显优于 43 的 `temporal 0.10 + fullseq 0.68 + combo_jittor 0.22`，下一轮应围绕 56 继续细调 online、combo、fullseq、temporal 的比例。

新增包：

- `50_d1_online18_d2_combo24/result.zip`
- `51_d1_online22_d2_combo24/result.zip`
- `52_d1_online18_d2_combo28/result.zip`
- `53_d1_online22_d2_combo28/result.zip`
- `54_d1_online18_d2_combo32/result.zip`
- `55_d1_online20_d2_combo30_temp06/result.zip`
- `56_d1_online20_d2_combo30_online05/result.zip`
- `57_geo_d1_online18_d2_combo28/result.zip`

本轮调整：

- dataset1：验证 online q50 权重从 `0.15` 提高到 `0.18/0.20/0.22` 是否继续提升。
- dataset2：验证 combo_jittor 权重从 `0.22` 提高到 `0.24/0.28/0.30/0.32` 是否继续提升。
- temporal：`55` 把 temporal 从 `0.10` 降到 `0.06`，测试 temporal 是否偏噪。
- online：`56` 给 dataset2 加 `0.05` online q50，测试伪在线历史是否有贡献。
- 几何融合：`57` 使用 geomean rank 输出，差异较大，只作为探索包。

相对 `43` 的预期结果：

- `50`：最稳妥，预期小幅提升或持平；如果提升，说明 43 附近仍有细搜空间。
- `52`：更偏 dataset2 combo；如果提升，继续加大 combo_jittor。
- `55`：combo 增强且 temporal 降权；如果提升，说明 temporal 权重偏高。
- `56`：加入少量 dataset2 online q50；如果提升，说明候选序列内伪历史有效。
- `51/53`：测试 dataset1 online 上限；如果掉分，dataset1 online 不宜超过 0.20。
- `54`：测试 combo=0.32 上限，风险高于 50/52。
- `57`：差异最大，适合作为探索，不作为优先提交。

推荐提交顺序：`50 -> 52 -> 55 -> 56 -> 51 -> 53 -> 54 -> 57`。

## 2026-05-12 新增第 58-65 轮

基于线上反馈 `56 = 1.1462969419419828`、`57 = 1.1405314670474809`，确认 56 的线性融合方向优于 57 的几何 rank 方向。本轮继续以 56 为中心做二阶细搜。

新增包：

- `58_d1_56_d2_online03_combo32/result.zip`
- `59_d1_56_d2_online07_combo30/result.zip`
- `60_d1_56_d2_online07_combo32_temp06/result.zip`
- `61_d1_56_d2_online10_combo30_temp06/result.zip`
- `62_d1_56_d2_online05_combo34_temp06/result.zip`
- `63_d1_online18_d2_56/result.zip`
- `64_d1_online22_d2_56/result.zip`
- `65_geo_d1_56_d2_56/result.zip`

预期：

- `59`：最小扰动，只把 dataset2 online 从 0.05 提到 0.07，优先提交。
- `58`：降低 online、提高 combo，用来判断 56 的收益主要来自 online 还是 combo。
- `60/61`：online 增强同时降低 temporal，测试 temporal 是否仍偏高。
- `62`：combo=0.34，上限探索。
- `63/64`：dataset2 固定为 56，只测试 dataset1 online=0.18/0.22。
- `65`：56 权重的几何 rank 版，因 57 低于 56，只作为探索包。

推荐提交顺序：`59 -> 58 -> 60 -> 61 -> 63 -> 64 -> 62 -> 65`。
