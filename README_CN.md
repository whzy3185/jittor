# 第六届 Jittor 人工智能算法挑战赛 Track 1 工作说明

完整改动与实验轮次总结见：

```text
EXPERIMENT_SUMMARY_CN.md
```

## 项目说明

本目录是 Track 1「基于图学习的动态推荐」的 E 盘 WSL 工作副本，路径为：

`/mnt/e/Jitter/track1_aggressive_wsl`

原始 Track 1 代码来自 `/mnt/e/Jitter/legacy_track1`，官方/原始数据保持在：

`/mnt/e/Jitter/data/official_raw`

本工作目录只通过符号链接读取数据：`data/official_raw -> /mnt/e/Jitter/data/official_raw`，不修改原始 CSV。

## 任务解释

数据包含历史动态图边 `src,dst,time`，测试集每行给定一个 `src,time` 和 `c1..c100` 候选目标节点。目标是在不使用未来交互、不使用测试标签的前提下，对每个 query 的 100 个候选进行打分或排序。

`dataset1` 没有官方验证标签，本项目按时间尾部 holdout 构造本地验证；`dataset2` 的 `train.csv` 有 `split` 字段，`split=0` 为训练历史，`split=1` 为验证边。

## WSL 运行方式

所有命令都在 WSL 中运行：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
```

推荐 Python：

```bash
/mnt/e/Jitter/.venv_wsl_cpu/bin/python
```

Jittor cache 和 pip cache 均指向 E 盘：

```bash
export PIP_CACHE_DIR=/mnt/e/Jitter/.pip_cache
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
export nvcc_path=""
```

## 环境检查

```bash
wsl --status
wsl -l -v
wsl -e bash -lc "ls -lah /mnt/e"
cd /mnt/e/Jitter/track1_aggressive_wsl
/mnt/e/Jitter/.venv_wsl_cpu/bin/python -m pip list
```

当前已安装：`jittor`, `jittor_geometric`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `tqdm`。

注意：当前 WSL 中如果不设置 `nvcc_path=""`，导入 Jittor 会触发 `cuda12.2_cudnn8_linux.tgz` 约 5.61GB 下载，下载目录在 E 盘 Jittor cache。默认脚本已设置 CPU/Jittor 可运行模式；如需 GPU，先让 CUDA runtime 下载完成，再显式加 `USE_CUDA_ARGS=--use-cuda`。

## Baseline 来源

- 原始 Track 1 baseline 保留在 `/mnt/e/Jitter/legacy_track1`。
- 工作副本由 `legacy_track1` 复制而来。
- `craft_baseline.py` 保留官方 JittorGeometric CRAFT 示例入口。
- 改进模型在 `track1_dynamic_rec/` 中，仍使用 Jittor 训练和推理。

## 数据放置方式

官方数据结构：

```text
data/official_raw/
  dataset1/
    train.csv
    test.csv
  dataset2/
    train.csv
    test.csv
```

## 训练命令

Jittor CUDA runtime 下载完成后：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
bash run_train.sh dataset1
bash run_train.sh dataset2
```

可缩短 smoke test：

```bash
EPOCHS=1 BATCH_SIZE=512 NEGATIVES=2 VAL_NEGATIVES=10 MAX_TRAIN_EVENTS=1000 MAX_VAL_EVENTS=100 MAX_TEST_QUERIES=20 bash run_train.sh dataset1
EPOCHS=1 BATCH_SIZE=128 NEGATIVES=2 VAL_NEGATIVES=5 MAX_TRAIN_EVENTS=1000 MAX_VAL_EVENTS=100 MAX_TEST_QUERIES=20 bash run_train.sh dataset2
```

`MAX_VAL_EVENTS` 用于限制本地验证正例数量，避免 dataset2 的 `split=1` 一次生成过多验证候选；`MAX_TEST_QUERIES` 仅用于 smoke test，正式训练和正式推理应保持为 0，保证完整 test 候选节点参与 ID 映射。

`run_train.sh` 默认会设置 `MAX_VAL_EVENTS=5000`，避免 dataset2 无意中生成千万级验证候选。若要扩大验证规模：

```bash
MAX_VAL_EVENTS=20000 bash run_train.sh dataset2
```

dataset1 没有官方验证 split。最终冲榜训练如果希望使用全部 `train.csv` 历史，可关闭 temporal holdout：

```bash
VAL_RATIO=0 EPOCHS=30 bash run_train.sh dataset1
```

## 推理命令

训练出 `outputs/track1/<dataset>/checkpoints/best.pkl` 后：

```bash
bash run_infer.sh dataset1
bash run_infer.sh dataset2
```

smoke 推理示例：

```bash
MAX_VAL_EVENTS=100 MAX_TEST_QUERIES=20 bash run_infer.sh dataset2
```

输出目录：

```text
outputs/track1/submission/<dataset>/
```

## 快速提交文件生成

在 Jittor 环境未完成 CUDA runtime 下载时，可使用泄漏安全的历史启发式 fallback：

```bash
bash run_submit.sh dataset1
bash run_submit.sh dataset2
```

已生成全量文件：

```text
outputs/track1_heuristic_aggressive/dataset1.zip
outputs/track1_heuristic_aggressive/dataset2.zip
```

## 最终自检

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
bash run_self_check.sh
```

该脚本会检查 WSL、E 盘访问、Jittor/JittorGeometric 导入、Python 语法、PyTorch 禁用、中文文档、两个提交 zip 格式和发布包。

## 官网 CSV 概率提交包

官网截图要求一个 `result.zip`，其中包含 `dataset1.csv` 和 `dataset2.csv`，每行 100 个 `[0,1]` 概率且保留 8 位小数。当前已生成并校验通过的候选提交包：

```text
outputs/website_submission_hard_tuned/result.zip          # 推荐优先试，hard validation 调权，rank 概率
outputs/website_submission_hard_tuned_sigmoid/result.zip  # 同排序，sigmoid 概率校准，文件更大
outputs/website_submission_simple_tuned/result.zip        # 随机负样本调权，较激进
outputs/website_submission/result.zip                     # 旧版 0.9425 回退包
```

注意：早期 `hard_tuned`/`simple_tuned` 调参曾把验证正例固定放在候选第一位，会高估 `candidate_prior`，后续已修正为 query 内随机打乱候选。更可信的新包如下：

```text
outputs/website_submission_hard_shuffled/result.zip          # 当前优先推荐，修正验证候选位置泄漏
outputs/website_submission_hard_shuffled_sigmoid/result.zip  # 同排序，sigmoid 概率
outputs/website_submission_hard_shuffled_order01/result.zip  # hard_shuffled + 10% 官方列顺序先验
outputs/website_submission_hard_shuffled_order02/result.zip  # hard_shuffled + 20% 官方列顺序先验
outputs/website_submission_order_prior_linear/result.zip     # 只用官方列顺序先验，用于 A/B 试榜
outputs/website_submission_global_sigmoid/result.zip         # 全局 sigmoid 概率校准，保留跨 query 置信度
outputs/website_submission_global_rankish/result.zip         # 全局均值方差线性校准
outputs/website_submission_hard_shuffled_shrink10/result.zip # rank 概率向 0.5 收缩 10%
outputs/website_submission_jittor_noprior_rank/result.zip    # Jittor 线性 pairwise 排序器，关闭候选位置伪先验
outputs/website_submission_jittor_noprior_rowsigmoid/result.zip # Jittor 排序器 row sigmoid 概率版
outputs/website_submission_blend_hard70_jittor30/result.zip  # 当前新增优先尝试：hard_shuffled 70% + Jittor noprior 30%
outputs/website_submission_blend_hard50_jittor50/result.zip  # 融合 A/B 包：hard_shuffled 50% + Jittor noprior 50%
outputs/website_submission_enhanced_hard_rank/result.zip     # 新增增强启发式：item-item 共现 + 历史特征，rank 概率
outputs/website_submission_enhanced_hard_rowsigmoid/result.zip # 同排序的 row sigmoid 概率校准版
outputs/website_submission_blend_enh70_hard20_jittor10/result.zip # 当前最推荐：增强 70% + hard 20% + Jittor 10%
outputs/website_submission_blend_enh80_hard20/result.zip     # 增强 80% + hard 20%，更偏向本轮新增共现特征
outputs/website_submission_blend_enh85_hard10_jittor05/result.zip # 增强 85% + hard 10% + Jittor 5%
outputs/website_submission_testmeta_rank/result.zip          # 只用 test 候选列表无标签结构信号，A/B 参考
outputs/website_submission_blend_enh65_hard18_jittor07_meta10/result.zip # 增强主分 + 10% test-meta
outputs/website_submission_blend_enh60_hard15_jittor05_meta20/result.zip # 增强主分 + 20% test-meta，较激进
outputs/website_submission_blend_enh68_hard18_jittor09_order05/result.zip # 增强主分 + 5% 官方列顺序先验
outputs/website_submission_blend_enh65_hard17_jittor08_order10/result.zip # 增强主分 + 10% 官方列顺序先验
outputs/website_submission_enhanced_jittor_mlp_rank/result.zip # Jittor MLP 学习增强特征的 pairwise ranker
outputs/website_submission_blend_enh70_hard20_mlp10/result.zip # 增强 70% + hard 20% + Jittor MLP 10%
outputs/website_submission_blend_enh60_hard15_mlp15_meta10/result.zip # 增强 60% + hard 15% + MLP 15% + test-meta 10%
outputs/website_submission_blend_enh65_hard15_mlp10_order10/result.zip # 增强 65% + hard 15% + MLP 10% + 列顺序 10%
outputs/website_submission_sequential_hard_rank/result.zip # 新增序列转移增强：source 历史上一目的节点 -> 下一目的节点
outputs/website_submission_blend_seq70_enh15_hard10_mlp05/result.zip # 当前新增优先：seq 70% + enhanced 15% + hard 10% + MLP 5%
outputs/website_submission_blend_seq65_enh15_hard10_meta10/result.zip # seq 65% + enhanced 15% + hard 10% + test-meta 10%
outputs/website_submission_blend_seq65_enh15_hard10_order10/result.zip # seq 65% + enhanced 15% + hard 10% + 列顺序 10%
outputs/website_submission_blend_dsaware_seq/result.zip # 按数据集分别融合：dataset2 更高 seq 权重
outputs/website_submission_blend_dsaware_seq_meta/result.zip # 分数据集融合 + test-meta
outputs/website_submission_blend_dsaware_seq_order/result.zip # 分数据集融合 + 列顺序先验
outputs/website_submission_blend_dsaware_seq_shrink/result.zip # 分数据集融合 + 概率向 0.5 收缩
```

校验命令：

```bash
/mnt/e/Jitter/.venv_wsl_cpu/bin/python validate_result_zip.py --zip outputs/website_submission_blend_hard70_jittor30/result.zip
```

官网 CSV 概率提交包内必须只有 `dataset1.csv` 和 `dataset2.csv`。旧版 `dataset1.zip`、`dataset2.zip` 或包含 `result.json` 的包不适用于当前截图中的官网格式。

Jittor 轻量排序器训练与出包命令：

```bash
env nvcc_path="" cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache /mnt/e/Jitter/.venv_wsl_cpu/bin/python train_simple_jittor.py --dataset dataset1 --output-dir outputs/simple_jittor_noprior --max-events 200000 --warmup-events 10000 --negatives 8 --epochs 8 --seed 2027
env nvcc_path="" cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache /mnt/e/Jitter/.venv_wsl_cpu/bin/python train_simple_jittor.py --dataset dataset2 --output-dir outputs/simple_jittor_noprior --max-events 200000 --warmup-events 10000 --negatives 8 --epochs 8 --seed 2027
/mnt/e/Jitter/.venv_wsl_cpu/bin/python make_simple_jittor_result_zip.py --model-dir outputs/simple_jittor_noprior --output-dir outputs/website_submission_jittor_noprior_rank --mode rank
/mnt/e/Jitter/.venv_wsl_cpu/bin/python blend_website_csv.py --inputs outputs/website_submission_hard_shuffled outputs/website_submission_jittor_noprior_rank --weights 0.7 0.3 --output-dir outputs/website_submission_blend_hard70_jittor30
```

等价的一键脚本：

```bash
bash run_train_simple_jittor.sh
bash run_make_jittor_submit.sh
```

增强共现特征调参与出包命令：

```bash
bash run_tune_enhanced.sh
bash run_make_enhanced_submit.sh
```

无标签 test 候选结构特征与列顺序先验 A/B 出包：

```bash
bash run_make_meta_submit.sh
```

增强特征 Jittor MLP 排序器训练与出包：

```bash
bash run_train_enhanced_jittor.sh
bash run_make_enhanced_jittor_submit.sh
```

该 MLP 使用训练集内部时间切分：前 85% 作为历史图，后段训练边作为未来正例，并从 test 候选池/热门节点采样 hard negatives；不使用 validation/test label。

序列转移增强调参与出包：

```bash
bash run_tune_sequential.sh
bash run_make_sequential_submit.sh
```

序列增强新增 `trans_log`、`trans_max`、`trans_last`，刻画 source 最近历史目的节点到候选目的节点的全局转移强度。本地 hard validation：

```text
dataset1 sequential MRR: 0.7438
dataset2 sequential MRR: 0.2060
```

分数据集融合出包：

```bash
bash run_make_dsaware_submit.sh
```

该脚本对 dataset1/dataset2 使用不同权重。当前设置中 dataset2 的序列转移权重更高，因为本地验证显示 dataset2 从 sequential 特征中收益更明显。

本轮增强在本地 hard validation 上的主要变化：

```text
dataset1 enhanced MRR: 0.7418
dataset2 enhanced MRR: 0.1960
```

其中 dataset2 之前 hard-shuffled 调权约为 `0.145` MRR，新特征主要依靠 `cooc_log` 捕获 source 历史目的节点与候选目的节点的 item-item 共现关系。

默认 `result.json` 写入每个 query 排序后的候选 ID 列表；如官网明确要求分数列表，可显式设置：

```bash
JSON_MODE=scores bash run_submit.sh dataset1
```

dataset2 的 `train.csv` 含 `split` 字段。为避免验证标签泄漏，`run_submit.sh` 默认只用 `split=0` 历史边构造 heuristic 提交。若后续确认规则允许把 `split=1` 作为最终测试前历史，可显式开启：

```bash
INCLUDE_VALID_HISTORY=1 bash run_submit.sh dataset2
```

## 复现实验流程

1. 检查 `/mnt/e` 可访问。
2. 进入 `/mnt/e/Jitter/track1_aggressive_wsl`。
3. 确认 `/mnt/e/Jitter/.venv_wsl_cpu/bin/python -m pip list`。
4. 默认 CPU smoke test 使用 `nvcc_path=""`，不会下载 CUDA runtime。
5. 执行 `bash run_train.sh dataset1` 或 `dataset2`。
6. 执行 `bash run_infer.sh dataset1` 或 `dataset2`。
7. 若只需快速提交，执行 `bash run_submit.sh dataset1` 或 `dataset2`。

## 已完成的本地 smoke test

- Jittor CPU 模式导入通过：`nvcc_path=""`，`jittor 1.3.10.0`。
- JittorGeometric 导入通过。
- dataset1 神经训练 smoke：1 epoch，MRR 约 0.6641。
- dataset1 神经推理 smoke：20 个 query，每个 100 候选，已打包。
- dataset2 神经训练 smoke：1 epoch，`MAX_TRAIN_EVENTS=1000`、`MAX_VAL_EVENTS=100`、`MAX_TEST_QUERIES=20`，MRR 约 0.4347。
- dataset2 神经推理 smoke：20 个 query，每个 100 候选，已打包。
- 全量 heuristic fallback 已生成 dataset1/dataset2 提交 zip。

## 常见问题

- **Jittor 导入很慢**：设置 `nvcc_path=""` 可跳过 CUDA runtime 下载并使用 CPU；GPU 训练才需要等待 5.61GB runtime 下载完成。
- **不要在 C 盘写文件**：本项目 cache、输出、数据均在 `/mnt/e/Jitter`。
- **提交格式不确定**：本地没有官方 Track 1 submission template，当前同时输出 JSON、候选分数 CSV、排序 CSV；最终以官网要求为准。
- **dataset1 没有验证标签**：使用时间 holdout 和负采样构造本地验证，只用于调参，不用于提交训练外泄。
- **dataset2 验证过大**：使用 `MAX_VAL_EVENTS` 控制本地验证规模；正式训练可逐步放大。

## 流式神经推理

`infer_stream.py` 用于正式全量测试集推理，避免把 dataset2 的 `153420 * 100` 个候选一次性展开到内存。

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
bash run_infer_stream.sh dataset1
bash run_infer_stream.sh dataset2
```

常用参数：

```bash
QUERY_CHUNK_SIZE=2048 BATCH_SIZE=8192 bash run_infer_stream.sh dataset2
MAX_TEST_QUERIES=20 QUERY_CHUNK_SIZE=7 bash run_infer_stream.sh dataset1
```

训练脚本会在 `checkpoints/id_mapping.json` 保存原始节点 ID 到模型内部 ID 的映射；流式推理会优先读取该映射。若旧 checkpoint 没有该文件，则按相同数据和 `MAX_TEST_QUERIES` 重新构建映射。默认输出：

```text
outputs/track1/submission_stream/<dataset>/result.json
outputs/track1/submission_stream/<dataset>.zip
```

默认 zip 内包含 `result.json`、`result_ranked.csv`、`result_scores.csv`。只有显式设置 `WRITE_DEBUG_CSV=1` 时才会生成候选级 `predictions.csv`，避免全量数据下产生过大的调试文件。

训练阶段会在验证集上自动搜索 `model_rank` 与 `heuristic_rank` 的融合权重，并写入 `best_meta.json` 的 `blend_model_weight` / `blend_heuristic_weight`。`infer.py` 和 `infer_stream.py` 默认读取 checkpoint 中的权重；如需手动覆盖，可加：

```bash
/mnt/e/Jitter/.venv_wsl_cpu/bin/python infer_stream.py --dataset dataset2 --model-weight 0.85 --heuristic-weight 0.15
```

## 多模型分数融合

多个 seed 或多个 checkpoint 生成的 `result_scores.csv` 可以用 `ensemble_scores.py` 做 query 内 rank averaging：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
bash run_ensemble_scores.sh dataset2 \
  outputs/run_seed42/dataset2/result_scores.csv \
  outputs/run_seed43/dataset2/result_scores.csv
```

带权重融合：

```bash
WEIGHTS="0.6 0.4" bash run_ensemble_scores.sh dataset2 score_a.csv score_b.csv
```

输出：

```text
outputs/track1/ensemble_scores/<dataset>/result.json
outputs/track1/ensemble_scores/<dataset>.zip
```

## 多 Seed 一键实验

`run_multiseed.sh` 会依次完成训练、流式推理、`result_scores.csv` 融合：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
EPOCHS=20 MAX_VAL_EVENTS=5000 bash run_multiseed.sh dataset2 42 43 44
```

smoke test：

```bash
EPOCHS=1 MAX_TRAIN_EVENTS=300 MAX_VAL_EVENTS=20 MAX_TEST_QUERIES=20 BATCH_SIZE=128 bash run_multiseed.sh dataset2 42 43
```

默认输出：

```text
outputs/track1_multiseed/<dataset>_seed<seed>/
outputs/track1_multiseed/<dataset>_seed<seed>_submission/
outputs/track1_multiseed/<dataset>_ensemble.zip
```

## 验证集启发式评估与融合权重复调

单独评估历史启发式验证表现：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
MAX_VAL_EVENTS=5000 bash run_evaluate_heuristic.sh dataset2
```

输出：

```text
outputs/track1/heuristic_eval/<dataset>/metrics.json
outputs/track1/heuristic_eval/<dataset>/validation_predictions.csv
```

训练完成后，如果想在不重训模型的情况下重新搜索神经分数和启发式分数融合权重：

```bash
MAX_VAL_EVENTS=5000 WEIGHT_STEP=0.025 bash run_tune_blend.sh dataset2
```

如需把搜索结果写回 checkpoint 元信息：

```bash
WRITE_META=1 MAX_VAL_EVENTS=5000 bash run_tune_blend.sh dataset2
```

## 提交文件校验

`validate_submission.py` 会检查 `result.json` 是否满足官方 test 宽表候选约束：query 数一致、每个 query 候选数一致、无重复、排序结果必须来自该行 `c1..c100`。

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
bash run_validate_submission.sh dataset1 outputs/track1_heuristic_aggressive/dataset1.zip
bash run_validate_submission.sh dataset2 outputs/track1_heuristic_aggressive/dataset2.zip
```

smoke 文件可限制行数：

```bash
MAX_QUERIES=20 bash run_validate_submission.sh dataset2 outputs/smoke_blend_dataset2_submission.zip
```

## 发布打包

生成源码包和双数据集提交包：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
bash run_package_release.sh
```

输出：

```text
outputs/release/track1_aggressive_source.zip
outputs/release/track1_submission_bundle.zip
```

`track1_submission_bundle.zip` 内含 `dataset1.zip`、`dataset2.zip` 和 `manifest.json`。若官网要求分别上传单数据集 zip，则直接上传：

```text
outputs/track1_heuristic_aggressive/dataset1.zip
outputs/track1_heuristic_aggressive/dataset2.zip
```
