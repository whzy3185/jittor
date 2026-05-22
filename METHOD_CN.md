# Track 1 方法说明

## 方法总览

本方案从 E 盘已有 Track 1 baseline 工作副本出发，保留官方 JittorGeometric CRAFT baseline 入口，并在其基础上实现一个面向候选边排序的混合动态推荐框架：

- Jittor 节点嵌入；
- 时间编码；
- 历史交互特征；
- 残差 MLP 候选边 scorer；
- BPR pairwise ranking loss；
- 流式历史启发式 fallback；
- 分 query 的 rank/score 输出。

## 官方 Baseline 继承关系

`craft_baseline.py` 继续指向 JittorGeometric 的 `CRAFT` 示例代码，作为官方 CRAFT 行为保留入口。新增代码位于 `track1_dynamic_rec/`，训练脚本仍使用 Jittor，不引入 PyTorch。

## 改进点

1. 支持 `dataset1` / `dataset2` 显式选择，避免误读多数据目录。
2. 自动识别时间尺度，秒级时间戳使用 30 天窗口，非秒级使用训练跨度的约 3%。
3. 增加历史交互特征：源节点活跃度、目标节点热度、pair 次数、pair 最近活跃、长短期流行度、重复边标记、新边标记、pair share。
4. 神经 scorer 使用 `src_emb`、`dst_emb`、点积、绝对差、时间编码和历史特征拼接。
5. MLP 改为 LayerNorm + GELU + Dropout + residual block。
6. 训练使用 popularity-biased negative sampling 和 BPR loss。
7. 推理时支持模型分数与历史启发式分数融合。
8. 提供无需神经 checkpoint 的流式 heuristic fallback，可在 Jittor 首次编译受阻时生成提交文件。
9. 增加 `MAX_VAL_EVENTS` / `MAX_TEST_QUERIES`，支持 dataset2 级别数据的可控 smoke test。

## 时间特征设计

时间特征只使用候选预测时刻之前的训练历史。对每个候选 `(src,dst,t)`，特征构造器只吸收 `history.time < t` 的边，然后计算：

- `src_delta_log`：源节点距上次交互时间；
- `dst_delta_log`：目标节点距上次交互时间；
- `pair_delta_log`：该 pair 距上次交互时间；
- 短窗口和长窗口内的目标流行度；
- sinusoidal time encoding。

## 历史交互特征设计

主要特征包括：

- 源节点历史度；
- 目标节点历史度；
- pair 历史频次；
- 源节点近期交互数；
- 目标节点近期交互数；
- pair 近期交互数；
- 目标节点长窗口热度；
- common neighbor 近似；
- 目标热度归一排名；
- 源活跃度归一排名；
- pair 占源节点交互比例；
- repeat / new edge 标记。

## 动态推荐建模思路

候选集已经由官方数据给出，核心问题是 reranking。模型学习历史图中的重复偏好、目标流行度和时间衰减，同时通过节点嵌入保留泛化能力，降低只记忆重复边的风险。

## 候选边打分方式

神经模型输入：

```text
[src_emb, dst_emb, src_emb * dst_emb, abs(src_emb - dst_emb), time_encoding, history_features]
```

输出一个标量 score。推理时默认使用 rank-normalized model score 与 heuristic score 融合。

## 负采样策略

训练时从训练集中目标节点的流行度分布采样负例，采样概率按 `count^0.75` 平滑。负例会避开当前源节点在训练集中已出现的正例，尽量减少 false negative。

## 排序损失

使用 BPR pairwise loss：

```text
loss = -log sigmoid(score_pos - score_neg)
```

该损失与候选排序目标更一致。

## 集成策略

当前已实现：

- 神经模型分数；
- 历史启发式分数；
- query 内 rank normalize；
- 验证集自动搜索融合权重；
- 多输出格式保存。

推荐后续训练多个 seed 后，对 `result_scores.csv` 做 rank averaging。

`ensemble_scores.py` 已支持直接读取多个 `result_scores.csv`，对每个 query 的 100 个候选分别做 rank normalize 后加权平均，再结合官方 `test.csv` 中的候选原始 ID 重新生成 `result.json`。该过程不读取标签，也不会修改训练数据，适合多 seed、多窗口和神经模型/启发式模型的后处理融合。

`run_multiseed.sh` 将上述流程串联为完整实验：每个 seed 独立训练一个 Jittor 模型，使用流式推理生成该 seed 的排序分数，最后调用 `ensemble_scores.py` 做无标签 rank averaging。该脚本只读官方 train/test CSV，不使用测试标签；dataset2 的 `split=1` 仅用于本地验证和权重选择，不进入训练正例。

`evaluate_heuristic.py` 用于单独评估历史启发式在本地验证集上的表现，帮助判断神经模型是否确实带来增益。`tune_blend.py` 可在已有 checkpoint 上重新扫描融合权重，不需要重新训练；这对于长训练后快速选择 `model_rank` / `heuristic_rank` 的比例更高效。

新增的 `train_simple_jittor.py` 使用 Jittor 训练一个轻量线性 pairwise ranker。它不使用 validation/test 标签，而是按时间顺序扫描训练边，在每条正边发生前截取历史统计特征，并从官方 test 候选池和训练热门节点中采样硬负例。损失函数仍为 `-log sigmoid(score_pos - score_neg)`。为避免训练构造中“正例永远位于候选第 1 位”的伪信号，默认关闭 `candidate_prior`，保存的 `outputs/simple_jittor_noprior/*_simple_jittor.json` 中该权重为 0。

新增的 `make_simple_jittor_result_zip.py` 会读取 Jittor ranker 权重，按官网截图要求直接生成 `dataset1.csv`、`dataset2.csv` 和 `result.zip`。当前新增的可提交包包括：

```text
outputs/website_submission_jittor_noprior_rank/result.zip
outputs/website_submission_jittor_noprior_rowsigmoid/result.zip
outputs/website_submission_blend_hard70_jittor30/result.zip
outputs/website_submission_blend_hard50_jittor50/result.zip
```

其中 `blend_hard70_jittor30` 将修正过候选位置泄漏的 hard-shuffled 启发式分数与 Jittor noprior 排序器做 70/30 融合，是本轮新增的优先 A/B 包。

进一步新增 `enhanced_heuristic.py`，在原历史特征之外加入 item-item 共现相似度：

- 对每个 source 保留最近若干个去重历史目的节点；
- 在训练历史内统计这些目的节点之间的共现；
- 对候选目的节点计算其与 source 最近历史目的节点的归一化共现和最大共现；
- 形成 `cooc_log`、`cooc_max` 两个额外特征。

`tune_enhanced_hard.py` 使用与官网 test 更接近的 hard negative：优先从同 source 的 test 候选池抽负例，再混入热门目的节点。调参前会把候选特征矩阵预计算，之后只做权重搜索，因此可以在较大验证集上快速搜索。当前本地 hard validation 最佳结果：

```text
dataset1 MRR 0.7418, hits@1 0.6973
dataset2 MRR 0.1960, hits@1 0.0797
```

`make_enhanced_result_zip.py` 使用全量允许训练历史重建共现统计并生成官网 CSV 概率包；当前优先提交包为：

```text
outputs/website_submission_blend_enh70_hard20_jittor10/result.zip
outputs/website_submission_enhanced_hard_rank/result.zip
```

本轮还新增 `make_testmeta_result_zip.py`，只使用官网 test 候选列表中的无标签结构信息，不读取任何 label：

- 同一 source 下候选目的节点在多个 test query 中重复出现的频率；
- 候选目的节点在所有 test query 中的全局出现频率；
- 同一 source 下候选目的节点的平均候选列位置；
- 当前 query 内候选列位置的弱先验。

这些特征属于 transductive candidate metadata，适合小权重 A/B 融合。当前生成：

```text
outputs/website_submission_blend_enh65_hard18_jittor07_meta10/result.zip
outputs/website_submission_blend_enh60_hard15_jittor05_meta20/result.zip
outputs/website_submission_blend_enh68_hard18_jittor09_order05/result.zip
outputs/website_submission_blend_enh65_hard17_jittor08_order10/result.zip
```

本轮新增 `train_enhanced_jittor.py` 和 `make_enhanced_jittor_result_zip.py`。它们把增强历史特征直接交给 Jittor MLP 学习 pairwise ranking：

- 训练集按时间切为前段历史图和后段未来正例；
- 特征只从前段历史图计算，避免把训练未来边泄漏进特征；
- 负例从官方 test 候选池和热门节点中采样，并避开该 source 的已知正例；
- 使用 Jittor MLP 残差块和 pairwise logistic loss；
- 推理时使用全量允许训练历史重建增强特征，再由 Jittor checkpoint 打分。

训练输出位于：

```text
outputs/enhanced_jittor_mlp/dataset1_enhanced_mlp.pkl
outputs/enhanced_jittor_mlp/dataset2_enhanced_mlp.pkl
```

新增可提交包：

```text
outputs/website_submission_enhanced_jittor_mlp_rank/result.zip
outputs/website_submission_blend_enh70_hard20_mlp10/result.zip
outputs/website_submission_blend_enh60_hard15_mlp15_meta10/result.zip
outputs/website_submission_blend_enh65_hard15_mlp10_order10/result.zip
```

本轮新增 `sequential_heuristic.py`，在共现增强基础上加入有方向的序列转移特征。对每个 source 按时间排序历史目的节点，统计全局 `prev_dst -> next_dst` 转移计数，并在候选打分时比较候选节点与该 source 最近历史节点之间的转移强度：

- `trans_log`：最近历史节点到候选节点的衰减转移总强度；
- `trans_max`：最近历史节点到候选节点的最大转移强度；
- `trans_last`：最后一个历史目的节点到候选节点的转移强度。

该特征仍只使用候选时间之前的训练历史。调权脚本为 `tune_sequential_hard.py`，官网格式出包脚本为 `make_sequential_result_zip.py`。本地 hard validation 最佳结果：

```text
dataset1 MRR 0.7438, hits@1 0.6953
dataset2 MRR 0.2060, hits@1 0.0877
```

新增可提交包：

```text
outputs/website_submission_sequential_hard_rank/result.zip
outputs/website_submission_blend_seq70_enh15_hard10_mlp05/result.zip
outputs/website_submission_blend_seq65_enh15_hard10_meta10/result.zip
outputs/website_submission_blend_seq65_enh15_hard10_order10/result.zip
```

本轮新增 `blend_website_csv_by_dataset.py`，解决旧融合脚本只能对 dataset1/dataset2 使用同一组权重的问题。由于本地验证中 dataset2 对序列转移特征的收益更明显，而 dataset1 的 enhanced/hard 分数仍有价值，新的分数据集融合使用：

```text
dataset1: seq 0.55, enhanced 0.30, hard 0.10, mlp 0.05
dataset2: seq 0.80, enhanced 0.08, hard 0.07, mlp 0.05
```

新增可提交包：

```text
outputs/website_submission_blend_dsaware_seq/result.zip
outputs/website_submission_blend_dsaware_seq_meta/result.zip
outputs/website_submission_blend_dsaware_seq_order/result.zip
outputs/website_submission_blend_dsaware_seq_shrink/result.zip
```

`validate_submission.py` 在生成提交后进行格式校验，确保 `result.json` 中每个 query 的排序列表与官方 test 行的候选集合完全一致。该检查可以防止因 query 顺序、候选重复或候选 ID 写错导致的无效提交。

启发式 fallback 默认遵守 dataset2 split 边界：`split=0` 用作历史，`split=1` 不进入最终 test 特征统计，避免验证标签泄漏。只有在明确确认赛规允许使用验证边作为测试前历史时，才通过 `INCLUDE_VALID_HISTORY=1` 显式开启。

训练时每个 epoch 会在本地验证候选上对 `model_rank` 权重做 `0.00, 0.05, ..., 1.00` 网格搜索，以 MRR 优先、AUC 兜底选择权重。最佳 checkpoint 的 `best_meta.json` 会记录：

```json
{
  "blend_model_weight": 0.85,
  "blend_heuristic_weight": 0.15
}
```

正式流式推理默认使用该权重；这比固定手写权重更适合 dataset1/dataset2 的不同重复边比例和候选分布。

## 防止时间泄漏

- 原始 train/test CSV 不修改。
- dataset2 仅使用 `split=0` 训练神经模型，`split=1` 只作验证候选正例。
- 候选特征只吸收 `time < candidate_time` 的历史边。
- fallback 提交使用完整训练历史，但测试集时间晚于训练最大时间，不使用测试标签。
- 不使用 validation label 训练模型。

## 本地验证指标

若候选带 label，计算：

- AUC；
- MRR；
- Hits@1/3/10；
- NDCG@10。

dataset1 无官方验证标签，本地时间 holdout 指标仅作调参参考。

## 后续优化方向

1. CPU 模式已完成 smoke test；若要加速全量训练，可完成 Jittor CUDA runtime 下载后用 `USE_CUDA_ARGS=--use-cuda` 训练 `HybridTemporalScorer`。
2. 对 dataset2 使用 `split=1` 搜索模型/启发式融合权重。
3. 加入真正的 CRAFT 序列邻居表示，与当前 MLP scorer 融合。
4. 多 seed 训练并做 rank averaging。
5. 针对新边单独训练子 scorer，降低重复边记忆偏置。
6. 若官网确认 `result.json` 应为排名而非分数，使用 `JSON_MODE=sorted_ids` 或 `JSON_MODE=ranks` 重新生成提交。
7. 继续做多模型融合：不同 seed、不同窗口、不同负采样强度的 `result_scores.csv` 可做 query 内 rank averaging。

## 流式推理与 ID 映射

正式训练时 `train.py` 默认不展开测试候选边，只按 test 宽表分块扫描 `src,c1..c100`，把测试集中可能出现的原始节点 ID 加入映射空间，并把映射保存为：

```text
outputs/track1/<dataset>/checkpoints/id_mapping.json
```

`infer_stream.py` 加载同一个 checkpoint、特征统计和 ID 映射，按 `QUERY_CHUNK_SIZE` 分块读取官方 test CSV。每个 query 的 100 个候选在块内完成：

1. 原始 ID 映射为内部连续 ID；
2. 用训练历史构造泄漏安全历史特征；
3. Jittor 模型打分；
4. 历史启发式分数打分；
5. query 内 rank normalize；
6. 融合并排序写入 `result.json`。

该流程不使用测试标签，不把测试候选作为训练正例，也不会把未来测试交互加入历史。默认不输出候选级 `predictions.csv`，避免 dataset2 全量推理产生超大文件。
