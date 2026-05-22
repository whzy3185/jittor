# Track1 动态推荐工作总总结

日期：2026-05-19

本文汇总从开始到当前为止的主要改动、实验方向、线上反馈、有效结论、失败分支、提交注意事项和后续建议。

## 1. 基本原则与工作区

、
】

### 工作区

```text
WSL 工作目录：
/mnt/e/Jitter/track1_aggressive_wsl

Windows 对应目录：
E:\Jitter\track1_aggressive_wsl

数据目录：
/mnt/e/Jitter/data/official_raw

虚拟环境：
/mnt/e/Jitter/.venv_wsl_cpu
```

### 重要约束

```text
所有项目操作应在 WSL 中进行。
不要在 C 盘生成项目文件或大文件。
数据通过 /mnt/e 访问。
原始 baseline 和官方数据不得覆盖。
提交包统一放在 submissions/ 下。
```

### Jittor 注意事项

当前环境中 Jittor 有时会尝试走 CUDA/nvcc，并因为 gcc 版本不匹配报错。之前可用方式是使用 CPU 环境变量：

```bash
export nvcc_path=
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
```

训练/推理命令中如需 Jittor，应优先带上以上环境变量。

## 2. 数据与防泄漏原则

### 数据特点

```text
dataset1/train.csv: src,dst,time
dataset1/test.csv: src,time,c1...c100

dataset2/train.csv: src,dst,time,split
dataset2/test.csv: src,time,c1...c100
```

提交格式：

```text
result.zip
  dataset1.csv
  dataset2.csv

每个 csv：
无 header
每行 100 个概率
英文半角逗号分隔
概率范围 [0,1]
保留 8 位小数
```

### 防泄漏要求

```text
不使用测试标签。
不使用 validation/test label 训练。
不把未来真实交互写入当前历史。
涉及 test 的 pseudo-online replay 时，只使用已有提交分数作为 teacher，并且按时间顺序处理；同一时间片内不互相泄漏。
dataset2 的官方 split=1 不作为训练目标，内部模型训练使用 split=0 内部时间 holdout。
```

## 3. Baseline 继承关系

所有改动都建立在官方 Jittor/JittorGeometric/CRAFT baseline 和已有特征工程上，不是从零重写 PyTorch 项目。

核心继承点：

```text
Jittor baseline / CRAFT 体系
sequential_heuristic.py
make_temporal_result_zip.py
make_assoc_result_zip.py
make_combo_jittor_result_zip.py
make_online_combo_jittor_result_zip.py
train_combo_jittor.py
train_combo_mlp_jittor.py
```

后续引入的 LightGBM 是辅助强排序专家，最终仍与 Jittor/replay 主线融合；不是把项目改成 PyTorch。

## 4. 早期有效路线：启发式、在线 replay、组合模型

### 80 之前与 80

早期主要围绕：

```text
时间特征
序列特征
关联共现
候选集合特征
多种 score blending
```

重要节点：

```text
80_major_top4q85_35
线上约 1.2061364461925292
```

### 88-89：teacher online replay

脚本：

```text
make_teacher_online_combo_jittor_result_zip.py
```

思路：

```text
用已有强提交作为 teacher。
在 test 时间顺序中选择高置信 topK 伪边。
伪边写入 seq/temporal/assoc 历史统计。
再用 Jittor combo 模型重新打分。
```

线上结果：

```text
88 1.2095967342783347
89 1.2107659416475025
```

结论：

```text
pseudo-online replay 是有效方向。
但过高覆盖或错误 teacher 会回退。
```

## 5. MLP 与 rank boost 路线

### 93-97

脚本：

```text
train_combo_mlp_jittor.py
make_combo_mlp_result_zip.py
blend_rank_transplant.py
blend_dual_mlp_intersection.py
```

核心：

```text
使用 Jittor MLP 在 split=0 内部 holdout 上训练候选边 ranker。
再把 MLP rank 作为专家，和已有强包做 boost/consensus。
```

线上结果：

```text
93 1.2113817385926295
94 1.2114108457479125
95 约 1.21145
96 约 1.21154
97 约 1.21255
```

结论：

```text
Jittor MLP 作为辅助专家有效。
但不能大范围接管排序。
```

### 98-100 明显失败

路线：

```text
双 MLP rank transplant
topK 强移植
triple consensus MLP
```

线上：

```text
98 约 1.17398
99 约 1.17571
100 1.1788037969609755
```

结论：

```text
MLP 共识或 rank transplant 过强会破坏线上有效排序。
停止全行接管、强 topK 移植方向。
```

## 6. 101-105：多教师一致 replay 与分桶门控

### 101-102

脚本：

```text
make_consensus_online_replay_result_zip.py
blend_rank_consensus.py
```

思路：

```text
使用 80/89/94/97 多教师。
只有教师 top1/topK 一致且 margin 足够高时写入伪历史。
101 是严格一致 replay。
102 是 97 与 101 的 rank 融合。
```

线上：

```text
101 1.2135024872359346
102 1.2148036842867738
```

结论：

```text
多教师 replay 是明显有效的新信号。
简单 rank 融合优于复杂门控。
```

### 103-105

脚本：

```text
blend_adaptive_replay_gate.py
blend_hard_expert_switch.py
```

线上：

```text
103 1.2144142783655167
104 1.2143281923992328
105 提交失败
```

结论：

```text
源节点分桶门控没有超过 102。
硬切换风险高，暂不优先。
```

## 7. 106-109：候选集合 Meta 专家

脚本：

```text
make_candidate_meta_expert.py
blend_candidate_meta_gate.py
```

思路：

```text
利用公开 test.csv 候选集合结构：
候选节点全局重复
源内候选重复
候选列均值
重复候选
与强教师 top1 的一致性
```

状态：

```text
106 纯 meta 专家已生成
107 rank fusion 提交失败
108/109 meta gate 已生成
```

结论：

```text
该方向尚未证明线上有效。
在 107 提交失败后，不作为当前优先路线。
```

## 8. 110-111：线上反馈加权 Rank Ensemble

脚本：

```text
blend_online_score_rank_ensemble.py
```

思路：

```text
只纳入线上确认有效的专家：
89, 94, 97, 101, 102
按线上反馈加权 rank ensemble。
```

线上：

```text
110 1.2151966152453952
111 1.2151332588531099
```

结论：

```text
110 成为当时新最优。
111 保留 102 原分数尺度后略低，说明纯 rank 分布更适合。
```

## 9. 112-118：110 锚点的小分支

### 112-115

脚本：

```text
calibrate_submission_power.py
```

思路：

```text
112：更偏 101 replay
113：加入 103/104 少量多样性
114/115：只做概率幂校准，不改排序
```

注意：

```text
112 原始包格式检查通过，并已生成重打包版。
```

### 116-118

脚本：

```text
blend_row_consensus_gate.py
```

思路：

```text
以 110 为 base。
根据 101/112/113/102/97 的 top1 一致性做行级组合。
```

结论：

```text
117 信息量最大，偏 replay 推进。
116 过于接近 110。
118 接近 113。
```

## 10. 119-121：110 Teacher Pseudo-online Replay

脚本：

```text
make_teacher_online_combo_jittor_result_zip.py
blend_online_score_rank_ensemble.py
```

思路：

```text
用当时最优 110 作为 teacher。
重新做 pseudo-online replay。
119 是纯 teacher110 replay。
120 是 110 与 119 的 rank 融合。
121 是 110/119/112/102/101 多 replay ensemble。
```

线上：

```text
121 1.2156836736721401
```

结论：

```text
121 成为当前已知最好。
110 teacher replay 有效，但纯 replay 119 风险过高。
```

## 11. 122-125：LightGBM LambdaRank 外部强专家

用户要求停止小幅改动后，转向更先进的外部排序包。

### 环境

安装并使用：

```text
lightgbm 4.6.0
sklearn 1.8.0
scipy 1.17.1
```

注意：

```text
LightGBM 只作为辅助排序专家。
最终提交仍与 Jittor/replay 主线融合。
没有改成 PyTorch。
```

### 训练脚本

```text
train_lgbm_ranker.py
make_lgbm_result_zip.py
```

训练设置：

```text
dataset2 split=0 内部时间 holdout
holdout events = 70000
每组候选 = 80
训练样本 = 5600000
特征维度 = 44
模型 = LightGBM LambdaRank
num_boost_round = 550
num_leaves = 127
```

生成包：

```text
122_major_lgbm_lambdarank_expert
123_major_lgbm45_replay_fusion
124_major_lgbm65_replay_fusion
125_major_lgbm25_replay_fusion
```

相对 121 的变化：

```text
122: top1_change_vs_121 = 0.729142
123: top1_change_vs_121 = 0.475649
124: top1_change_vs_121 = 0.606746
125: top1_change_vs_121 = 0.339584
```

结论：

```text
122-125 是真正的大幅探索，不是小幅融合。
122 是纯 LGBM 专家，风险最高。
125 是较保守但仍大幅变化的 LGBM 融合。
123/124 是中高强度 LGBM 接管。
```

## 12. 提交失败与重打包问题

平台出现过：

```text
105 提交失败
107 提交失败
121 原始包提交失败，但重打包版成功
```

本地检查显示这些包大多格式正常。因此提交失败更可能是：

```text
平台上传/排队偶发问题
zip 元数据兼容问题
当天提交次数或队列状态
非公开平台限制
```

推荐提交时优先使用 `_repack_checked` 包。

已做格式检查：

```text
FORMAT_CHECK_121_CN.md
FORMAT_CHECK_112_CN.md
```

112-121 原始包全量检查结果：

```text
CRC 通过
根目录文件正确
dataset1.csv 61051 行
dataset2.csv 153420 行
每行 100 列
无 NaN/Inf/空值
概率范围 [0,1]
小数 8 位
```

122-125 也已生成重打包版：

```text
submissions/122_major_lgbm_lambdarank_expert_repack_checked/result.zip
submissions/123_major_lgbm45_replay_fusion_repack_checked/result.zip
submissions/124_major_lgbm65_replay_fusion_repack_checked/result.zip
submissions/125_major_lgbm25_replay_fusion_repack_checked/result.zip
```

## 13. 当前已知最好与建议提交

当前已知最好：

```text
121_major_teacher110_multi_replay_ensemble_repack_checked
线上分数：1.2156836736721401
```

如果继续追求大幅提升，建议提交顺序：

```text
1. submissions/125_major_lgbm25_replay_fusion_repack_checked/result.zip
2. submissions/123_major_lgbm45_replay_fusion_repack_checked/result.zip
3. submissions/124_major_lgbm65_replay_fusion_repack_checked/result.zip
4. submissions/122_major_lgbm_lambdarank_expert_repack_checked/result.zip
```

理由：

```text
125 相对 121 改 34% top1，属于大改但比 123/124/122 风险低。
123 改 47.6% top1，是中高强度 LGBM 接管。
124 改 60.7% top1，更激进。
122 改 72.9% top1，是纯 LGBM 专家，风险最高但最能测试上限。
```

如果需要稳健提交：

```text
submissions/121_major_teacher110_multi_replay_ensemble_repack_checked/result.zip
```

如果 125 明显回退：

```text
说明 LGBM split=0 内部 holdout 学到的排序与线上隐藏分布不匹配。
后续应降低 LGBM 权重，或者改训练样本/特征，而不是继续提高 LGBM 权重。
```

如果 125 提升：

```text
继续围绕 LightGBM 做：
多 seed LGBM
不同 holdout 窗口
不同 negatives 数
加入 121/110/101/102 分数作为 stacking 特征
训练 second-level ranker
```

## 14. 后续真正值得做的大方向

### A. LightGBM stacking ranker

当前 LGBM 只用了 CRAFT/Jittor 特征。下一步可把以下分数作为特征：

```text
110 score/rank
121 score/rank
101 replay rank
102 rank
112 rank
119 replay rank
candidate meta rank
```

训练一个二阶段 LambdaRank stacking 模型。

### B. 多 seed / 多窗口 LGBM ensemble

```text
seed 4242, 2026, 3407
holdout 30000, 70000, 120000
negatives 49, 79, 99
```

只要 125 有线上正反馈，这个方向值得扩大。

### C. 更强 pseudo-online replay

```text
teacher 使用 121 或 LGBM 融合结果
按 source 活跃度动态 update_topk
按 teacher margin 分层选择伪边
避免纯 119 那种过高 top1 偏移
```

### D. 提交系统稳定性

继续提交时：

```text
优先使用 repack_checked
一次只上传一个 result.zip
如果失败先重试同一个包，不要马上判定模型失败
记录提交时间、平台编号、得分或失败状态
```

## 15. 重要提醒

```text
不要在 C 盘生成大文件。
不要删除 E 盘官方数据。
不要直接改原始 baseline。
不要把 PyTorch 作为训练/推理主框架。
可以使用 LightGBM/sklearn 作为辅助专家，但最终方案说明中应强调 Jittor baseline/replay 主线仍在。
提交前优先检查 zip 根目录是否只有 dataset1.csv 和 dataset2.csv。
每天提交次数有限，不要再提交变化极小的包。
```

## 16. 关键文件索引

```text
CURRENT_SUBMISSION_ORDER_CN.md
FORMAT_CHECK_112_CN.md
FORMAT_CHECK_121_CN.md
LATEST_MAJOR_101_102_CN.md
LATEST_MAJOR_103_105_CN.md
LATEST_MAJOR_106_109_CN.md
LATEST_MAJOR_110_111_CN.md
LATEST_MAJOR_112_115_CN.md
LATEST_MAJOR_116_118_CN.md
LATEST_MAJOR_119_121_CN.md
train_lgbm_ranker.py
make_lgbm_result_zip.py
blend_online_score_rank_ensemble.py
make_teacher_online_combo_jittor_result_zip.py
make_consensus_online_replay_result_zip.py
```
