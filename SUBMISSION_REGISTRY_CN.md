# Track1 Submission Registry

Updated: 2026-05-28T14:07:02+08:00

## 当前结论

- 当前已知最好：`121_major_teacher110_multi_replay_ensemble_repack_checked`
- 当前已知最好线上分数：`1.2156836736721401`
- 保守保底包：`submissions/121_major_teacher110_multi_replay_ensemble_repack_checked/result.zip`
- 本次唯一默认提交包：`submissions/125_major_lgbm25_replay_fusion_repack_checked/result.zip`
- `125` 当前按未提交/未知线上反馈处理。
- 当前不要提交 `123` / `124` / `122`，等待 `125` 线上反馈。

## 已知线上结果

| package_id | package_name | online_score | status | notes |
|---|---|---:|---|---|
| 101 | `101_major_consensus_online_replay_strict` | 1.2135024872359346 | scored | 多教师一致 replay 有效 |
| 102 | `102_major_replay_rank_fusion` | 1.2148036842867738 | scored | 110 前的强基线 |
| 103 | `103_major_adaptive_replay_gate_bold` | 1.2144142783655167 | scored | 低于 102 |
| 104 | `104_major_adaptive_replay_gate_stable` | 1.2143281923992328 | scored | 低于 102 |
| 110 | `110_major_online_weighted_rank_ensemble` | 1.2151966152453952 | scored | 121 前的强锚点 |
| 111 | `111_major_online_weighted_rank_ensemble_scaled` | 1.2151332588531099 | scored | 略低于 110 |
| 121 | `121_major_teacher110_multi_replay_ensemble_repack_checked` | 1.2156836736721401 | scored_best_known | 当前已知最好 |

## 本次候选包校验

| package_id | zip_path | format_check_status | top1_change_vs_121 | risk_level | submit_priority |
|---|---|---|---:|---|---|
| 121 | `submissions/121_major_teacher110_multi_replay_ensemble_repack_checked/result.zip` | valid; dataset1=(61051,100); dataset2=(153420,100) | 0 | low | conservative_fallback |
| 125 | `submissions/125_major_lgbm25_replay_fusion_repack_checked/result.zip` | valid; dataset1=(61051,100); dataset2=(153420,100) | 0.339584 | medium_high | 1_default_submit_now |
| 123 | `submissions/123_major_lgbm45_replay_fusion_repack_checked/result.zip` | valid; dataset1=(61051,100); dataset2=(153420,100) | 0.475649 | high | hold_do_not_submit_now |
| 124 | `submissions/124_major_lgbm65_replay_fusion_repack_checked/result.zip` | valid; dataset1=(61051,100); dataset2=(153420,100) | 0.606746 | very_high | hold_do_not_submit_now |
| 122 | `submissions/122_major_lgbm_lambdarank_expert_repack_checked/result.zip` | valid; dataset1=(61051,100); dataset2=(153420,100) | 0.729142 | extreme | hold_do_not_submit_now |

## 提交策略

1. 若平台可提交且今日次数大于 0，只提交 `125`。
2. 若 `125` 正反馈，再考虑 126 LGBM stacking / multi-seed / multi-window。
3. 若 `125` 明显回退，不继续提交 `123` / `124` / `122`，改为降低 LGBM 权重或回到 121/低权重 stacking。
4. 若平台需要登录、验证码、短信、人机验证、二次确认，停止并由用户手动处理。

## 当前提交状态

- `125` 平台提交编号：unknown
- `125` 提交时间：unknown
- `125` 初始状态：submit_attempt_blocked
- 阻塞原因：
  - Playwright 无头新会话打开比赛页后仍显示“登录 / 注册”，无文件上传控件。
  - 点击“提交结果”只能看到提交结果表，无法上传。
  - 点击“立即报名”触发登录相关内容。
  - 现有 Edge 窗口未开放远程调试端口，无法接管。
  - 尝试复用 Edge 用户资料目录失败，因为当前 Edge 正在运行并锁定资料目录。
  - 未绕过登录、验证码、人机验证或二次确认；未伪造提交结果。

## 125 平台回写

- 更新时间：`2026-05-28T17:09:26+08:00`
- request_id：`2026052817073429008219`
- 提交时间：`2026-05-28 17:07:34`
- 平台状态：`完成`
- 分数：`1.199824283554553`
- 信息：`Evaluation completed successfully`

## 127_low_weight_lgbm_guarded_fusion

- 生成状态：已生成
- 校验状态：已通过
- 自动评估：submit
- 提交状态：online_best
- zip 路径：submissions/127_low_weight_lgbm_guarded_fusion_repack_checked/result.zip
- base：121_major_teacher110_multi_replay_ensemble
- LGBM 来源：125_major_lgbm25_replay_fusion
- 专家参考：110 / 112 / 119
- dataset1：完全复制 121
- dataset2：低权重 LGBM 门控修正
- 融合权重：0.08
- margin 分位：0.95
- top1 改动上限：0.05
- dataset1_mad_vs_121：0.0
- dataset2_mad_vs_121：0.0028176502739714507
- top1_change_vs_121：0.0021965845391735106
- changed_rows：101658
- gated_pass_rows：476
- risk_score：0
- risk_level：SUBMITTABLE
- online_score：1.2160804635626843
- 备注：125 已线上回退到 1.199824283554553。本包仅验证 LGBM 局部正信号；若 127 回退，则暂停 LGBM 路线。

## 130_dataset2_segment_router

- 生成状态：已生成
- 校验状态：已通过
- 自动评估：submit
- 提交状态：valid_but_not_best
- zip 路径：submissions/130_dataset2_segment_router_repack_checked/result.zip
- base：127_low_weight_lgbm_guarded_fusion
- 稳定参考：121_major_teacher110_multi_replay_ensemble
- 专家参考：112 / 117 / 119 / 110
- dataset1：完全复制 127
- dataset2：低 margin 区 replay 共识局部 router
- changed_rows_vs_127：305
- top1_change_vs_127：0.001988006778777213
- top1_change_vs_121：0.004132446877851649
- dataset2_mad_vs_127：1.0953793788293565e-06
- dataset2_mad_vs_121：0.002818594588017208
- router_candidate_rows：1607
- router_used_rows：305
- risk_score：0
- risk_level：SUBMITTABLE
- online_score：1.2160322667785048
- request_id：2026053014380875593672
- 备注：128 诊断判断互补性集中在 dataset2 少数 segment，因此 130 只做局部专家路由；不得解释为扩大 LGBM 权重。

## 133_objective_postprocess_audit

- 类型：审计分析
- 输出：analysis/133_objective_postprocess_audit.md / analysis/133_objective_postprocess_audit.json
- 结论：未发现需要优先生成 134 的明确 objective/postprocess 错误；建议先做 131 calibration-only。; 127 相对 121 的 dataset2 MAD=0.0028176503，top1_change=0.0021965845，属于极小扰动。; 130 相对 121 的 dataset2 top1_change=0.0041324469，且线上略低于 127，提示 top1 router 替换收益不稳。
- 是否建议 134：False
- 备注：用于判断当前瓶颈是否来自 objective/postprocess/dataset 子域，而非继续微调阈值。

## 131_teacher_replay_diversity_refusion

- 类型：calibration-only 候选
- base：127_low_weight_lgbm_guarded_fusion
- LGBM：不使用
- router top1 替换：不使用
- dataset1：复制 127
- dataset2：teacher/replay rank calibration，top1 guarded
- zip：submissions/131_teacher_replay_diversity_refusion_repack_checked/result.zip
- dataset1_mad_vs_127：0.0
- dataset2_mad_vs_127：2.15668054297472e-07
- top1_change_vs_127：0.0
- changed_rows：386
- auto_eval_decision：submit
- online_status：valid_but_not_best
- online_score：1.216080439955626

## 129_lgbm_guarded_ablation_suite

- 类型：反向消融定位
- 默认提交：否
- 候选：129a_p975_lgbm_guarded, 129b_p990_lgbm_guarded, 129c_p995_lgbm_guarded, 129d_same_top1_only_calibration, 129e_gated_top1_only_no_calibration
- 目的：定位 127 收益来源
- 结论：详见 analysis/129_lgbm_guarded_ablation_report.md
- auto_eval_decision：do_not_submit

## 140_144_structural_breakthrough

- 140 segments: 1
- 141 useful_probe: False
- 142 decision: review_only / mad=0.0003172552350169469 / changed=80622
- 143 decision: submit / mad=0.0025555798800228125 / changed=78000
- 144 generated: False
- submitted: 143 if status is not_submitted false; status=pending_submit; score=unknown; request_id=-

## 140_144_structural_breakthrough

- 140 segments: 1
- 141 useful_probe: False
- 142 decision: review_only / mad=0.0003172552350169469 / changed=80622
- 143 decision: submit / mad=0.0025555798800228125 / changed=78000
- 144 generated: False
- submitted: 143 if status is not_submitted false; status=online_best; score=1.216080479081863; request_id=2026053015443332062280

## 145_148_score_shape_strong_threshold_0_02

- 145 generated candidates: 12
- retained submit candidates: 0
- review_only candidates: 10
- rejected below 0.02 MAD: 2
- 146 recommend_147: False
- 147 generated: false
- 148 decision: stop_score_shape_line
- submit_target: -
- online_status: not_submitted
- reason: no meaningful strong score-shape candidate found above 0.02 MAD threshold

## 150_155_new_signal_breakthrough

- 150 can_build_retrieval_expert: False
- 150 can_build_new_model_family: False
- 151 retrieval_signal_available: False
- 151 auto_eval_decision: do_not_submit
- 152 pseudo_validation_reliable: True
- 153 generated: False
- 154 generated: False
- 155 decision: stop_new_signal_line
- submit_target: -
- online_status: not_submitted
- reason: no meaningful real data-derived new-signal candidate found above 0.02 MAD threshold

## 163_171_official_data_recovery_pipeline

- 类型：人工下载接收与真实数据 pipeline 恢复任务链
- 当前 best：143_score_shape_rebuild / 1.216080479081863
- 163 下载接收：files_received=0，status=no_downloads_found
- 164 安全解压：extract_status=ok，file_count=0
- 165 数据结构识别：has_train_labels=false，has_test_features=false，can_build_retrieval=false，can_train_model=false，can_build_validation=false
- 166 真实 pipeline：ready=false，仅生成空 schema / summary 占位，不复制任何官方数据
- 167 validation：split_built=false
- 168 real retrieval：generated=false，auto_eval_decision=do_not_submit
- 169 real model probe：generated=false，auto_eval_decision=do_not_submit
- 170 final real-data candidate：generated=false，auto_eval_decision=do_not_submit
- 171 决策：manual_review_required
- 提交状态：not_submitted
- 备注：official_data_recheck/downloads 当前为空。需要用户登录 Educoder 比赛页手动下载官方数据后，再重新运行 163-171。

## 172_175_manual_download_gate_cleanup

- 类型：官方数据下载门禁与空跑产物归档
- 当前 best：143_score_shape_rebuild / 1.216080479081863
- download_gate：blocked
- manual_download_required：true
- 170_final_real_data_candidate：不是有效候选
- 170 状态：archived_empty_placeholder / not_submitted / manual_download_required
- 170 原因：official_data_recheck/downloads 为空，没有构建真实数据派生候选
- 归档目录：_archive_unused/empty_real_data_pipeline/
- 提交状态：not_submitted
- 下一步：用户登录 Educoder 下载官方数据到 official_data_recheck/downloads/ 后，才允许重新运行 163-171

## 162_171_e_drive_real_data_recovery

- 类型：E盘官方压缩包恢复 + 真实数据 pipeline 恢复
- 找到文件：craft_baseline.zip / data_A (1).zip
- 复制目录：official_data_recheck/downloads/
- download_gate：pass
- 解压状态：ok
- schema：has_train_labels=true，has_test_features=true，can_build_retrieval=true，can_train_model=true，can_build_validation=true
- raw_dir：data/official_raw_recovered/
- validation：time split built
- 168 real retrieval：generated=True，validate_pass=True，auto_eval_decision=review_only
- 168 zip：submissions/168_real_retrieval_cooccurrence_expert_repack_checked/result.zip
- 168 dataset2_mad_vs_143：0.03562290966510773
- 168 changed_rows_vs_143：153420
- 168 top1_change_vs_143：0.12951375309607613
- 169 model probe：未训练，轻量恢复阶段仅记录可训练数据已存在
- 170 final：not generated / do_not_submit
- 171 decision：manual_review_required
- 提交状态：not_submitted
- 备注：168 是真实数据派生候选，但 top1_change_vs_143 超过自动提交阈值，必须人工复核或先做更保守的 168b 门控版本。

## 176_180_guarded_real_retrieval

- 类型：真实数据 retrieval 保守化网格与自动筛选
- 当前 best：143_score_shape_rebuild / 1.216080479081863
- 176 retrieval_signal_quality：medium
- 176 main_risk：raw 168 changes too many top1 rows
- 176 recommended_guard：top1_guard + confidence_gate + class_gate
- 177 generated_candidates：20
- 177 valid_candidates：20
- 177 auto_submit_eligible_candidates：0
- 178 best_submit_candidate：None
- 178 top_scored_candidate：177b_w010_top1_guard
- 178 top_scored_mad_vs_143：0.01680343970656395
- 178 top_scored_top1_change_vs_143：0.0
- 179 generated：false
- 180 decision：stop_guarded_retrieval_line
- 提交状态：not_submitted
- 备注：所有 177 候选均通过格式校验，但 validation probe 显示 retrieval 弱于 popularity baseline，因此 local_validation_not_bad=false，没有自动提交候选。

## 181_184_exploratory_retrieval_and_jittor_baseline

- 181 policy：新增探索性提交通道；严格提交候选仍为 0。
- 181 selected：177b_w010_top1_guard。
- 182 submit_target：`submissions/177b_w010_top1_guard_repack_checked/result.zip`
- 182 request_id：2026053018275085120483
- 182 online_status：online_best
- 182 online_score：1.2170530157111252
- 182 dataset2_mad_vs_143：0.01680343970656395
- 182 changed_rows_vs_143：144713
- 182 top1_change_vs_143：0.0
- 183 baseline：官方 craft_baseline 是 Jittor/CRAFT 路线，建议后续回到真实 Jittor 框架。
- 184 smoke：当前 Python 环境 jittor / jittor_geometric 不可用，未生成 184 提交包。

## 191_200_wsl_only_retrieval_refine

- 当前 best before run：177b_w010_top1_guard / 1.2170530157111252
- WSL-only：是，训练/生成/校验均从 WSL venv 执行。
- 197 generated：24
- 197 valid：24
- 198 strict_submit_candidates：8
- 198 exploratory_submit_candidates：2
- 198 best_submit_candidate：197a_w0075_top1_guard
- 199 source：197a_w0075_top1_guard
- 199 zip：`submissions/199_final_wsl_candidate_repack_checked/result.zip`
- 199 dataset2_mad_vs_177b：0.010981998406350613
- 199 top1_change_vs_177b：0.0
- 200 decision：submit_199
- 200 request_id：2026053019255289888115
- 200 online_score：1.2168948054252828
- 200 online_status：regressed_or_not_best
- 结论：199 未超过 177b，当前 best 仍为 177b。w=0.075 方向相对 177b 过度偏离，下一步应围绕更窄/分段的 177b retrieval refinement，而不是全局继续放大。
