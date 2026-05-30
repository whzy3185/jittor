# LATEST_MAJOR_177B_CN

## 名称

177b_w010_top1_guard

## 定位

真实官方数据派生 retrieval 的一次性探索提交候选。该包不是严格 local-validation 通过候选，而是在 181 新增探索通道下提交：dataset1 完全复制 143，dataset2 做 top1 不变的 guarded retrieval 融合。

## 提交结果

- zip：`submissions/177b_w010_top1_guard_repack_checked/result.zip`
- request_id：2026053018275085120483
- submitted_at：2026-05-30 18:27:51
- online_status：online_best
- online_score：1.2170530157111252

## 本地指标

- dataset1_mad_vs_143：0.0
- dataset2_mad_vs_143：0.01680343970656395
- changed_rows_vs_143：144713
- top1_change_vs_143：0.0
- validate_pass：True
- zip_root_pass：True

## 结论

177b 线上超过 143，说明真实 retrieval 信号在 top1 guard 下存在有效增益。下一步应沿真实数据/Jittor pipeline 深化，而不是回到 submission-output-only 后处理。
