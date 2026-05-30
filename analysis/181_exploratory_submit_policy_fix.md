# 181_exploratory_submit_policy_fix

## 结论

- strict_submit_candidates: 0
- exploratory_submit_candidates: 4
- selected_exploratory_candidate: 177b_w010_top1_guard

## 选择理由

- zip: `submissions/177b_w010_top1_guard_repack_checked/result.zip`
- dataset2_mad_vs_143: 0.01680343970656395
- changed_rows_vs_143: 144713
- top1_change_vs_143: 0.0
- local_validation_not_bad 为 false，因此不是严格提交；本策略只把它标为一次性探索提交候选。

## 禁止项

- 不提交 raw 168。
- 不提交 LGBM takeover / router expansion / score-shape-only / 批量包。
- 提交前仍需 182 再次做 zip root 与格式校验，并确认单包脚本。
