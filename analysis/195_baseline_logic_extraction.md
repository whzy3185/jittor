# 195_baseline_logic_extraction

- id: 195
- name: baseline_logic_extraction
- created_at: 2026-05-30T19:09:16+08:00
- src_dst_time_processing: True
- candidate_c1_c100_processing: True
- use_graph_neighbor: True
- use_time_decay: True
- use_src_embedding: False
- use_dst_embedding: False
- use_global_popularity: False
- use_negative_sampling: True
- score_only_candidate_set: True
- normalization_or_softmax: True
- generates_dataset1_dataset2: False
- usable_for_retrieval_refine: True
- recommended_refine_rules: use_candidate_mask, prefer graph-neighbor/history overlap, use time-aware recent history, keep scoring restricted to c1..c100 candidates
