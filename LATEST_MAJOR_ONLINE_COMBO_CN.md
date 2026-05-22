# 大规模结构性改动：Online Combo Jittor

## 背景

当前已知最强是：

```text
submissions/69_d1_62_d2_online07_combo36_temp06/result.zip
```

线上分数：

```text
1.1550793918085815
```

第一名约 `1.37`，继续做小幅融合权重扰动意义有限。本轮改动不再调整 0.01 级别权重，而是把 `combo_jittor` 从静态 full-history 推理改成 pseudo-online 动态推理。

## 新增脚本

```text
make_online_combo_jittor_result_zip.py
```

核心变化：

- 读取已有 `outputs/combo_jittor/dataset2_combo_jittor.json`。
- dataset2 测试集按 `time, row_id` 顺序推理。
- 每个时间片内先打分，不立即泄漏同时间片后续信息。
- 时间片结束后，只把高置信 top1 预测写回历史。
- 同步更新：
  - sequential transition stats；
  - simple pair/pop/recency stats；
  - temporal dst/src/pair time series；
  - assoc recent/context stats。
- 再用更新后的历史影响后续时间片的 combo_jittor 特征。

本轮使用参数：

```text
update_topk = 1
margin_quantile = 0.5
mode = rank
include_valid_history = true
```

生成的结构性组件：

```text
outputs/website_submission_online_combo_jittor_q50/result.zip
```

该组件已通过提交格式校验。

## 新增提交包

只生成两个结构性融合包：

```text
submissions/76_major_onlinecombo10/result.zip
submissions/77_major_onlinecombo20/result.zip
```

含义：

- `76`: dataset1 等同 69；dataset2 = 90% 69 + 10% online-combo-jittor。
- `77`: dataset1 等同 69；dataset2 = 80% 69 + 20% online-combo-jittor。

相对 69 的差异：

```text
76: d1_mad 0.00000000, d2_mad 0.00185241
77: d1_mad 0.00000000, d2_mad 0.00370482
```

## 推荐提交

优先提交：

```text
submissions/76_major_onlinecombo10/result.zip
```

如果 76 提升，再提交：

```text
submissions/77_major_onlinecombo20/result.zip
```

如果 76 下降，则说明当前 combo_jittor 的 pseudo-online 自训练误差传播大于收益，下一步应转向重新训练 ranker 或增强 group-level/test-candidate 特征，而不是继续加 online-combo 权重。

## 环境备注

运行过程中 Jittor 曾自动尝试 CUDA 编译，因 WSL 内 `nvcc 12.2` 与 `gcc 13` 不兼容失败。之后使用：

```bash
export nvcc_path=""
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
```

强制走 CPU Jittor 路线后成功完成生成。
