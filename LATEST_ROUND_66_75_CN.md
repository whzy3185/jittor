# 第 66-75 轮实验记录

## 线上反馈

- `60_d1_56_d2_online07_combo32_temp06/result.zip`: `1.1483767539666028`
- `61_d1_56_d2_online10_combo30_temp06/result.zip`: `1.1474276333757119`
- `62_d1_56_d2_online05_combo34_temp06/result.zip`: `1.1508469294360446`，当前已验证最强。
- `63_d1_online18_d2_56/result.zip`: `1.1457228170853109`
- `64_d1_online22_d2_56/result.zip`: `1.1469300966948146`
- `65_geo_d1_56_d2_56/result.zip`: `1.1455310552542752`
- `66_d1_62_d2_combo36_temp06/result.zip`: `1.1539027723062398`
- `67_d1_62_d2_combo36_temp04/result.zip`: `1.1520795170027691`
- `69_d1_62_d2_online07_combo36_temp06/result.zip`: `1.1550793918085815`，当前已验证最强。
- `71_d1_62_d2_online03_combo38_temp04/result.zip`: `1.1534209191115772`
- `73_d1_online18_d2_62/result.zip`: `1.1502728045793726`
- `75_s62_97_geo57_03/result.zip`: `1.1486193631185126`

结论：69 的 `dataset2 online=0.07 + temporal=0.06 + seq=0.51 + combo=0.36` 当前最强。相比 62，继续提高 combo_jittor 到 0.36 且把 online q50 提到 0.07 有收益；temporal 降到 0.04 的 67/71 不如 69，说明 temporal=0.06 更稳。73 只调整 dataset1 后接近 62 但低于 69，主增益仍在 dataset2。

第一名约 `1.37`，与当前 `1.155` 仍有明显断档。后续不再为小权重扰动批量整理提交包，只在完成大规模结构性改动后再生成新的提交包。

## 新增提交包

- `66_d1_62_d2_combo36_temp06/result.zip`
- `67_d1_62_d2_combo36_temp04/result.zip`
- `68_d1_62_d2_online03_combo36_temp06/result.zip`
- `69_d1_62_d2_online07_combo36_temp06/result.zip`
- `70_d1_62_d2_combo38_temp04/result.zip`
- `71_d1_62_d2_online03_combo38_temp04/result.zip`
- `72_d1_online22_d2_62/result.zip`
- `73_d1_online18_d2_62/result.zip`
- `74_s62_95_geo65_05/result.zip`
- `75_s62_97_geo57_03/result.zip`

## 推荐提交顺序

```text
69 -> 66 -> 71 -> 67 -> 73 -> 75
```

已提交结果表明：combo=0.36 有效，online=0.07 优于 0.05/0.03，temporal=0.06 优于 0.04。后续应转向新特征、新模型或新训练方式，而不是继续小步权重搜索。
