# 新电脑继续工作说明

日期：2026-05-19

## 包内容

本代码包只包含继续开发需要的内容：

```text
Python 脚本
Shell/PowerShell 运行脚本
configs/
scripts/
track1_dynamic_rec/
requirements-track1.txt
中文记录与实验总结 MD
```

不包含：

```text
官方数据
outputs/
submissions/
checkpoints/
logs/
虚拟环境
Jittor 编译缓存
__pycache__
```

这些目录体积大，且数据/提交包应在新电脑按需重新放置或单独拷贝。

## 推荐目录

新电脑仍建议使用 E 盘和 WSL：

```text
E:\Jitter\track1_aggressive_wsl
/mnt/e/Jitter/track1_aggressive_wsl

E:\Jitter\data\official_raw
/mnt/e/Jitter/data/official_raw
```

## 环境建议

进入 WSL 后：

```bash
cd /mnt/e/Jitter/track1_aggressive_wsl
python3 -m venv /mnt/e/Jitter/.venv_wsl_cpu
source /mnt/e/Jitter/.venv_wsl_cpu/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-track1.txt
python -m pip install lightgbm scikit-learn scipy pandas numpy
```

Jittor 建议使用 CPU 环境变量，避免 nvcc/gcc 不匹配：

```bash
export nvcc_path=
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
```

## 数据放置

需要保证：

```text
/mnt/e/Jitter/data/official_raw/dataset1/train.csv
/mnt/e/Jitter/data/official_raw/dataset1/test.csv
/mnt/e/Jitter/data/official_raw/dataset2/train.csv
/mnt/e/Jitter/data/official_raw/dataset2/test.csv
```

如果路径不同，运行脚本时修改 `--data-root`。

## 继续实验优先读

```text
FULL_MAJOR_WORK_SUMMARY_CN.md
CURRENT_SUBMISSION_ORDER_CN.md
FORMAT_CHECK_112_CN.md
FORMAT_CHECK_121_CN.md
LATEST_MAJOR_119_121_CN.md
```

## 当前重要脚本

```text
make_teacher_online_combo_jittor_result_zip.py
make_consensus_online_replay_result_zip.py
blend_online_score_rank_ensemble.py
blend_rank_consensus.py
blend_row_consensus_gate.py
train_lgbm_ranker.py
make_lgbm_result_zip.py
calibrate_submission_power.py
```

## 注意

```text
不要在 C 盘生成项目和大文件。
不要把官方数据提交进代码包。
不要删除原始 baseline 或官方数据。
LightGBM 是辅助专家；Jittor/JittorGeometric baseline 仍是主线。
```
