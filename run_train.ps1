$ErrorActionPreference = "Stop"
param([string]$DataDir = "data")
python train.py --data-dir $DataDir --output-dir outputs/track1 --epochs 30 --batch-size 2048 --negatives 5 --val-negatives 50
