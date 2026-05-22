$ErrorActionPreference = "Stop"
param([string]$DataDir = "data")
python infer.py --data-dir $DataDir --checkpoint outputs/track1/checkpoints/best.pkl --meta outputs/track1/checkpoints/best_meta.json --output-dir outputs/track1/submission
