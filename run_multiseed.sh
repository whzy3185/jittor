#!/usr/bin/env bash
set -euo pipefail
export PIP_CACHE_DIR=/mnt/e/Jitter/.pip_cache
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
export nvcc_path="${nvcc_path-}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"
DATASET="${1:-dataset1}"
shift || true
if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
elif [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEEDS <<< "$SEEDS"
else
  SEEDS=(42 43 44)
fi

BASE_OUT="${BASE_OUT:-outputs/track1_multiseed}"
SCORE_FILES=()
for SEED_VALUE in "${SEEDS[@]}"; do
  RUN_DIR="$BASE_OUT/${DATASET}_seed${SEED_VALUE}"
  SUB_DIR="$BASE_OUT/${DATASET}_seed${SEED_VALUE}_submission"
  echo "== train $DATASET seed=$SEED_VALUE -> $RUN_DIR"
  SEED="$SEED_VALUE" OUTPUT_DIR="$RUN_DIR" "$PYTHON_BIN" train.py \
    --data-root data/official_raw \
    --dataset "$DATASET" \
    --output-dir "$RUN_DIR" \
    --epochs "${EPOCHS:-30}" \
    --batch-size "${BATCH_SIZE:-2048}" \
    --emb-dim "${EMB_DIM:-128}" \
    --hidden-dim "${HIDDEN_DIM:-256}" \
    --time-dim "${TIME_DIM:-32}" \
    --layers "${LAYERS:-3}" \
    --dropout "${DROPOUT:-0.15}" \
    --lr "${LR:-0.001}" \
    --weight-decay "${WEIGHT_DECAY:-0.00001}" \
    --negatives "${NEGATIVES:-5}" \
    --val-negatives "${VAL_NEGATIVES:-50}" \
    --max-val-events "${MAX_VAL_EVENTS:-5000}" \
    --max-train-events "${MAX_TRAIN_EVENTS:-0}" \
    --max-test-queries "${MAX_TEST_QUERIES:-0}" \
    --val-ratio "${VAL_RATIO:-0.1}" \
    --seed "$SEED_VALUE" \
    --patience "${PATIENCE:-5}" \
    ${USE_CUDA_ARGS:-}

  echo "== infer $DATASET seed=$SEED_VALUE -> $SUB_DIR"
  "$PYTHON_BIN" infer_stream.py \
    --data-root data/official_raw \
    --dataset "$DATASET" \
    --checkpoint "$RUN_DIR/checkpoints/best.pkl" \
    --meta "$RUN_DIR/checkpoints/best_meta.json" \
    --output-dir "$SUB_DIR" \
    --batch-size "${INFER_BATCH_SIZE:-8192}" \
    --query-chunk-size "${QUERY_CHUNK_SIZE:-2048}" \
    --max-val-events "${MAX_VAL_EVENTS:-5000}" \
    --max-test-queries "${MAX_TEST_QUERIES:-0}" \
    ${USE_CUDA_ARGS:-}
  SCORE_FILES+=("$SUB_DIR/result_scores.csv")
done

echo "== ensemble ${#SCORE_FILES[@]} score files"
"$PYTHON_BIN" ensemble_scores.py \
  --data-root data/official_raw \
  --dataset "$DATASET" \
  --scores "${SCORE_FILES[@]}" \
  --output-dir "$BASE_OUT/${DATASET}_ensemble" \
  --chunk-size "${ENSEMBLE_CHUNK_SIZE:-4096}" \
  --max-queries "${MAX_TEST_QUERIES:-0}"
