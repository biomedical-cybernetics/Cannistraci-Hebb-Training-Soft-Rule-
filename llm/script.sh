#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Sweep configuration
# ---------------------------
SPARSITIES=(0.8 0.9 0.95)
DELTAS=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

UPDATE_INTERVAL=100
PRUNING_METHOD="ri"
REMOVE_METHOD="weight_magnitude_soft"
SPARSITY_DISTRIBUTION="uniform"

LR="3e-3"
ZETA="0.1"
ITERATIVE_WARMUP_STEPS=10

# ---------------------------
# Training configuration
# ---------------------------
GPUS="0,1,2,3,4,5,6,7"
NPROC=8

RUN_NAME="llama60m"
MODEL_CONFIG="configs/llama_60m.json"
DATASET="c4"
SAVE_DIR="checkpoints"

BATCH_SIZE=64
TOTAL_BATCH_SIZE=512
NUM_TRAIN_STEPS=10000
WARMUP_STEPS=1000
EVAL_EVERY=1000
WEIGHT_DECAY=0
DTYPE="bfloat16"
OPTIMIZER="adam"

COMMON_ARGS=(
  --run_name "${RUN_NAME}"
  --model_config "${MODEL_CONFIG}"
  --dataset_name "${DATASET}"
  --lr "${LR}"
  --batch_size "${BATCH_SIZE}"
  --total_batch_size "${TOTAL_BATCH_SIZE}"
  --num_training_steps "${NUM_TRAIN_STEPS}"
  --warmup_steps "${WARMUP_STEPS}"
  --weight_decay "${WEIGHT_DECAY}"
  --dtype "${DTYPE}"
  --eval_every "${EVAL_EVERY}"
  --optimizer "${OPTIMIZER}"
  --iterative_warmup_steps "${ITERATIVE_WARMUP_STEPS}"
  --update_interval "${UPDATE_INTERVAL}"
  --dst_scheduler
  --remove_method "${REMOVE_METHOD}"
  --zeta "${ZETA}"
  --adaptive_zeta
  --BRF
  --log_to_file
  --save_dir "${SAVE_DIR}"
  --only_save_last
  --degree_dist uniform
  --start_T 1.0
  --end_T 9.0
)

run() {
  local tag="$1"; shift
  echo
  echo "============================================================"
  echo "[RUN] tag=${tag}"
  echo "CMD: $*"
  echo "============================================================"
  "$@"
}

for sparsity in "${SPARSITIES[@]}"; do
  for delta in "${DELTAS[@]}"; do

    # ---------------------------
    # CHTs
    # ---------------------------
    run "CHTs_s${sparsity}_d${delta}" \
      env CUDA_VISIBLE_DEVICES="${GPUS}" \
      python3 -m torch.distributed.run --standalone --nproc_per_node "${NPROC}" \
      torchrun_main.py \
        "${COMMON_ARGS[@]}" \
        --sparsity "${sparsity}" \
        --regrow_method "CH2_L3n_soft" \
        --brf_r "${delta}"

    # ---------------------------
    # CHTss (Density Decay version)
    # ---------------------------
    run "CHTss_granet_s${sparsity}_d${delta}" \
      env CUDA_VISIBLE_DEVICES="${GPUS}" \
      python3 -m torch.distributed.run --standalone --nproc_per_node "${NPROC}" \
      torchrun_main.py \
        "${COMMON_ARGS[@]}" \
        --sparsity "${sparsity}" \
        --regrow_method "CH2_L3n_soft" \
        --granet \
        --granet_init_sparsity 0.5 \
        --sparsity_distribution "${SPARSITY_DISTRIBUTION}" \
        --pruning_method "${PRUNING_METHOD}" \
        --pruning_scheduler "s_shape" \
        --pruning_T_end 8000 \
        --brf_r "${delta}"

  done
done
