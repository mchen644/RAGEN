#!/bin/bash
set -euo pipefail

# usage: bash run_frozen_lake_filtering.sh [gpus_per_exp]
GPUS_PER_EXP="${1:-1}" 

# -----------------------
# GPU AUTO-DETECTION
# -----------------------
detect_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true
  else
    echo 0
  fi
}

TOTAL_GPUS=$(detect_gpus)
if [ "$TOTAL_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected via nvidia-smi." >&2
    exit 1
fi
echo "INFO: Detected $TOTAL_GPUS GPUs."

# -----------------------
# Parallel GPU Management
# -----------------------
GPU_POOL_FIFO="/tmp/gpu_pool_frozen_lake_$$"
mkfifo "$GPU_POOL_FIFO"
exec 3<>"$GPU_POOL_FIFO"
rm "$GPU_POOL_FIFO"

for ((i=0; i<TOTAL_GPUS; i++)); do
    echo "$i" >&3
done

# -----------------------
# Experiment Parameters
# -----------------------
SUCCESS_RATES=(
    "0.01" # 1%
    "0.02" # 2%
    "0.05" # 5%
    "0.1"  # 10%
    "0.2"  # 20%
    "0.3"  # 30%
    "0.5"  # 50%
    "1.0"  # 100%
)

FILTER_MODES=(
    "filter"    # top_p 0.9
    "no_filter" # top_p 1.0
)

# -----------------------
# Setup
# -----------------------
ENV="_3_frozen_lake"
OUTPUT_DIR="/mnt/permanent/xjin/20260126_frozen_lake_grid"
mkdir -p "$OUTPUT_DIR"
DONE_LIST="frozen_lake_grid_donelist.txt"
touch "$DONE_LIST"

COMMON_FLAGS_BASE="trainer.total_training_steps=400 micro_batch_size_per_gpu=4 ppo_mini_batch_size=32 trainer.save_freq=-1 \
    trainer.project_name=frozen_lake_filtering \
    algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 \
    es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8]"

echo "========================================"
echo "Starting Frozen Lake Filtering Grid"
echo "Pool: $TOTAL_GPUS GPUs | Per-Exp: $GPUS_PER_EXP"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# -----------------------
# Execution Loops
# -----------------------
for rate in "${SUCCESS_RATES[@]}"; do
    for mode in "${FILTER_MODES[@]}"; do
        
        top_p="0.9"
        [ "$mode" == "no_filter" ] && top_p="1.0"
        
        # Clean rate for naming (e.g., 0.01 -> 01)
        rate_name=$(echo "$rate" | sed 's/0\.//; s/1\.0/100/')
        exp_name="fl_3b_sr${rate_name}_${mode}"
        
        if grep -q "^${exp_name}$" "$DONE_LIST"; then
            echo "Skipping ${exp_name} (Already Done)"
            continue
        fi

        # Acquire GPUs
        allocated_gpus=()
        for ((i=0; i<GPUS_PER_EXP; i++)); do
            read -u 3 gid
            allocated_gpus+=("$gid")
        done
        gpu_csv=$(IFS=,; echo "${allocated_gpus[*]}")

        (
            echo "Running: $exp_name on GPUs $gpu_csv (success_rate=$rate, top_p=$top_p)"
            
            if CUDA_VISIBLE_DEVICES="$gpu_csv" python train.py --config-name "$ENV" \
                trainer.experiment_name="${exp_name}" \
                custom_envs.CoordFrozenLake.env_config.success_rate="${rate}" \
                actor_rollout_ref.rollout.rollout_filter_metric="reward_variance" \
                actor_rollout_ref.rollout.rollout_filter_strategy="top_p" \
                actor_rollout_ref.rollout.rollout_filter_value="${top_p}" \
                actor_rollout_ref.rollout.rollout_filter_type="largest" \
                trainer.n_gpus_per_node="${GPUS_PER_EXP}" \
                system.CUDA_VISIBLE_DEVICES="\"${gpu_csv}\"" \
                $COMMON_FLAGS_BASE \
                trainer.default_local_dir="${OUTPUT_DIR}/${exp_name}"; then
                
                echo "$exp_name" >> "$DONE_LIST"
            else
                echo "ERROR: $exp_name failed on GPUs $gpu_csv." >&2
            fi

            # Release GPUs
            for gid in "${allocated_gpus[@]}"; do
                echo "$gid" >&3
            done
        ) &
    done
done

wait
exec 3>&-
echo "All Frozen Lake grid experiments completed."
