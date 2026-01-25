#!/bin/bash
set -euo pipefail

# usage: bash run_filtering_pergpu.sh [grpo|ppo|all] [metric] [largest|smallest|all] [total_gpus] [gpus_per_exp]
ALGO="${1:-grpo}" # default to grpo
METRIC="${2:-reward_variance}" # default to reward_variance
TYPE_ARG="${3:-largest,smallest}" # default to both
NGPUS_LIMIT="${4:-8}" # total GPUs available
GPUS_PER_EXP="${5:-1}" # GPUs to use per experiment
EXP_NAME="final0123"
DONE_LIST="filter_exp_donelist.txt"
touch "$DONE_LIST"

# -----------------------
# Parallel GPU Management
# -----------------------
# Create a FIFO to act as a semaphore/pool for GPU IDs
GPU_POOL_FIFO="/tmp/gpu_pool_$$"
mkfifo "$GPU_POOL_FIFO"
exec 3<>"$GPU_POOL_FIFO"
rm "$GPU_POOL_FIFO" # Securely cleanup but keep fd open

# Fill the pool with GPU IDs [0, 1, ..., N-1]
for ((i=0; i<NGPUS_LIMIT; i++)); do
    echo "$i" >&3
done

# -----------------------
# Common flags helper
# -----------------------
get_common_flags() {
  local metric=$1
  local gpu_ids=$2
  echo "trainer.total_training_steps=400 micro_batch_size_per_gpu=4 ppo_mini_batch_size=32 trainer.save_freq=-1 \
    trainer.n_gpus_per_node=${GPUS_PER_EXP} system.CUDA_VISIBLE_DEVICES=\"${gpu_ids}\" \
    algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.rollout_filter_metric=${metric} \
    es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8]"
}

ENV="_2_sokoban"
OUTPUT_DIR="/mnt/permanent/xjin/20260120_sokoban_filters"
mkdir -p "$OUTPUT_DIR"

# Define configurations to iterate
CONFIGS=(
    "top_p 0.5 topp50"
    "top_p 0.7 topp70"
    "top_p 0.9 topp90"
    "top_p 0.95 topp95"
    "min_p 0.5 minp50"
    "min_p 0.8 minp80"
    "min_p 0.95 minp95"
    "top_k 4 topk4"
)

INC_ZEROS=(
    "False noinc0"
)

LOSS_SCALES=(
    "sqrt sqrtscale"
)

run_exps_for_algo() {
    local alg_name=$1
    local alg_flag=$2
    local metric=$3
    local selected_types=("${@:4}")

    echo "========================================"
    echo "Starting parallel experiments for: $alg_name | Metric: $metric | Pool: $NGPUS_LIMIT GPUs | Per-Exp: $GPUS_PER_EXP"
    echo "========================================"

    # 1. Baseline: No Filtering
    local base_exp_name="soko_3b_${alg_name}_nofilter"
    mkdir -p "${OUTPUT_DIR}/${base_exp_name}"
    if grep -q "^${base_exp_name}$" "$DONE_LIST"; then
        echo "Skipping ${base_exp_name} (Already in done-list)"
    else
        # Acquire GPUS_PER_EXP GPUs
        local allocated_gpus=()
        for ((i=0; i<GPUS_PER_EXP; i++)); do
            read -u 3 gid
            allocated_gpus+=("$gid")
        done
        local gpu_csv=$(IFS=,; echo "${allocated_gpus[*]}")

        (
            echo "Running Baseline: $base_exp_name on GPUs $gpu_csv"
            local flags=$(get_common_flags "$metric" "$gpu_csv")
            if CUDA_VISIBLE_DEVICES="$gpu_csv" python train.py --config-name "$ENV" \
                trainer.experiment_name="${base_exp_name}" \
                actor_rollout_ref.rollout.rollout_filter_strategy="top_p" \
                actor_rollout_ref.rollout.rollout_filter_value=1.0 \
                actor_rollout_ref.rollout.rollout_filter_type="largest" \
                actor_rollout_ref.rollout.rollout_filter_include_zero=True \
                $alg_flag \
                $flags \
                trainer.default_local_dir="${OUTPUT_DIR}/${base_exp_name}"; then
                
                echo "$base_exp_name" >> "$DONE_LIST"
            else
                echo "ERROR: Baseline $base_exp_name failed on GPUs $gpu_csv." >&2
            fi
            # Release GPUs
            for gid in "${allocated_gpus[@]}"; do
                echo "$gid" >&3
            done
        ) &
    fi

    # 2. Grid Search
    for config_str in "${CONFIGS[@]}"; do
        read -r strategy value stra_suffix <<< "$config_str"

        for ftype in "${selected_types[@]}"; do
            local type_suffix
            if [ "$ftype" == "smallest" ]; then
                type_suffix="small"
            else
                type_suffix="large"
            fi

            for inc_str in "${INC_ZEROS[@]}"; do
                read -r inc_bool inc_suffix <<< "$inc_str"

                for scale_str in "${LOSS_SCALES[@]}"; do
                    read -r scaling scale_suffix <<< "$scale_str"

                    local exp_name="soko_3b_${alg_name}_${metric}_${stra_suffix}_${type_suffix}_${inc_suffix}_${scale_suffix}"
                    mkdir -p "${OUTPUT_DIR}/${exp_name}"
                    
                    if grep -q "^${exp_name}$" "$DONE_LIST"; then
                        echo "Skipping ${exp_name} (Already in done-list)"
                    else
                        # Acquire GPUS_PER_EXP GPUs (blocks if not enough available)
                        local allocated_gpus=()
                        for ((i=0; i<GPUS_PER_EXP; i++)); do
                            read -u 3 gid
                            allocated_gpus+=("$gid")
                        done
                        local gpu_csv=$(IFS=,; echo "${allocated_gpus[*]}")

                        (
                            echo "Running Experiment: $exp_name on GPUs $gpu_csv (Strategy: $strategy, Value: $value)"
                            local flags=$(get_common_flags "$metric" "$gpu_csv")
                            if CUDA_VISIBLE_DEVICES="$gpu_csv" python train.py --config-name "$ENV" \
                                trainer.experiment_name="${exp_name}" \
                                actor_rollout_ref.rollout.rollout_filter_strategy="${strategy}" \
                                actor_rollout_ref.rollout.rollout_filter_value=${value} \
                                actor_rollout_ref.rollout.rollout_filter_type="${ftype}" \
                                actor_rollout_ref.rollout.rollout_filter_include_zero=${inc_bool} \
                                actor_rollout_ref.actor.filter_loss_scaling="${scaling}" \
                                $alg_flag \
                                $flags \
                                trainer.default_local_dir="${OUTPUT_DIR}/${exp_name}"; then
                                
                                echo "$exp_name" >> "$DONE_LIST"
                            else
                                echo "ERROR: Experiment $exp_name failed on GPUs $gpu_csv." >&2
                            fi
                            # Release GPUs
                            for gid in "${allocated_gpus[@]}"; do
                                echo "$gid" >&3
                            done
                        ) &
                    fi
                done
            done
        done
    done
    wait # Wait for all parallel jobs in this algo/metric combo to finish
}

IFS=',' read -ra ALGOS <<< "$ALGO"
IFS=',' read -ra METRICS <<< "$METRIC"

# Handle TYPE_ARG
if [ "$TYPE_ARG" == "all" ]; then
    SELECTED_TYPES=("largest" "smallest")
else
    IFS=',' read -ra SELECTED_TYPES <<< "$TYPE_ARG"
fi

for m in "${METRICS[@]}"; do
    for a in "${ALGOS[@]}"; do
        if [ "$a" == "grpo" ]; then
            run_exps_for_algo "grpo" "algorithm.adv_estimator=grpo" "$m" "${SELECTED_TYPES[@]}"
        elif [ "$a" == "ppo" ]; then
            run_exps_for_algo "ppo" "algorithm.adv_estimator=gae" "$m" "${SELECTED_TYPES[@]}"
        elif [ "$a" == "all" ]; then
            run_exps_for_algo "grpo" "algorithm.adv_estimator=grpo" "$m" "${SELECTED_TYPES[@]}"
            run_exps_for_algo "ppo" "algorithm.adv_estimator=gae" "$m" "${SELECTED_TYPES[@]}"
        else
            echo "Unknown algorithm argument: $a"
            echo "Usage: bash run_filtering_pergpu.sh [algo] [metric] [type] [total_gpus] [gpus_per_exp]"
            exit 1
        fi
    done
done

exec 3>&- # Close the pool FIFO
echo "All requested experiments completed."
