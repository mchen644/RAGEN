#!/bin/bash
# GRPO vs Dr. GRPO vs Dr. GRPO+ Comparison Script
# Compares:
#   - GRPO: original (norm_adv_by_std=True, seq-mean-token-mean)
#   - Dr. GRPO: no std normalization + seq-mean-token-sum
#   - Dr. GRPO+ Soft: Dr. GRPO + soft advantage reweighting by reward variance

set -euo pipefail

# Default settings
STEPS=200
MODEL_SIZE="3B"
TASK="sokoban"
CONFIG="_2_sokoban"
GPU_LIST="0,1,2,3,4,5,6,7"

declare -a GPUS
declare -a AVAILABLE_GPUS
declare -A JOB_GPU
ACQUIRED_GPU=""
FAILED=0

initialize_gpus() {
    IFS=',' read -r -a GPUS <<< "$GPU_LIST"
    if [ ${#GPUS[@]} -eq 0 ]; then
        echo "GPU list is empty" >&2
        exit 1
    fi
    AVAILABLE_GPUS=("${GPUS[@]}")
}

acquire_gpu() {
    while [ ${#AVAILABLE_GPUS[@]} -eq 0 ]; do
        wait_for_job
    done
    local gpu="${AVAILABLE_GPUS[0]}"
    AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
    ACQUIRED_GPU="$gpu"
}

release_gpu() {
    AVAILABLE_GPUS+=("$1")
}

wait_for_job() {
    if [ ${#JOB_GPU[@]} -eq 0 ]; then
        return
    fi
    set +e
    local pid=""
    wait -n -p pid
    local exit_code=$?
    set -e
    if [ -z "$pid" ]; then
        return
    fi
    local gpu="${JOB_GPU[$pid]}"
    release_gpu "$gpu"
    unset JOB_GPU[$pid]
    if [ "$exit_code" -ne 0 ]; then
        FAILED=1
    fi
}

wait_for_all_jobs() {
    while [ ${#JOB_GPU[@]} -gt 0 ]; do
        wait_for_job
    done
}

spawn_experiment() {
    local name=$1
    local adv_norm=$2
    local loss_mode=$3
    local soft_reweight=${4:-False}
    local filter_k=${5:-4}
    local gpu
    acquire_gpu
    gpu="$ACQUIRED_GPU"
    echo ">>> Scheduling: $name on GPU $gpu"
    run_experiment "$gpu" "$name" "$adv_norm" "$loss_mode" "$soft_reweight" "$filter_k" &
    JOB_GPU[$!]="$gpu"
}

run_experiment() {
    local gpu=$1
    shift
    local name=$1
    local adv_norm=$2
    local loss_mode=$3
    local soft_reweight=${4:-False}
    local filter_k=${5:-4}

    echo ">>> Running: $name on GPU $gpu"
    echo "    norm_adv_by_std_in_grpo=$adv_norm, loss_agg_mode=$loss_mode"
    echo "    soft_advantage_reweight=$soft_reweight, rollout_filter_strategy=top_k, rollout_filter_value=$filter_k"

    CUDA_VISIBLE_DEVICES=$gpu python train.py --config-name $CONFIG \
        model_path="$MODEL_PATH" \
        trainer.project_name="ragen_drgrpo_compare" \
        trainer.total_training_steps=$STEPS \
        trainer.experiment_name="${TASK}-${name}-${MODEL_SIZE}" \
        trainer.logger="['console','wandb']" \
        trainer.val_before_train=False \
        trainer.save_freq=-1 \
        trainer.n_gpus_per_node=1 \
        system.CUDA_VISIBLE_DEVICES="'${gpu}'" \
        algorithm.adv_estimator=grpo \
        algorithm.norm_adv_by_std_in_grpo=$adv_norm \
        algorithm.soft_advantage_reweight=$soft_reweight \
        actor_rollout_ref.actor.loss_agg_mode=$loss_mode \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.rollout.rollout_filter_strategy=top_k \
        actor_rollout_ref.rollout.rollout_filter_value=$filter_k \
        2>&1 | tee "$LOG_DIR/${name}.log"

    echo ">>> Finished: $name"
    echo ""
}

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --model_size SIZE   Model size (default: 3B)"
    echo "  --steps N           Training steps (default: 200)"
    echo "  --task TASK         Task: bandit|sokoban|frozenlake (default: sokoban)"
    echo "  --gpus LIST         Comma-separated GPU IDs (default: 0-7)"
    echo "  -h, --help          Show this help"
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --model_size=*) MODEL_SIZE="${1#*=}"; shift ;;
        --steps) STEPS="$2"; shift 2 ;;
        --steps=*) STEPS="${1#*=}"; shift ;;
        --task) TASK="$2"; shift 2 ;;
        --task=*) TASK="${1#*=}"; shift ;;
        --gpus) GPU_LIST="$2"; shift 2 ;;
        --gpus=*) GPU_LIST="${1#*=}"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown: $1"; usage ;;
    esac
done

# Set config based on task
case "$TASK" in
    bandit) CONFIG="_1_bandit" ;;
    sokoban) CONFIG="_2_sokoban" ;;
    frozenlake) CONFIG="_3_frozen_lake" ;;
    *) echo "Unknown task: $TASK"; exit 1 ;;
esac

MODEL_PATH="Qwen/Qwen2.5-${MODEL_SIZE}-Instruct"
initialize_gpus
LOG_DIR="logs/drgrpo_compare_${TASK}_${MODEL_SIZE}"
mkdir -p "$LOG_DIR"

echo "=== GRPO vs Dr. GRPO vs Dr. GRPO+ Comparison ==="
echo "Task: $TASK | Model: $MODEL_SIZE | Steps: $STEPS | GPUs: ${GPUS[*]}"
echo ""

# Algorithm configurations
# GRPO: norm_adv_by_std=True, loss_agg_mode=seq-mean-token-mean
# Dr. GRPO: norm_adv_by_std=False, loss_agg_mode=seq-mean-token-sum
# Dr. GRPO+ Soft: Dr. GRPO + soft_advantage_reweight=True (no hard filtering)
#
###############################################################################
# Stage 1: No filtering (top_k=8) - Isolate the effect of soft reweighting
# Question: "Does DrGRPO/Soft itself provide benefits?"
###############################################################################
echo "========== Stage 1: No Filtering (top_k=8, keep all) =========="

# GRPO baseline without filtering
spawn_experiment "GRPO-NoFilter" True "seq-mean-token-mean" False 8

# Dr. GRPO without filtering (isolate effect of removing std norm + token-sum)
spawn_experiment "DrGRPO-NoFilter" False "seq-mean-token-sum" False 8

# Dr. GRPO+ Soft Reweighting (soft reweight benefit over DrGRPO)
spawn_experiment "DrGRPO-Soft" False "seq-mean-token-sum" True 8

###############################################################################
# Stage 2: With hard filter (top_k=4) - See interaction effects
# Question: "Is hard filtering still necessary? Can soft replace hard?"
# Note: top_k=4 keeps top 4 out of 8 groups (50%)
###############################################################################
echo "========== Stage 2: Hard Filtering (top_k=4, keep 50%) =========="

# GRPO with hard filtering
spawn_experiment "GRPO-Filter0.5" True "seq-mean-token-mean" False 4

# Dr. GRPO with hard filtering
spawn_experiment "DrGRPO-Filter0.5" False "seq-mean-token-sum" False 4

# Dr. GRPO+ Soft with hard filtering
spawn_experiment "DrGRPO-Soft-Filter0.5" False "seq-mean-token-sum" True 4

echo "=== Comparison Complete ==="
echo "Logs saved to: $LOG_DIR/"
echo ""
echo "Stage 1 (No Filter, top_k=8) - Check if DrGRPO/Soft has intrinsic benefit:"
echo "  - ${TASK}-GRPO-NoFilter-${MODEL_SIZE}"
echo "  - ${TASK}-DrGRPO-NoFilter-${MODEL_SIZE}"
echo "  - ${TASK}-DrGRPO-Soft-${MODEL_SIZE}"
echo ""
echo "Stage 2 (Hard Filter, top_k=4) - Check interaction with filtering:"
echo "  - ${TASK}-GRPO-Filter0.5-${MODEL_SIZE}"
echo "  - ${TASK}-DrGRPO-Filter0.5-${MODEL_SIZE}"
echo "  - ${TASK}-DrGRPO-Soft-Filter0.5-${MODEL_SIZE}"

wait_for_all_jobs
if [ "$FAILED" -ne 0 ]; then
    echo "Some experiments failed; check the logs above." >&2
    exit 1
fi
