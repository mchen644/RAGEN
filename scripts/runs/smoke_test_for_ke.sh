#!/bin/bash
# Smoke Test Script for B200
# PPO + 3B + Thinking Sokoban, 50 steps
# Group 1: KL sweep (4 filter ratios × 4 KL values = 16 runs), entropy=0
# Group 2: Entropy sweep (4 filter ratios × 4 entropy values = 16 runs), KL=0
# Group 3: KL=0 and Entropy=0 (4 filter ratios = 4 runs)
# Group 4: Task-agnostic KL x Entropy (1 filter ratio × 4 KL × 4 entropy = 16 runs)
# Group 5: No filter, KL x Entropy (1 filter ratio × 4 KL × 4 entropy = 16 runs)
# Sync wandb: wandb sync wandb/offline-run-* (per-batch diff)

export RAY_TMPDIR=/dev/shm/ray

# Common settings
STEPS=50
MODEL_SIZE="3B"
MODEL_PATH="Qwen/Qwen2.5-${MODEL_SIZE}-Instruct"
CONFIG="_2_sokoban"
ROLLOUT_FILTER_STRATEGY="top_p"

# Batch settings: prompt_batch_size=8, samples_per_prompt=16
ENV_GROUPS=8
GROUP_SIZE=16

LOG_BASE="smoke_test_results_${MODEL_SIZE}"

# GPU settings (one job per GPU)
GPUS=(0 1 2 3 4 5 6 7)
# Skip runs already recorded in the log (set to "success" or "success fail")
SKIP_STATUSES=("success")

# Group selection
# Usage: bash scripts/runs/smoke_test_for_ke.sh --groups 1,3,5 --kl-type kl
# Default: all groups
GROUP_SET="1,2,3,4,5"
# KL loss type (single). Examples: kl, mse, low_var_kl
KL_LOSS_TYPE="low_var_kl"
while [ $# -gt 0 ]; do
    case "$1" in
        -g|--groups)
            if [ -z "${2:-}" ] || [[ "${2:-}" == -* ]]; then
                echo "Error: --groups requires a value like 1,2,3"
                exit 1
            fi
            GROUP_SET="$2"
            shift 2
            ;;
        --groups=*)
            GROUP_SET="${1#--groups=}"
            shift 1
            ;;
        --kl-type)
            if [ -z "${2:-}" ] || [[ "${2:-}" == -* ]]; then
                echo "Error: --kl-type requires a value like kl|mse|low_var_kl"
                exit 1
            fi
            KL_LOSS_TYPE="$2"
            shift 2
            ;;
        --kl-type=*)
            KL_LOSS_TYPE="${1#--kl-type=}"
            shift 1
            ;;
        --steps)
            if [ -z "${2:-}" ] || [[ "${2:-}" == -* ]]; then
                echo "Error: --steps requires a value like 50"
                exit 1
            fi
            STEPS="$2"
            shift 2
            ;;
        --steps=*)
            STEPS="${1#--steps=}"
            shift 1
            ;;
        --rollout_filter_strategy)
            if [ -z "${2:-}" ] || [[ "${2:-}" == -* ]]; then
                echo "Error: --rollout_filter_strategy requires a value like top_p|top_k|min_p|top_f"
                exit 1
            fi
            ROLLOUT_FILTER_STRATEGY="$2"
            shift 2
            ;;
        --rollout_filter_strategy=*)
            ROLLOUT_FILTER_STRATEGY="${1#--rollout_filter_strategy=}"
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [--groups 1,2,3] [--kl-type kl|mse|low_var_kl] [--steps N] [--rollout_filter_strategy top_p|top_k|min_p|top_f]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--groups 1,2,3] [--kl-type kl|mse|low_var_kl] [--steps N] [--rollout_filter_strategy top_p|top_k|min_p|top_f]"
            exit 1
            ;;
    esac
done

# normalize spaces if user passes "1, 4" etc.
GROUP_SET="${GROUP_SET//[[:space:]]/}"
KL_LOSS_TYPE="${KL_LOSS_TYPE//[[:space:]]/}"
ROLLOUT_FILTER_STRATEGY="${ROLLOUT_FILTER_STRATEGY//[[:space:]]/}"

IFS=',' read -r -a GROUP_LIST <<< "$GROUP_SET"

has_group() {
    local g="$1"
    local item
    for item in "${GROUP_LIST[@]}"; do
        if [ "$item" = "$g" ]; then
            return 0
        fi
    done
    return 1
}

# Collapse detection settings (multi-turn only)
COLLAPSE_FIRST_TURN=true
COLLAPSE_MULTI_TURN=true
COLLAPSE_NUM_SAMPLES=64
if [ "$COLLAPSE_FIRST_TURN" = true ] && [ "$COLLAPSE_MULTI_TURN" = true ]; then
    COLLAPSE_TAG="ftmt"
elif [ "$COLLAPSE_FIRST_TURN" = true ] && [ "$COLLAPSE_MULTI_TURN" = false ]; then
    COLLAPSE_TAG="ft"
elif [ "$COLLAPSE_FIRST_TURN" = false ] && [ "$COLLAPSE_MULTI_TURN" = true ]; then
    COLLAPSE_TAG="mt"
else
    COLLAPSE_TAG="nocd"
fi

LOG_DIR="logs/smoke_test_${ROLLOUT_FILTER_STRATEGY}"
LOG_DETAIL_DIR="logs/smoke_test_${ROLLOUT_FILTER_STRATEGY}_details"
LOG_FILE="${LOG_DIR}/${LOG_BASE}_${KL_LOSS_TYPE}_${COLLAPSE_TAG}.log"

mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DETAIL_DIR"
if [ ! -f "$LOG_FILE" ]; then
    echo "=== Smoke Test for $MODEL_SIZE on B200: $(date) ===" | tee "$LOG_FILE"
else
    echo "=== Smoke Test Resume for $MODEL_SIZE on B200: $(date) ===" | tee -a "$LOG_FILE"
fi
echo "Config: PPO + Thinking Sokoban + ${STEPS} steps" | tee -a "$LOG_FILE"
echo "Batch: env_groups=${ENV_GROUPS}, group_size=${GROUP_SIZE}" | tee -a "$LOG_FILE"
echo "Collapse: first_turn=${COLLAPSE_FIRST_TURN}, multi_turn=${COLLAPSE_MULTI_TURN}, num_samples=${COLLAPSE_NUM_SAMPLES}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Filter ratios (TopK 100/80/60/40)
FILTER_RATIOS=(1.0 0.8 0.6 0.4)
FILTER_NAMES=("topk100" "topk80" "topk60" "topk40")

# Filter ratios for task-agnostic group
TA_FILTER_RATIOS=(1.0)
TA_FILTER_NAMES=("topk100")

# KL coefficients
KL_COEFFS=(0.001 0.003 0.01 0.03)

# Entropy coefficients
ENTROPY_COEFFS=(0.001 0.003 0.01 0.03)

run_experiment() {
    local name=$1
    local filter_ratio=$2
    local kl_coef=$3
    local entropy_coef=$4
    local zero_task_advantage=$5
    local kl_loss_type=$6
    local gpu_id=$7

    # Determine if KL loss should be enabled
    local use_kl_loss="False"
    if [ "$kl_coef" != "0" ]; then
        use_kl_loss="True"
    fi
    # top_k expects an integer count of groups; scale ratio by env_groups
    local filter_value="$filter_ratio"
    if [ "$ROLLOUT_FILTER_STRATEGY" = "top_k" ]; then
        filter_value=$(awk -v r="$filter_ratio" -v g="$ENV_GROUPS" 'BEGIN{v=int(r*g); if(v<1) v=1; print v}')
    fi

    START=$(date +%s)
    WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES="${gpu_id}" python train.py --config-name $CONFIG \
        model_path="${MODEL_PATH}" \
        trainer.project_name="ragen-smoke_test_${ROLLOUT_FILTER_STRATEGY}" \
        trainer.total_training_steps=${STEPS} \
        trainer.experiment_name=${name} \
        trainer.logger="['console','wandb']" \
        trainer.val_before_train=False \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        system.CUDA_VISIBLE_DEVICES="${gpu_id}" \
        trainer.n_gpus_per_node=1 \
        trainer.generations_to_log_to_wandb.val=0 \
        algorithm.zero_task_advantage=${zero_task_advantage} \
        agent_proxy.enable_think=True \
        algorithm.adv_estimator=gae \
        collapse_detection.first_turn_enabled=${COLLAPSE_FIRST_TURN} \
        collapse_detection.multi_turn_enabled=${COLLAPSE_MULTI_TURN} \
        collapse_detection.num_samples=${COLLAPSE_NUM_SAMPLES} \
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        actor_rollout_ref.actor.kl_loss_coef=${kl_coef} \
        actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
        actor_rollout_ref.actor.entropy_coeff=${entropy_coef} \
        actor_rollout_ref.actor.filter_loss_scaling="sqrt" \
        actor_rollout_ref.rollout.rollout_filter_value=${filter_value} \
        actor_rollout_ref.rollout.rollout_filter_strategy=${ROLLOUT_FILTER_STRATEGY} \
        es_manager.train.env_groups=${ENV_GROUPS} \
        es_manager.train.group_size=${GROUP_SIZE} \
        2>&1 | tee "${LOG_DETAIL_DIR}/${name}.log"
    EXIT_CODE=${PIPESTATUS[0]}
    END=$(date +%s)

    TOTAL_TIME=$((END - START))

    # Extract timing from log
    TRAIN_TIME_RAW=$(grep -oP 'timing_s/train_total[:\s]+\K[\d.]+' "${LOG_DETAIL_DIR}/${name}.log" | tail -1 || echo "")
    EVAL_TIME_RAW=$(grep -oP 'timing_s/eval_total[:\s]+\K[\d.]+' "${LOG_DETAIL_DIR}/${name}.log" | tail -1 || echo "")
    TOTAL_TIME_RAW=$(grep -oP 'timing_s/total[:\s]+\K[\d.]+' "${LOG_DETAIL_DIR}/${name}.log" | tail -1 || echo "")
    COLLAPSE_TIME_RAW=$(grep -oP 'timing_s/collapse_total[:\s]+\K[\d.]+' "${LOG_DETAIL_DIR}/${name}.log" | tail -1 || echo "")
    COLLAPSE_FIRST_TIME_RAW=$(grep -oP 'timing_s/collapse_first_turn_total[:\s]+\K[\d.]+' "${LOG_DETAIL_DIR}/${name}.log" | tail -1 || echo "")
    COLLAPSE_MULTI_TIME_RAW=$(grep -oP 'timing_s/collapse_multi_turn_total[:\s]+\K[\d.]+' "${LOG_DETAIL_DIR}/${name}.log" | tail -1 || echo "")
    TRAIN_TIME=$([ -n "$TRAIN_TIME_RAW" ] && printf "%.2f" "$TRAIN_TIME_RAW" || echo "N/A")
    EVAL_TIME=$([ -n "$EVAL_TIME_RAW" ] && printf "%.2f" "$EVAL_TIME_RAW" || echo "N/A")
    TOTAL_TIME_METRIC=$([ -n "$TOTAL_TIME_RAW" ] && printf "%.2f" "$TOTAL_TIME_RAW" || echo "N/A")
    COLLAPSE_TIME=$([ -n "$COLLAPSE_TIME_RAW" ] && printf "%.2f" "$COLLAPSE_TIME_RAW" || echo "N/A")
    COLLAPSE_FIRST_TIME=$([ -n "$COLLAPSE_FIRST_TIME_RAW" ] && printf "%.2f" "$COLLAPSE_FIRST_TIME_RAW" || echo "N/A")
    COLLAPSE_MULTI_TIME=$([ -n "$COLLAPSE_MULTI_TIME_RAW" ] && printf "%.2f" "$COLLAPSE_MULTI_TIME_RAW" || echo "N/A")

    if [ $EXIT_CODE -eq 0 ]; then
        STATUS="success"
    else
        STATUS="fail"
        ERROR=$(tail -2 "${LOG_DETAIL_DIR}/${name}.log" | tr '\n' ' ')
    fi

    echo "${name} | filter=${filter_value} | kl=${kl_coef} | entropy=${entropy_coef} | collapse=first:${COLLAPSE_FIRST_TURN},multi:${COLLAPSE_MULTI_TURN},samples:${COLLAPSE_NUM_SAMPLES} | train_time=${TRAIN_TIME}s | eval_time=${EVAL_TIME}s | collapse_time=${COLLAPSE_TIME}s | collapse_first_time=${COLLAPSE_FIRST_TIME}s | collapse_multi_time=${COLLAPSE_MULTI_TIME}s | total_time=${TOTAL_TIME_METRIC}s | wall_time=${TOTAL_TIME}s | gpu=${gpu_id} | status=${STATUS}" | tee -a $LOG_FILE
    [ "$STATUS" = "fail" ] && echo "  error: ${ERROR}" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
}

mkdir -p logs

is_status_logged() {
    local name=$1
    local status=$2
    awk -v n="$name" -v s="status=${status}" '$1==n && index($0, s)>0 {found=1} END{exit !found}' "$LOG_FILE"
}

already_done() {
    local name=$1
    local st
    for st in "${SKIP_STATUSES[@]}"; do
        if is_status_logged "$name" "$st"; then
            return 0
        fi
    done
    return 1
}

# Batch scheduler: run 8 experiments concurrently, then wait + sleep 30s.
MAX_PARALLEL=8
batch_count=0
batch_pids=()
SYNC_WANDB_EVERY_BATCH=true
WANDB_OFFLINE_DIR="wandb"
WANDB_OFFLINE_GLOB="${WANDB_OFFLINE_DIR}/offline-run-*"
BATCH_BASELINE_FILE=""

list_offline_runs() {
    ls -d ${WANDB_OFFLINE_GLOB} 2>/dev/null | sort
}

init_batch_baseline() {
    if [ -n "$BATCH_BASELINE_FILE" ] && [ -f "$BATCH_BASELINE_FILE" ]; then
        rm -f "$BATCH_BASELINE_FILE"
    fi
    BATCH_BASELINE_FILE=$(mktemp)
    list_offline_runs > "$BATCH_BASELINE_FILE"
}

sync_wandb_batch() {
    if [ -z "$BATCH_BASELINE_FILE" ] || [ ! -f "$BATCH_BASELINE_FILE" ]; then
        echo "W&B sync skipped (no baseline for this batch)" | tee -a "$LOG_FILE"
        return 0
    fi

    local after_file
    after_file=$(mktemp)
    list_offline_runs > "$after_file"

    mapfile -t new_runs < <(comm -13 "$BATCH_BASELINE_FILE" "$after_file")

    rm -f "$after_file" "$BATCH_BASELINE_FILE"
    BATCH_BASELINE_FILE=""

    if [ ${#new_runs[@]} -eq 0 ]; then
        echo "W&B sync: no new offline runs in this batch." | tee -a "$LOG_FILE"
        return 0
    fi

    echo "W&B sync: ${#new_runs[@]} runs" | tee -a "$LOG_FILE"
    wandb sync "${new_runs[@]}" 2>&1 | tee -a "$LOG_FILE"
}

wait_batch() {
    if [ $batch_count -eq 0 ]; then
        return 0
    fi
    for pid in "${batch_pids[@]}"; do
        wait "$pid"
    done
    batch_pids=()
    batch_count=0
    if [ "$SYNC_WANDB_EVERY_BATCH" = true ]; then
        sync_wandb_batch
    fi
    sleep 30
}

start_job() {
    local name=$1
    local filter_ratio=$2
    local kl_coef=$3
    local entropy_coef=$4
    local zero_task_advantage=$5
    local kl_loss_type=$6
    local gpu_id

    if already_done "$name"; then
        echo "Skipping: ${name} (already logged: ${SKIP_STATUSES[*]})" | tee -a "$LOG_FILE"
        return 0
    fi

    if [ $batch_count -eq 0 ]; then
        init_batch_baseline
    fi

    gpu_id=${GPUS[$batch_count]}
    run_experiment "$name" "$filter_ratio" "$kl_coef" "$entropy_coef" "$zero_task_advantage" "$kl_loss_type" "$gpu_id" &
    batch_pids+=("$!")
    batch_count=$((batch_count + 1))

    if [ $batch_count -ge $MAX_PARALLEL ]; then
        wait_batch
    fi
}

if has_group 1; then
    echo "=== Group 1: KL sweep (entropy=0) ===" | tee -a $LOG_FILE
    echo "Filter ratios: ${FILTER_RATIOS[*]}" | tee -a $LOG_FILE
    echo "KL coeffs: ${KL_COEFFS[*]}" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE

    for i in "${!FILTER_RATIOS[@]}"; do
        filter_ratio=${FILTER_RATIOS[$i]}
        filter_name=${FILTER_NAMES[$i]}
        for kl_coef in "${KL_COEFFS[@]}"; do
            kl_name=$(echo $kl_coef | sed 's/\.//g')
            exp_name="g1-smoke-ppo-sokoban-${filter_name}-kl${kl_name}-ent0-${KL_LOSS_TYPE}-${COLLAPSE_TAG}"
            start_job "$exp_name" "$filter_ratio" "$kl_coef" "0" "False" "$KL_LOSS_TYPE"
        done
    done

    wait_batch
fi

if has_group 2; then
    echo "=== Group 2: Entropy sweep (KL=0) ===" | tee -a $LOG_FILE
    echo "Filter ratios: ${FILTER_RATIOS[*]}" | tee -a $LOG_FILE
    echo "Entropy coeffs: ${ENTROPY_COEFFS[*]}" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE

    for i in "${!FILTER_RATIOS[@]}"; do
        filter_ratio=${FILTER_RATIOS[$i]}
        filter_name=${FILTER_NAMES[$i]}
        for entropy_coef in "${ENTROPY_COEFFS[@]}"; do
            ent_name=$(echo $entropy_coef | sed 's/\.//g')
            exp_name="g2-smoke-ppo-sokoban-${filter_name}-kl0-ent${ent_name}-${KL_LOSS_TYPE}-${COLLAPSE_TAG}"
            start_job "$exp_name" "$filter_ratio" "0" "$entropy_coef" "False" "$KL_LOSS_TYPE"
        done
    done

    wait_batch
fi

if has_group 3; then
    echo "=== Group 3: KL=0 and Entropy=0 ===" | tee -a $LOG_FILE
    echo "Filter ratios: ${FILTER_RATIOS[*]}" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE

    for i in "${!FILTER_RATIOS[@]}"; do
        filter_ratio=${FILTER_RATIOS[$i]}
        filter_name=${FILTER_NAMES[$i]}
        exp_name="g3-smoke-ppo-sokoban-${filter_name}-kl0-ent0-${KL_LOSS_TYPE}-${COLLAPSE_TAG}"
        start_job "$exp_name" "$filter_ratio" "0" "0" "False" "$KL_LOSS_TYPE"
    done

    wait_batch
fi

if has_group 4; then
    echo "=== Group 4: KL x Entropy sweep (task advantage=0, no filter) ===" | tee -a $LOG_FILE
    echo "Filter ratios: ${TA_FILTER_RATIOS[*]}" | tee -a $LOG_FILE
    echo "KL coeffs: ${KL_COEFFS[*]}" | tee -a $LOG_FILE
    echo "Entropy coeffs: ${ENTROPY_COEFFS[*]}" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE

    for i in "${!TA_FILTER_RATIOS[@]}"; do
        filter_ratio=${TA_FILTER_RATIOS[$i]}
        filter_name=${TA_FILTER_NAMES[$i]}
        for kl_coef in "${KL_COEFFS[@]}"; do
            kl_name=$(echo $kl_coef | sed 's/\.//g')
            for entropy_coef in "${ENTROPY_COEFFS[@]}"; do
                ent_name=$(echo $entropy_coef | sed 's/\.//g')
                exp_name="g4-smoke-ppo-sokoban-ta-${filter_name}-kl${kl_name}-ent${ent_name}-${KL_LOSS_TYPE}-${COLLAPSE_TAG}"
                start_job "$exp_name" "$filter_ratio" "$kl_coef" "$entropy_coef" "True" "$KL_LOSS_TYPE"
            done
        done
    done

    wait_batch
fi

if has_group 5; then
    echo "=== Group 5: KL x Entropy sweep (task advantage=1, no filter) ===" | tee -a $LOG_FILE
    echo "Filter ratios: 1.0 (no filter)" | tee -a $LOG_FILE
    echo "KL coeffs: ${KL_COEFFS[*]}" | tee -a $LOG_FILE
    echo "Entropy coeffs: ${ENTROPY_COEFFS[*]}" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE

    NOFILTER_RATIO=1.0
    NOFILTER_NAME="nofilter"

    for kl_coef in "${KL_COEFFS[@]}"; do
        kl_name=$(echo $kl_coef | sed 's/\.//g')
        for entropy_coef in "${ENTROPY_COEFFS[@]}"; do
            ent_name=$(echo $entropy_coef | sed 's/\.//g')
            exp_name="g5-smoke-ppo-sokoban-${NOFILTER_NAME}-kl${kl_name}-ent${ent_name}-${KL_LOSS_TYPE}-${COLLAPSE_TAG}"
            start_job "$exp_name" "$NOFILTER_RATIO" "$kl_coef" "$entropy_coef" "False" "$KL_LOSS_TYPE"
        done
    done

    wait_batch
fi

echo "=== Smoke Test Completed: $(date) ===" | tee -a $LOG_FILE

# Summary
echo "" | tee -a $LOG_FILE
echo "=== Summary ===" | tee -a $LOG_FILE
TOTAL_RUNS=$(grep -c "status=success\|status=fail" $LOG_FILE || echo 0)
SUCCESS_RUNS=$(grep -c "status=success" $LOG_FILE || echo 0)
FAIL_RUNS=$(grep -c "status=fail" $LOG_FILE || echo 0)
echo "Total: $TOTAL_RUNS | Success: $SUCCESS_RUNS | Failed: $FAIL_RUNS" | tee -a $LOG_FILE
