#!/bin/bash
# RAGEN Profiling Script - 10 steps per run
# 4 algo configs (GRPO/PPO × with/without filtering) × 3 tasks × 2 thinking modes = 24 runs
# Multi-GPU parallel: each GPU runs a sequential queue with a 30s cooldown between tasks

# Common settings
STEPS=10
MODEL_SIZE=""
MODEL_PATH="Qwen/Qwen2.5-${MODEL_SIZE}-Instruct"

# Collapse detection settings (multi-turn only)
COLLAPSE_FIRST_TURN=true
COLLAPSE_MULTI_TURN=true
COLLAPSE_NUM_SAMPLES=128

# GPU settings
GPUS=()
GPUS_PROVIDED=false
GPUS_PER_EXP=1
COOLDOWN_SECONDS=30
declare -A GPU_LABELS

usage() {
    echo "Usage: $0 --model_size 3B [--samples 128] [--steps 1] [--gpus 0,1,2,3] [--gpus-per-exp 1]"
    echo "  --gpus: optional, default auto-detect via nvidia-smi"
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --model_size=*)
            MODEL_SIZE="${1#--model_size=}"
            shift 1
            ;;
        -s|--samples|--collapse-samples)
            COLLAPSE_NUM_SAMPLES="$2"
            shift 2
            ;;
        --samples=*|--collapse-samples=*)
            COLLAPSE_NUM_SAMPLES="${1#*=}"
            shift 1
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --steps=*)
            STEPS="${1#--steps=}"
            shift 1
            ;;
        --gpus)
            IFS=',' read -r -a GPUS <<< "$2"
            GPUS_PROVIDED=true
            shift 2
            ;;
        --gpus=*)
            IFS=',' read -r -a GPUS <<< "${1#--gpus=}"
            GPUS_PROVIDED=true
            shift 1
            ;;
        --gpus-per-exp)
            GPUS_PER_EXP="$2"
            shift 2
            ;;
        --gpus-per-exp=*)
            GPUS_PER_EXP="${1#--gpus-per-exp=}"
            shift 1
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

if [ -z "$MODEL_SIZE" ]; then
    echo "Error: --model_size is required"
    usage
fi

MODEL_PATH="Qwen/Qwen2.5-${MODEL_SIZE}-Instruct"

if [ "$GPUS_PROVIDED" = false ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && [ "$GPU_COUNT" -gt 0 ]; then
            GPUS=()
            for ((i=0; i<GPU_COUNT; i++)); do
                GPUS+=("$i")
            done
        fi
    fi
    if [ ${#GPUS[@]} -eq 0 ]; then
        echo "Warning: failed to auto-detect GPUs, falling back to 0-7" >&2
        GPUS=(0 1 2 3 4 5 6 7)
    fi
fi

if ! [[ "$GPUS_PER_EXP" =~ ^[0-9]+$ ]] || [ "$GPUS_PER_EXP" -lt 1 ]; then
    echo "Error: --gpus-per-exp must be a positive integer"
    exit 1
fi
if (( ${#GPUS[@]} < GPUS_PER_EXP )); then
    echo "Error: --gpus-per-exp (${GPUS_PER_EXP}) exceeds available GPUs (${#GPUS[@]})"
    exit 1
fi
if (( ${#GPUS[@]} % GPUS_PER_EXP != 0 )); then
    echo "Error: GPU count (${#GPUS[@]}) must be divisible by --gpus-per-exp (${GPUS_PER_EXP})"
    exit 1
fi

GPU_GROUPS=()
for ((i=0; i<${#GPUS[@]}; i+=GPUS_PER_EXP)); do
    group="${GPUS[$i]}"
    for ((j=1; j<GPUS_PER_EXP; j++)); do
        group+=",${GPUS[$((i+j))]}"
    done
    GPU_GROUPS+=("$group")
done
NUM_SLOTS=${#GPU_GROUPS[@]}

short_gpu_name() {
    local name="$1"
    local cleaned
    cleaned=$(echo "$name" | sed -E 's/^NVIDIA //; s/^Tesla //; s/^GeForce //; s/^Quadro //; s/^RTX //')
    if [[ "$cleaned" =~ (B[0-9]{2,3}|H[0-9]{2,3}|A[0-9]{2,3}|L[0-9]{2,3}|V100|T4|P100|K80) ]]; then
        echo "${BASH_REMATCH[1]}"
        return
    fi
    echo "${cleaned%% *}"
}

get_gpu_label() {
    local gpu_id="$1"
    if [ -n "${GPU_LABELS[$gpu_id]+x}" ]; then
        echo "${GPU_LABELS[$gpu_id]}"
        return
    fi
    local name=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$gpu_id" 2>/dev/null | head -1)
    fi
    if [ -z "$name" ]; then
        GPU_LABELS[$gpu_id]="1xGPU${gpu_id}"
        echo "${GPU_LABELS[$gpu_id]}"
        return
    fi
    local short
    short=$(short_gpu_name "$name")
    GPU_LABELS[$gpu_id]="1x${short}"
    echo "${GPU_LABELS[$gpu_id]}"
}

get_gpu_model_label() {
    local models=()
    local id label model
    for id in "${GPUS[@]}"; do
        label=$(get_gpu_label "$id")
        model="${label#1x}"
        models+=("$model")
    done
    local unique_models=()
    local m found
    for m in "${models[@]}"; do
        found=false
        for u in "${unique_models[@]}"; do
            if [ "$u" = "$m" ]; then
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            unique_models+=("$m")
        fi
    done
    if [ ${#unique_models[@]} -eq 1 ]; then
        echo "${unique_models[0]}"
    else
        echo "mixed"
    fi
}

GPU_MODEL_LABEL=$(get_gpu_model_label)
LOG_BASENAME="profiling_results_${MODEL_SIZE}_samples${COLLAPSE_NUM_SAMPLES}_${GPU_MODEL_LABEL}"
LOG_FILE="${LOG_BASENAME}.log"
RESULT_DIR="logs/${LOG_BASENAME}"

echo "=== RAGEN Profiling for $MODEL_SIZE: $(date) ===" | tee $LOG_FILE
echo "GPU per exp: ${GPUS_PER_EXP}x${GPU_MODEL_LABEL} | Model Size: ${MODEL_SIZE} | Steps: ${STEPS} | Collapse: first_turn=${COLLAPSE_FIRST_TURN}, multi_turn=${COLLAPSE_MULTI_TURN}, num_samples=${COLLAPSE_NUM_SAMPLES}" | tee -a $LOG_FILE
echo "GPUS: ${GPUS[*]} | groups: ${GPU_GROUPS[*]} | cooldown=${COOLDOWN_SECONDS}s" | tee -a $LOG_FILE


# Algorithm parameters
GRPO_FILTER="algorithm.adv_estimator=grpo actor_rollout_ref.rollout.rollout_filter_value=0.5"
GRPO_NO_FILTER="algorithm.adv_estimator=grpo actor_rollout_ref.rollout.rollout_filter_value=1"
PPO_FILTER="algorithm.adv_estimator=gae actor_rollout_ref.rollout.rollout_filter_value=0.5"
PPO_NO_FILTER="algorithm.adv_estimator=gae actor_rollout_ref.rollout.rollout_filter_value=1"

run_experiment() {
    local task=$1
    local algo=$2
    local filter=$3
    local think=$4
    local config=$5
    local gpu_list=$6
    shift 6
    local extra_args=("$@")
    local think_label="yes"
    local think_name="thinking"
    if [ "$think" = "False" ]; then
        think_label="no"
        think_name="nothink"
    fi
    local name="${task}-${algo}-${filter}-${MODEL_SIZE}-${think_name}"
    IFS=',' read -r -a gpu_ids <<< "$gpu_list"
    local gpus_per_exp=${#gpu_ids[@]}

    START=$(date +%s)
    CUDA_VISIBLE_DEVICES="${gpu_list}" python train.py --config-name $config \
        model_path="${MODEL_PATH}" \
        trainer.total_training_steps=${STEPS} \
        trainer.experiment_name=${name} \
        trainer.logger="['console']" \
        trainer.val_before_train=False \
        trainer.save_freq=-1 \
        trainer.n_gpus_per_node=${gpus_per_exp} \
        system.CUDA_VISIBLE_DEVICES="'${gpu_list}'" \
        agent_proxy.enable_think=${think} \
        collapse_detection.first_turn_enabled=${COLLAPSE_FIRST_TURN} \
        collapse_detection.multi_turn_enabled=${COLLAPSE_MULTI_TURN} \
        collapse_detection.num_samples=${COLLAPSE_NUM_SAMPLES} \
        "${extra_args[@]}" 2>&1 | tee "logs/${name}.log"
    EXIT_CODE=${PIPESTATUS[0]} 
    END=$(date +%s)

    TOTAL_TIME=$((END - START))

    # Extract timing from log (if available) and format to 2 decimal places
    TRAIN_TIME_RAW=$(grep -oP 'timing_s/train_total[:\s]+\K[\d.]+' "logs/${name}.log" | tail -1 || echo "")
    EVAL_TIME_RAW=$(grep -oP 'timing_s/eval_total[:\s]+\K[\d.]+' "logs/${name}.log" | tail -1 || echo "")
    TOTAL_TIME_RAW=$(grep -oP 'timing_s/total[:\s]+\K[\d.]+' "logs/${name}.log" | tail -1 || echo "")
    COLLAPSE_TIME_RAW=$(grep -oP 'timing_s/collapse_total[:\s]+\K[\d.]+' "logs/${name}.log" | tail -1 || echo "")
    COLLAPSE_FIRST_TIME_RAW=$(grep -oP 'timing_s/collapse_first_turn_total[:\s]+\K[\d.]+' "logs/${name}.log" | tail -1 || echo "")
    COLLAPSE_MULTI_TIME_RAW=$(grep -oP 'timing_s/collapse_multi_turn_total[:\s]+\K[\d.]+' "logs/${name}.log" | tail -1 || echo "")
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
        ERROR=$(tail -2 "logs/${name}.log" | tr '\n' ' ')
    fi

    # Store result for grouped summary
    local gpu_label
    gpu_label=$(get_gpu_label_for_list "$gpu_list")
    local summary_line="task=${task} | algo=${algo} | filter=${filter} | model=${MODEL_SIZE} | thinking=${think_label} | steps=${STEPS} | collapse=first:${COLLAPSE_FIRST_TURN},multi:${COLLAPSE_MULTI_TURN},samples:${COLLAPSE_NUM_SAMPLES} | train_time=${TRAIN_TIME}s | eval_time=${EVAL_TIME}s | collapse_time=${COLLAPSE_TIME}s | collapse_first_time=${COLLAPSE_FIRST_TIME}s | collapse_multi_time=${COLLAPSE_MULTI_TIME}s | total_time=${TOTAL_TIME_METRIC}s | wall_time=${TOTAL_TIME}s | gpu=${gpu_label} | status=${STATUS}"
    echo "${summary_line}" > "${RESULT_DIR}/${name}.result"
    echo "${summary_line}" | tee -a "$LOG_FILE"
    [ "$STATUS" = "fail" ] && echo "  error: ${ERROR}" | tee -a $LOG_FILE
}

mkdir -p logs
mkdir -p "$RESULT_DIR"
rm -f "${RESULT_DIR}"/*.result

EXPERIMENTS=()
GROUP_LABELS=()
CURRENT_GROUP=""

set_group() {
    CURRENT_GROUP="$1"
    GROUP_LABELS+=("$1")
}

add_experiment() {
    local task=$1
    local algo=$2
    local filter=$3
    local think=$4
    local config=$5
    shift 5
    local extra="$*"
    EXPERIMENTS+=("${CURRENT_GROUP}|${task}|${algo}|${filter}|${think}|${config}|${extra}")
}

# Group 1: GRPO with filtering
set_group "Group 1: GRPO with filtering"
add_experiment bandit GRPO filter0.5 True _1_bandit $GRPO_FILTER
add_experiment sokoban GRPO filter0.5 True _2_sokoban $GRPO_FILTER
add_experiment frozenlake GRPO filter0.5 True _3_frozen_lake $GRPO_FILTER

# Group 2: GRPO without filtering
set_group "Group 2: GRPO without filtering"
add_experiment bandit GRPO no_filter True _1_bandit $GRPO_NO_FILTER
add_experiment sokoban GRPO no_filter True _2_sokoban $GRPO_NO_FILTER
add_experiment frozenlake GRPO no_filter True _3_frozen_lake $GRPO_NO_FILTER

# Group 3: PPO with filtering
set_group "Group 3: PPO with filtering"
add_experiment bandit PPO filter0.5 True _1_bandit $PPO_FILTER
add_experiment sokoban PPO filter0.5 True _2_sokoban $PPO_FILTER
add_experiment frozenlake PPO filter0.5 True _3_frozen_lake $PPO_FILTER

# Group 4: PPO without filtering
set_group "Group 4: PPO without filtering"
add_experiment bandit PPO no_filter True _1_bandit $PPO_NO_FILTER
add_experiment sokoban PPO no_filter True _2_sokoban $PPO_NO_FILTER
add_experiment frozenlake PPO no_filter True _3_frozen_lake $PPO_NO_FILTER

# Group 5: GRPO with filtering (no-think)
set_group "Group 5: GRPO with filtering (no-think)"
add_experiment bandit GRPO filter0.5 False _1_bandit $GRPO_FILTER
add_experiment sokoban GRPO filter0.5 False _2_sokoban $GRPO_FILTER
add_experiment frozenlake GRPO filter0.5 False _3_frozen_lake $GRPO_FILTER

# Group 6: GRPO without filtering (no-think)
set_group "Group 6: GRPO without filtering (no-think)"
add_experiment bandit GRPO no_filter False _1_bandit $GRPO_NO_FILTER
add_experiment sokoban GRPO no_filter False _2_sokoban $GRPO_NO_FILTER
add_experiment frozenlake GRPO no_filter False _3_frozen_lake $GRPO_NO_FILTER

# Group 7: PPO with filtering (no-think)
set_group "Group 7: PPO with filtering (no-think)"
add_experiment bandit PPO filter0.5 False _1_bandit $PPO_FILTER
add_experiment sokoban PPO filter0.5 False _2_sokoban $PPO_FILTER
add_experiment frozenlake PPO filter0.5 False _3_frozen_lake $PPO_FILTER

# Group 8: PPO without filtering (no-think)
set_group "Group 8: PPO without filtering (no-think)"
add_experiment bandit PPO no_filter False _1_bandit $PPO_NO_FILTER
add_experiment sokoban PPO no_filter False _2_sokoban $PPO_NO_FILTER
add_experiment frozenlake PPO no_filter False _3_frozen_lake $PPO_NO_FILTER

get_gpu_label_for_list() {
    local gpu_list="$1"
    IFS=',' read -r -a ids <<< "$gpu_list"
    local count=${#ids[@]}
    if [ "$count" -eq 0 ]; then
        echo "0xGPU"
        return
    fi
    local first_model
    first_model="$(get_gpu_label "${ids[0]}")"
    first_model="${first_model#1x}"
    local id model
    for id in "${ids[@]:1}"; do
        model="$(get_gpu_label "$id")"
        model="${model#1x}"
        if [ "$model" != "$first_model" ]; then
            echo "${count}xmixed"
            return
        fi
    done
    echo "${count}x${first_model}"
}

run_queue_for_slot() {
    local gpu_list=$1
    local slot=$2
    local -a indices=()
    local i
    for i in "${!EXPERIMENTS[@]}"; do
        if (( i % NUM_SLOTS == slot )); then
            indices+=("$i")
        fi
    done

    local total=${#indices[@]}
    local j
    for ((j=0; j<total; j++)); do
        local exp="${EXPERIMENTS[${indices[$j]}]}"
        IFS='|' read -r exp_group task algo filter think config extra <<< "$exp"
        if [ -n "$extra" ]; then
            run_experiment "$task" "$algo" "$filter" "$think" "$config" "$gpu_list" $extra
        else
            run_experiment "$task" "$algo" "$filter" "$think" "$config" "$gpu_list"
        fi
        if (( j + 1 < total )); then
            sleep "$COOLDOWN_SECONDS"
        fi
    done
}

pids=()
for idx in "${!GPU_GROUPS[@]}"; do
    run_queue_for_slot "${GPU_GROUPS[$idx]}" "$idx" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

{
    echo ""
    echo "=== Grouped Summary ==="
    echo "GPU per exp: ${GPUS_PER_EXP}x${GPU_MODEL_LABEL} | Model Size: ${MODEL_SIZE} | Steps: ${STEPS} | Collapse: first_turn=${COLLAPSE_FIRST_TURN}, multi_turn=${COLLAPSE_MULTI_TURN}, num_samples=${COLLAPSE_NUM_SAMPLES}"
    for group_label in "${GROUP_LABELS[@]}"; do
        echo "=== ${group_label} ==="
        for exp in "${EXPERIMENTS[@]}"; do
            IFS='|' read -r exp_group task algo filter think config extra <<< "$exp"
            if [ "$exp_group" != "$group_label" ]; then
                continue
            fi
            think_name="thinking"
            if [ "$think" = "False" ]; then
                think_name="nothink"
            fi
            name="${task}-${algo}-${filter}-${MODEL_SIZE}-${think_name}"
            if [ -f "${RESULT_DIR}/${name}.result" ]; then
                cat "${RESULT_DIR}/${name}.result"
            else
                echo "task=${task} | algo=${algo} | filter=${filter} | model=${MODEL_SIZE} | thinking=${think} | status=missing"
            fi
        done
    done
} >> "$LOG_FILE"

echo "=== Profiling Completed: $(date) ===" | tee -a $LOG_FILE
