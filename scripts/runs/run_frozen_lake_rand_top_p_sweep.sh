#!/bin/bash
# FrozenLake top-p sweep with selectable randomness (high/low).
set -euo pipefail

# Defaults (aligned with run_main_table_diff_algo)
STEPS=400
MODEL_NAME="Qwen2.5-3B-Instruct"
MODEL_PATH="Qwen/${MODEL_NAME}"
CONFIG_NAME="_3_frozen_lake"
SAVE_FREQ=-1
RAND_MODE=""
TOP_P_VALUES=""
GPUS=()
GPUS_PROVIDED=false
GPUS_PER_EXP=1
COOLDOWN_SECONDS=30
GPU_MEMORY_UTILIZATION=0.3
declare -A GPU_LABELS

usage() {
    cat <<'EOF'
Usage: $0 [options]
Options:
  --rand MODE                  Randomness mode: high or low (required)
  --top-p LIST                 Comma-separated top-p values (required)
  --steps N                    Training steps (default: 400)
  --gpus LIST                  Comma-separated GPU IDs (auto-detected if omitted)
  --gpus-per-exp N             GPUs per experiment (default: 1)
  --cooldown SECONDS           Cooldown between runs on the same GPU group (default: 30)
  --gpu-memory-utilization V   Rollout gpu_memory_utilization (default: 0.3)
  --save-freq N                Checkpoint save frequency (default: -1)
  -h, --help                   Show this help message
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --rand) RAND_MODE="$2"; shift 2 ;;
        --rand=*) RAND_MODE="${1#*=}"; shift ;;
        --top-p) TOP_P_VALUES="$2"; shift 2 ;;
        --top-p=*) TOP_P_VALUES="${1#*=}"; shift ;;
        --steps) STEPS="$2"; shift 2 ;;
        --steps=*) STEPS="${1#*=}"; shift ;;
        --gpus) IFS=',' read -r -a GPUS <<< "$2"; GPUS_PROVIDED=true; shift 2 ;;
        --gpus=*) IFS=',' read -r -a GPUS <<< "${1#*=}"; GPUS_PROVIDED=true; shift ;;
        --gpus-per-exp) GPUS_PER_EXP="$2"; shift 2 ;;
        --gpus-per-exp=*) GPUS_PER_EXP="${1#*=}"; shift ;;
        --cooldown) COOLDOWN_SECONDS="$2"; shift 2 ;;
        --cooldown=*) COOLDOWN_SECONDS="${1#*=}"; shift ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --gpu-memory-utilization=*) GPU_MEMORY_UTILIZATION="${1#*=}"; shift ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --save-freq=*) SAVE_FREQ="${1#*=}"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [ -z "$RAND_MODE" ]; then
    echo "Error: --rand is required (high or low)"
    exit 1
fi

case "$RAND_MODE" in
    high)
        SUCCESS_RATE="0.8"
        PROJECT_NAME="ragen_frozenlake_high_rand_top_p_sweep"
        ;;
    low)
        SUCCESS_RATE="0.98"
        PROJECT_NAME="ragen_frozenlake_low_rand_top_p_sweep"
        ;;
    *)
        echo "Error: --rand must be 'high' or 'low'"
        exit 1
        ;;
esac

if [ -z "$TOP_P_VALUES" ]; then
    echo "Error: --top-p is required"
    exit 1
fi

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

LOG_FILE="logs/frozenlake_${RAND_MODE}_rand_top_p_sweep_${MODEL_NAME}.log"
RESULT_ROOT="logs/frozenlake_${RAND_MODE}_rand_top_p_sweep_${MODEL_NAME}"
CHECKPOINT_ROOT="model_saving/frozenlake_${RAND_MODE}_rand_top_p_sweep_${MODEL_NAME}"

mkdir -p logs
mkdir -p "$RESULT_ROOT"
mkdir -p "$CHECKPOINT_ROOT"

echo "=== FrozenLake top-p sweep (${RAND_MODE} rand, ${MODEL_NAME}): $(date) ===" | tee "$LOG_FILE"
echo "Top-p: ${TOP_P_VALUES} | success_rate: ${SUCCESS_RATE} | Steps: ${STEPS}" | tee -a "$LOG_FILE"
echo "GPU groups: ${GPU_GROUPS[*]} | cooldown=${COOLDOWN_SECONDS}s" | tee -a "$LOG_FILE"

parse_top_p_values() {
    IFS=',' read -r -a raw <<< "$TOP_P_VALUES"
    TOP_P_LIST=()
    for token in "${raw[@]}"; do
        token="${token// /}"
        if [ -n "$token" ]; then
            TOP_P_LIST+=("$token")
        fi
    done
    if [ ${#TOP_P_LIST[@]} -eq 0 ]; then
        echo "Error: no top-p entries provided" >&2
        exit 1
    fi
}

parse_top_p_values

run_experiment() {
    local top_p="$1"
    local gpu_list="$2"
    local safe_label="${top_p//./p}"
    safe_label="${safe_label,,}"

    local name="frozenlake_top_p_${safe_label}-sr${SUCCESS_RATE//./p}-${MODEL_NAME}"
    local task_dir="${RESULT_ROOT}/${safe_label}"
    local log_path="${task_dir}/${name}.log"
    local checkpoint_dir="${CHECKPOINT_ROOT}/${safe_label}/${name}"
    local gpus_per_exp
    IFS=',' read -r -a gpu_ids <<< "$gpu_list"
    gpus_per_exp=${#gpu_ids[@]}

    mkdir -p "$task_dir"
    mkdir -p "$checkpoint_dir"

    START=$(date +%s)
    CUDA_VISIBLE_DEVICES="${gpu_list}" python train.py --config-name "$CONFIG_NAME" \
        model_path="${MODEL_PATH}" \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${name}" \
        trainer.total_training_steps="${STEPS}" \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.default_local_dir="${checkpoint_dir}" \
        trainer.logger="['console','wandb']" \
        trainer.val_before_train=True \
        trainer.n_gpus_per_node="${gpus_per_exp}" \
        system.CUDA_VISIBLE_DEVICES="'${gpu_list}'" \
        algorithm.adv_estimator=gae \
        actor_rollout_ref.actor.loss_agg_mode=token-mean \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_type=low-var-kl \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
        actor_rollout_ref.actor.filter_loss_scaling=none \
        actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
        actor_rollout_ref.rollout.rollout_filter_strategy=top_p \
        actor_rollout_ref.rollout.rollout_filter_value="${top_p}" \
        custom_envs.CoordFrozenLake.env_config.success_rate="${SUCCESS_RATE}" \
        actor_rollout_ref.actor.checkpoint.save_contents=[model] \
        critic.checkpoint.save_contents=[model] \
        2>&1 | tee "$log_path"
    EXIT_CODE=${PIPESTATUS[0]}
    END=$(date +%s)
    TOTAL_TIME=$((END - START))

    timing_values=()
    mapfile -t timing_values < <(
        python - "$log_path" <<'PY'
import re
import sys
from pathlib import Path

def last(pattern, text):
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""

try:
    text = Path(sys.argv[1]).read_text(errors="ignore")
except Exception:
    text = ""

patterns = [
    r"timing_s/train_total[:\\s]+([\\d.]+)",
    r"timing_s/eval_total[:\\s]+([\\d.]+)",
    r"timing_s/total[:\\s]+([\\d.]+)",
]

for pattern in patterns:
    print(last(pattern, text))
PY
    )

    TRAIN_TIME_RAW="${timing_values[0]:-}"
    EVAL_TIME_RAW="${timing_values[1]:-}"
    TOTAL_TIME_RAW="${timing_values[2]:-}"
    TRAIN_TIME=$([ -n "$TRAIN_TIME_RAW" ] && printf "%.2f" "$TRAIN_TIME_RAW" || echo "N/A")
    EVAL_TIME=$([ -n "$EVAL_TIME_RAW" ] && printf "%.2f" "$EVAL_TIME_RAW" || echo "N/A")
    TOTAL_TIME_METRIC=$([ -n "$TOTAL_TIME_RAW" ] && printf "%.2f" "$TOTAL_TIME_RAW" || echo "N/A")

    local status="success"
    local error_line=""
    if [ $EXIT_CODE -ne 0 ]; then
        status="fail"
        error_line=$(tail -2 "$log_path" | tr '\n' ' ')
    fi

    local gpu_label
    gpu_label=$(get_gpu_label_for_list "$gpu_list")
    local summary_line="top_p=${top_p} | success_rate=${SUCCESS_RATE} | train_time=${TRAIN_TIME}s | eval_time=${EVAL_TIME}s | total_time=${TOTAL_TIME_METRIC}s | wall_time=${TOTAL_TIME}s | gpu=${gpu_label} | status=${status}"
    echo "${summary_line}" > "${task_dir}/${name}.result"
    echo "${summary_line}" | tee -a "$LOG_FILE"
    if [ "$status" = "fail" ]; then
        echo "  error: ${error_line}" | tee -a "$LOG_FILE"
    fi
}

QUEUE_FILE=$(mktemp -t "ragen_frozenlake_${RAND_MODE}_rand_top_p_queue.XXXXXX")
echo 0 > "$QUEUE_FILE"
QUEUE_LOCK="${QUEUE_FILE}.lock"
QUEUE_LOCK_DIR="${QUEUE_LOCK}.d"
USE_FLOCK=false
MAIN_PID=$$

cleanup_queue() {
    if [ "$MAIN_PID" != "$$" ]; then
        return
    fi
    rm -f "$QUEUE_FILE" "$QUEUE_LOCK"
    rmdir "$QUEUE_LOCK_DIR" 2>/dev/null || true
}
trap cleanup_queue EXIT

if command -v flock >/dev/null 2>&1; then
    USE_FLOCK=true
fi

next_experiment_index() {
    local idx
    if [ "$USE_FLOCK" = true ]; then
        flock -x "$QUEUE_LOCK_FD"
        idx=$(cat "$QUEUE_FILE")
        if [ -z "$idx" ]; then
            idx=0
        fi
        if (( idx >= ${#TOP_P_LIST[@]} )); then
            flock -u "$QUEUE_LOCK_FD"
            echo -1
            return
        fi
        echo $((idx + 1)) > "$QUEUE_FILE"
        flock -u "$QUEUE_LOCK_FD"
        echo "$idx"
        return
    fi

    while ! mkdir "$QUEUE_LOCK_DIR" 2>/dev/null; do
        sleep 0.05
    done
    idx=$(cat "$QUEUE_FILE")
    if [ -z "$idx" ]; then
        idx=0
    fi
    if (( idx >= ${#TOP_P_LIST[@]} )); then
        rmdir "$QUEUE_LOCK_DIR"
        echo -1
        return
    fi
    echo $((idx + 1)) > "$QUEUE_FILE"
    rmdir "$QUEUE_LOCK_DIR"
    echo "$idx"
}

run_queue_for_slot() {
    local gpu_list="$1"
    if [ "$USE_FLOCK" = true ]; then
        exec {QUEUE_LOCK_FD}>"$QUEUE_LOCK"
    fi
    while true; do
        local idx
        idx=$(next_experiment_index)
        if [ "$idx" -lt 0 ]; then
            break
        fi
        local top_p="${TOP_P_LIST[$idx]}"
        run_experiment "$top_p" "$gpu_list" || true
        if [ "$COOLDOWN_SECONDS" -gt 0 ]; then
            sleep "$COOLDOWN_SECONDS"
        fi
    done
    if [ "$USE_FLOCK" = true ]; then
        exec {QUEUE_LOCK_FD}>&-
    fi
}

pids=()
for idx in "${!GPU_GROUPS[@]}"; do
    run_queue_for_slot "${GPU_GROUPS[$idx]}" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

{
    echo ""
    echo "=== FrozenLake top-p Sweep Summary (${RAND_MODE} rand) ==="
    echo "Project: ${PROJECT_NAME} | Steps: ${STEPS} | success_rate: ${SUCCESS_RATE}"
    for val in "${TOP_P_LIST[@]}"; do
        safe_label="${val//./p}"
        safe_label="${safe_label,,}"
        name="frozenlake_top_p_${safe_label}-sr${SUCCESS_RATE//./p}-${MODEL_NAME}"
        task_dir="${RESULT_ROOT}/${safe_label}"
        if [ -f "${task_dir}/${name}.result" ]; then
            cat "${task_dir}/${name}.result"
        else
            echo "top_p=${val} | status=missing"
        fi
    done
} | tee -a "$LOG_FILE"
