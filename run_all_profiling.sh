#!/bin/bash

GPUS_PER_EXP="1"

usage() {
    echo "Usage: $0 [--gpus-per-exp 1]"
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
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

EXTRA_ARGS=(--gpus-per-exp "$GPUS_PER_EXP")

# 3B
# bash scripts/runs/run_profiling.sh --model_size 3B --samples 32 --steps 10 "${EXTRA_ARGS[@]}"
# bash scripts/runs/run_profiling.sh --model_size 3B --samples 64 --steps 10 "${EXTRA_ARGS[@]}"
# bash scripts/runs/run_profiling.sh --model_size 3B --samples 128 --steps 10 "${EXTRA_ARGS[@]}"

# # 1.5B
# bash scripts/runs/run_profiling.sh --model_size 1.5B --samples 32 --steps 10 "${EXTRA_ARGS[@]}"
# bash scripts/runs/run_profiling.sh --model_size 1.5B --samples 64 --steps 10 "${EXTRA_ARGS[@]}"
# bash scripts/runs/run_profiling.sh --model_size 1.5B --samples 128 --steps 10 "${EXTRA_ARGS[@]}"

# 7B
bash scripts/runs/run_profiling.sh --model_size 7B --samples 32 --steps 10 "${EXTRA_ARGS[@]}"
bash scripts/runs/run_profiling.sh --model_size 7B --samples 64 --steps 10 "${EXTRA_ARGS[@]}"
bash scripts/runs/run_profiling.sh --model_size 7B --samples 128 --steps 10 "${EXTRA_ARGS[@]}"
