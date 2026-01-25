#!/bin/bash

ROLLOUT_FILTER_STRATEGY="top_p" # top_p, top_k, min_p

while [ $# -gt 0 ]; do
    case "$1" in
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
            echo "Usage: $0 [--rollout_filter_strategy top_p|top_k|min_p|top_f]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--rollout_filter_strategy top_p|top_k|min_p|top_f]"
            exit 1
            ;;
    esac
done

ROLLOUT_FILTER_STRATEGY="${ROLLOUT_FILTER_STRATEGY//[[:space:]]/}"

bash scripts/runs/smoke_test_for_ke.sh --groups 1,4,5 --kl-type kl --rollout_filter_strategy "${ROLLOUT_FILTER_STRATEGY}"
bash scripts/runs/smoke_test_for_ke.sh --groups 1,4,5 --kl-type mse --rollout_filter_strategy "${ROLLOUT_FILTER_STRATEGY}"
bash scripts/runs/smoke_test_for_ke.sh --groups 1,2,3,4,5 --kl-type low_var_kl --rollout_filter_strategy "${ROLLOUT_FILTER_STRATEGY}"
