#!/bin/bash
set -euo pipefail

export RAY_TMPDIR=/dev/shm/ray

# Base algorithm parameters
GRPO_NO_FILTER="trainer.save_freq=-1 algorithm.adv_estimator=grpo collapse_detection.enabled=true actor_rollout_ref.rollout.rollout_filter_ratio=1"
GRPO_FILTER="trainer.save_freq=-1 algorithm.adv_estimator=grpo collapse_detection.enabled=true actor_rollout_ref.rollout.rollout_filter_ratio=0.5"
PPO_NO_FILTER="trainer.save_freq=-1 algorithm.adv_estimator=gae collapse_detection.enabled=true actor_rollout_ref.rollout.rollout_filter_ratio=1"
PPO_FILTER="trainer.save_freq=-1 algorithm.adv_estimator=gae collapse_detection.enabled=true actor_rollout_ref.rollout.rollout_filter_ratio=0.5"

LOG_DIR="logs/sokoban_mi_runs"
mkdir -p "${LOG_DIR}"

# ==================== Sokoban Experiments ====================

# 1. GRPO no filter
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_grpo_no_filter" \
    ${GRPO_NO_FILTER} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_grpo_no_filter.log"

# 2. GRPO filter
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_grpo_filter0.5" \
    ${GRPO_FILTER} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_grpo_filter0.5.log"

# 3. PPO no filter
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_ppo_no_filter" \
    ${PPO_NO_FILTER} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_ppo_no_filter.log"

# 4. PPO filter
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_ppo_filter" \
    ${PPO_FILTER} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_ppo_filter.log"
