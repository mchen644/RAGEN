#!/bin/bash
set -euo pipefail

export RAY_TMPDIR=/dev/shm/ray

# Base algorithm parameters
GRPO_NO_FILTER="trainer.save_freq=-1 algorithm.adv_estimator=grpo collapse_detection.enabled=true actor_rollout_ref.rollout.rollout_filter_ratio=1"
GRPO_FILTER="trainer.save_freq=-1 algorithm.adv_estimator=grpo collapse_detection.enabled=true actor_rollout_ref.rollout.rollout_filter_ratio=0.5"
PPO_NO_FILTER="trainer.save_freq=-1 algorithm.adv_estimator=gae collapse_detection.enabled=true actor_rollout_ref.rollout.rollout_filter_ratio=1"
PPO_FILTER="trainer.save_freq=-1 algorithm.adv_estimator=gae collapse_detection.enabled=true actor_rollout_ref.rollout.rollout_filter_ratio=0.5"

# Multi-turn sampling strategies
FIRST_TURN_ONLY="collapse_detection.multi_turn_sampling.enabled=false"
TURN_UNIFORM_16="collapse_detection.multi_turn_sampling.enabled=true collapse_detection.multi_turn_sampling.strategy=turn_uniform collapse_detection.multi_turn_sampling.num_samples=16"
TRAJ_UNIFORM_16="collapse_detection.multi_turn_sampling.enabled=true collapse_detection.multi_turn_sampling.strategy=trajectory_uniform collapse_detection.multi_turn_sampling.num_samples=16"

LOG_DIR="logs/sokoban_mi_runs"
mkdir -p "${LOG_DIR}"

# ==================== Strategy 1: Turn-uniform (N=16) ====================

# 1. GRPO no filter + turn uniform
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_turn16_grpo_nofilter" \
    ${GRPO_NO_FILTER} ${TURN_UNIFORM_16} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_turn16_grpo_nofilter.log" || true

# 2. GRPO filter + turn uniform
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_turn16_grpo_filter" \
    ${GRPO_FILTER} ${TURN_UNIFORM_16} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_turn16_grpo_filter.log" || true

# 3. PPO no filter + turn uniform
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_turn16_ppo_nofilter" \
    ${PPO_NO_FILTER} ${TURN_UNIFORM_16} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_turn16_ppo_nofilter.log" || true

# 4. PPO filter + turn uniform
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_turn16_ppo_filter" \
    ${PPO_FILTER} ${TURN_UNIFORM_16} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_turn16_ppo_filter.log" || true

# ==================== Strategy 2: Trajectory-uniform (N=16) ====================

# 5. GRPO no filter + traj uniform
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_traj16_grpo_nofilter" \
    ${GRPO_NO_FILTER} ${TRAJ_UNIFORM_16} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_traj16_grpo_nofilter.log" || true

# 6. GRPO filter + traj uniform
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_traj16_grpo_filter" \
    ${GRPO_FILTER} ${TRAJ_UNIFORM_16} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_traj16_grpo_filter.log" || true

# 7. PPO no filter + traj uniform
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_traj16_ppo_nofilter" \
    ${PPO_NO_FILTER} ${TRAJ_UNIFORM_16} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_traj16_ppo_nofilter.log" || true

# 8. PPO filter + traj uniform
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_traj16_ppo_filter" \
    ${PPO_FILTER} ${TRAJ_UNIFORM_16} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_traj16_ppo_filter.log" || true

# ==================== Strategy 3: First Turn Only ====================

# 9. GRPO no filter + first turn
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_first_grpo_nofilter" \
    ${GRPO_NO_FILTER} ${FIRST_TURN_ONLY} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_first_grpo_nofilter.log" || true

# 10. GRPO filter + first turn
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_first_grpo_filter" \
    ${GRPO_FILTER} ${FIRST_TURN_ONLY} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_first_grpo_filter.log" || true

# 11. PPO no filter + first turn
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_first_ppo_nofilter" \
    ${PPO_NO_FILTER} ${FIRST_TURN_ONLY} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_first_ppo_nofilter.log" || true

# 12. PPO filter + first turn
python train.py --config-name "_2_sokoban" \
    trainer.experiment_name="sokoban_mi_first_ppo_filter" \
    ${PPO_FILTER} ${FIRST_TURN_ONLY} \
    2>&1 | tee "${LOG_DIR}/sokoban_mi_first_ppo_filter.log" || true
