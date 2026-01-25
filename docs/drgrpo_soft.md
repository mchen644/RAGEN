# Dr. GRPO and Dr. GRPO+ Soft Reweighting

## Overview

This document describes two algorithm variants for improved GRPO training:

1. **Dr. GRPO**: Removes std normalization from advantage computation and uses `seq-mean-token-sum` loss aggregation
2. **Dr. GRPO+ Soft**: Adds soft advantage reweighting based on reward variance (instead of hard filtering)

## Algorithm Comparison

| Algorithm | norm_adv_by_std | loss_agg_mode | soft_reweight | Description |
|-----------|-----------------|---------------|---------------|-------------|
| GRPO (Original) | True | seq-mean-token-mean | False | Standard GRPO with std normalization |
| Dr. GRPO | False | seq-mean-token-sum | False | No std normalization + token-sum loss |
| Dr. GRPO+ Soft | False | seq-mean-token-sum | True | Dr. GRPO + soft advantage reweighting |

---

## Dr. GRPO

### Key Changes
- **Advantage**: Uses `(R - mean)` instead of `(R - mean) / std`
- **Loss Aggregation**: Uses `seq-mean-token-sum` (sum over tokens, then mean over sequences)

### Configuration
```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: False  # Key: disable std normalization

actor_rollout_ref:
  actor:
    loss_agg_mode: seq-mean-token-sum  # Key: use token-sum
    use_kl_loss: False
```

---

## Dr. GRPO+ Soft Reweighting

### Motivation
Hard filtering drops low reward variance prompts entirely, which may discard useful training signal. Soft reweighting instead down-weights these prompts proportionally.

### Formula
```
weight = group_std / (max_group_std + epsilon)
advantages' = weight * advantages
```

Where:
- `group_std`: Reward standard deviation within each prompt group
- `max_group_std`: Maximum std across all groups in the batch
- Low variance prompts get `weight < 1.0`, high variance prompts get `weight ≈ 1.0`

### Configuration
```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: False
  soft_advantage_reweight: True  # Enable soft reweighting

actor_rollout_ref:
  actor:
    loss_agg_mode: seq-mean-token-sum
    use_kl_loss: False
  rollout:
    rollout_filter_strategy: top_k
    rollout_filter_value: 8  # Keep all groups (when env_groups=8)
```

### Logged Metrics
When `soft_advantage_reweight=True`:
- `train/soft_reweight_min`: Minimum weight across prompts
- `train/soft_reweight_max`: Maximum weight (always ~1.0)
- `train/soft_reweight_mean`: Average weight

---

## Running Experiments

### Comparison Script
Use the provided script to run ablation experiments (jobs are scheduled across the GPU list; with enough GPUs all six can run concurrently):

```bash
bash /scripts/runs/run_drgrpo_compare.sh --task sokoban --steps 200 --gpus 0,1,2,3
```

Options:
- `--task`: `bandit`, `sokoban`, or `frozenlake`
- `--steps`: Training steps (default: 200)
- `--model_size`: Model size, e.g., `3B` (default: 3B)
- `--gpus`: Comma-separated GPU IDs (default: `0,1,2,3,4,5,6,7`)

Note: the script sets `system.CUDA_VISIBLE_DEVICES` per job to ensure each experiment stays on its assigned GPU.

### Experiment Stages

**Stage 1: No Filtering (top_k=8, keep all groups)**

Isolates the intrinsic benefit of each algorithm:
- `GRPO-NoFilter`: Baseline
- `DrGRPO-NoFilter`: Effect of removing std normalization
- `DrGRPO-Soft`: Benefit of soft reweighting

**Stage 2: Hard Filtering (top_k=4)**

Tests interaction with hard filtering:
- `GRPO-Filter0.5`: GRPO with 50% filtering
- `DrGRPO-Filter0.5`: Dr. GRPO with 50% filtering
- `DrGRPO-Soft-Filter0.5`: Dr. GRPO + soft reweighting with 50% filtering

### Expected Outcomes
- Stage 1 answers: "Does Dr. GRPO/Soft provide benefits without filtering?"
- Stage 2 answers: "Can soft reweighting replace hard filtering?"

---

## Code References
- **Advantage Computation**: `ragen/trainer/core_algos.py` (`compute_grpo_outcome_advantage`)
- **Soft Reweighting**: `ragen/trainer/agent_trainer.py` (after `compute_advantage`)
- **Configuration**: `config/base.yaml` (`algorithm.soft_advantage_reweight`)
- **Experiment Script**: `scripts/runs/run_drgrpo_compare.sh`
