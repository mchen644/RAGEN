# Frozen Lake Filtering and Slipperiness Experiments

This document describes the configuration changes made to support experiments on Frozen Lake's slipperiness and the effects of rollout filtering.

## Configuration Changes

### Environment Configuration
The `success_rate` parameter in `FrozenLakeEnvConfig` is now exposed through the YAML configuration files. This parameter controls the probability of a successful move in the intended direction when the environment is slippery (`is_slippery=True`).

- **Global Default**: Set in [config/envs.yaml](file:///home/deimos/RAGEN/config/envs.yaml).
- **Experiment Override**: Can be overridden in experiment-specific configs like [config/_3_frozen_lake.yaml](file:///home/deimos/RAGEN/config/_3_frozen_lake.yaml) using a `custom_envs` block.

```yaml
custom_envs:
  CoordFrozenLake:
    env_config:
      success_rate: 0.8  # Probability of moving in the intended direction
```

## Grid Experiments

A new experiment script, `run_frozen_lake_filtering.sh`, has been created to conduct a grid search across different slipperiness levels and filtering modes.

### Experiment Matrix
- **Success Rates**: 1% (0.01) to 100% (1.0).
- **Filtering Modes**:
  - `filter`: Uses `top_p=0.9` for reward variance filtering.
  - `no_filter`: Uses `top_p=1.0` (filtering disabled).

### Execution
The script runs training for 400 steps per experiment and manages GPU allocation automatically.

```bash
bash run_frozen_lake_filtering.sh [gpus_per_exp]
```

Results are saved to `/mnt/permanent/xjin/20260126_frozen_lake_grid` by default.
