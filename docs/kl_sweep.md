# KL Sweep Script

`scripts/runs/run_kl_sweep.sh` keeps entropy at `0` while varying `actor_rollout_ref.actor.kl_loss_coef` (and all other rollouts stay in the same config as the main table experiments). The filter strategy is fixed to `top_p` with `rollout_filter_value=1` and an optional `rollout_filter_include_zero` flag (defaults to `True`).

## Defaults

- Model: `Qwen2.5-3B`
- Project name: `ragen_kl_sweep`
- Algorithm: `gae` with `algorithm.kl_ctrl.kl_coef=0.001`, `actor_rollout_ref.actor.kl_loss_coef` swept, `actor_rollout_ref.actor.entropy_coeff=0.0`
- Filter: `actor_rollout_ref.rollout.rollout_filter_strategy=top_p`, `rollout_filter_value=1`, `rollout_filter_type=largest`, `filter_loss_scaling=sqrt`
- Environments: `es_manager.train.env_groups=16`, `group_size=16`, `env_configs.n_groups=[16]`
- Sweep values: `0,0.001,0.003,0.01,0.03,0.1` for `actor_rollout_ref.actor.kl_loss_coef`
- GPU memory utilization defaults to `0.3`, `trainer.save_freq=-1`, and `trainer.total_training_steps=400`

## Running

```bash
bash scripts/runs/run_kl_sweep.sh --kl-values 0,0.001,0.01 --rollout_filter_include_zero False --save-freq 50
```

Available flags:

- `--kl-values LIST` – comma-separated KL coefficient sweep values (default list above)
- `--rollout_filter_include_zero BOOL` – toggle rollback inclusion of zero groups (default: `True`)
- `--steps N`, `--gpus LIST`, `--gpus-per-exp N`, `--gpu-memory-utilization V`, `--save-freq N`

Each GPU slot runs the entire sweep sequentially and writes per-value logs under `logs/kl_sweep_Qwen2.5-3B/<value-label>/`. Summary lines (timings + status) are appended to `logs/kl_sweep_Qwen2.5-3B.log`, checkpoints land in `model_saving/kl_sweep_Qwen2.5-3B/<value-label>/`.
