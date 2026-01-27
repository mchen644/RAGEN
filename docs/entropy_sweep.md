# Entropy Sweep Script

`scripts/runs/run_entropy_sweep.sh` keeps the KL coefficient fixed (0.001) while exploring multiple `actor_rollout_ref.actor.entropy_coeff` values. The filter strategy is again `top_p` with `rollout_filter_value=1`, `rollout_filter_type=largest`, and the optional `rollout_filter_include_zero` flag (default `True`).

## Defaults

- Model: `Qwen2.5-3B`
- Project name: `ragen_entropy_sweep`
- Algorithm: `gae` with `algorithm.kl_ctrl.kl_coef=0.001`, `actor_rollout_ref.actor.kl_loss_coef=0.001`
- Filter: `filter_loss_scaling=sqrt`, `rollout_filter_strategy=top_p`, `rollout_filter_value=1`, `rollout_filter_type=largest`
- Environment: `es_manager.train.env_groups=16`, `group_size=16`, `env_configs.n_groups=[16]`
- Sweep values: `0,0.001,0.003,0.01,0.03,0.1` for `actor_rollout_ref.actor.entropy_coeff`
- GPU memory utilization defaults to `0.3`, `trainer.total_training_steps=400`, `trainer.save_freq=-1`

## Running

```bash
bash scripts/runs/run_entropy_sweep.sh --entropy-values 0.001,0.01 --rollout_filter_include_zero False --save-freq 50
```

Options:

- `--entropy-values LIST` – comma-separated entropy coefficients
- `--rollout_filter_include_zero BOOL` – include zero-score rollout groups (default `True`)
- `--steps`, `--gpus`, `--gpus-per-exp`, `--gpu-memory-utilization`, `--save-freq`

Logs and checkpoints mirror the KL sweep structure: `logs/entropy_sweep_Qwen2.5-3B/<value>/` plus the summary log `logs/entropy_sweep_Qwen2.5-3B.log`, checkpoints under `model_saving/entropy_sweep_Qwen2.5-3B/<value>/`.
