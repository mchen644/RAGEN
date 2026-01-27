# Top-p Sweep Script

`scripts/runs/run_top_p_sweep.sh` runs a top_p sweep that tests different `actor_rollout_ref.rollout.rollout_filter_value` settings.

## Defaults

- Model: `Qwen2.5-3B`
- Project name: `ragen_top_p_sweep`
- Algorithm: `gae` with `kl_ctrl.kl_coef=0.001`, `actor_rollout_ref.actor.kl_loss_coef=0.001`, `actor_rollout_ref.actor.entropy_coeff=0.001`
- Filtering: `actor_rollout_ref.rollout.rollout_filter_strategy=top_p`, `filter_loss_scaling=sqrt`, `ppo_mini_batch_size=64`
- Environment: `es_manager.train.env_groups=16`, `group_size=16`, `env_configs.n_groups=[16]`
- Training steps: `trainer.total_training_steps=400` (unless `--steps` overrides)
- Rollout values swept: `1.0`, `0.98`, `0.95`, `0.9`, `0.8`, `0.6`, `0.4`, `nofilter`; the `nofilter` case uses `rollout_filter_include_zero=True` and keeps `rollout_filter_value=1.0`
- GPU memory utilization defaults to `0.3` and `trainer.save_freq=-1`

## Running

```bash
bash scripts/runs/run_top_p_sweep.sh \
  --rollout_filter_value 0.9,0.8,0.6 \
  --gpus 0,1 \
  --gpus-per-exp 1 \
  --save-freq 50
```

Available flags:

- `--steps N` – total training steps per run (default `400`)
- `--rollout_filter_value LIST` – comma-separated sweep values; `nofilter` toggles `rollout_filter_include_zero=True`
- `--gpus LIST` – comma-separated GPU IDs (auto-detects all GPUs when omitted)
- `--gpus-per-exp N` – how many GPUs to dedicate per experiment (must divide the total GPU count)
- `--gpu-memory-utilization V` – forwarded to `actor_rollout_ref.rollout.gpu_memory_utilization`
- `--save-freq N` – checkpoint frequency (`-1` disables saving by default)

The script auto-partitions the targeted GPUs into slots and queues experiments so each value runs once per available slot; summaries (including timing) are written to `logs/top_p_sweep_Qwen2.5-3B.log`.

## Outputs

- Experiment logs are nested under `logs/top_p_sweep_Qwen2.5-3B/<value-suffix>/`
- Summary lines per value are echoed to `logs/top_p_sweep_Qwen2.5-3B.log`
- Checkpoints land under `model_saving/top_p_sweep_Qwen2.5-3B/<value-suffix>/`
