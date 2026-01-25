# KL/Entropy Smoke Tests (smoke_test_for_ke.sh, run_all_kl_entropy_exps.sh)

This doc explains how to run the KL/entropy smoke-test grid and where to find outputs.

## What it does
- Launches PPO + Thinking Sokoban runs (default: Qwen2.5-3B-Instruct, 50 steps).
- Sweeps KL/entropy across 5 groups:
  - Group 1: KL sweep, entropy=0 (4 filter ratios x 4 KL values = 16 runs)
  - Group 2: Entropy sweep, KL=0 (4 filter ratios x 4 entropy values = 16 runs)
  - Group 3: KL=0 and entropy=0 (4 filter ratios = 4 runs)
  - Group 4: KL x entropy sweep, zero_task_advantage=True (1 filter ratio x 4 KL x 4 entropy = 16 runs)
  - Group 5: KL x entropy sweep, zero_task_advantage=False (1 filter ratio x 4 KL x 4 entropy = 16 runs)
- Runs up to 8 experiments in parallel (GPU 0-7), then waits for the batch and sleeps 30s.
- Uses W&B offline mode and syncs new offline runs per batch.
- Writes per-run timing (train/eval/total/collapse) and a summary with success/fail counts.

## How to run
From repo root:

```bash
bash scripts/runs/smoke_test_for_ke.sh --groups 1,3,5 --kl-type kl --steps 50 --rollout_filter_strategy top_p
```

Arguments:
- `--groups` (optional, default `1,2,3,4,5`)
- `--kl-type` (optional, default `low_var_kl`; choices: `kl|mse|low_var_kl`)
- `--steps` (optional, default `50`)
- `--rollout_filter_strategy` (optional, default `top_p`; choices: `top_p|top_k|min_p|top_f`)

Notes for `top_k`:
- The filter ratio is converted to an integer value via `int(ratio * env_groups)`, min 1.

### Run the full KL-loss sweep
The wrapper script runs multiple KL-loss types in sequence:

```bash
bash run_all_kl_entropy_exps.sh --rollout_filter_strategy top_k
```

It executes:
- `kl` with groups `1,4,5`
- `mse` with groups `1,4,5`
- `low_var_kl` with groups `1,2,3,4,5`

## Outputs
The smoke test writes:
- Summary log: `logs/smoke_test_${ROLLOUT_FILTER_STRATEGY}/smoke_test_results_${MODEL_SIZE}_${KL_LOSS_TYPE}_${COLLAPSE_TAG}.log`
- Per-run logs: `logs/smoke_test_${ROLLOUT_FILTER_STRATEGY}_details/${experiment_name}.log`
- W&B offline runs: `wandb/offline-run-*` (synced per batch)

## Notes
- Defaults: `MODEL_SIZE=3B`, `MODEL_PATH=Qwen/Qwen2.5-3B-Instruct`, `CONFIG=_2_sokoban`.
- Collapse detection is enabled for both first- and multi-turn by default; the log suffix uses `ftmt` when both are on.
- `SKIP_STATUSES=("success")` means already-successful runs are skipped on resume.
- GPU list and parallelism are hard-coded (`GPUS=(0..7)`, `MAX_PARALLEL=8`); edit the script if your hardware differs.
