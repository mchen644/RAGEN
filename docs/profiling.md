# Profiling Runs (run_profiling.sh)

This doc explains how to run the profiling script (one GPU per run, multiple GPUs in parallel) and where to find outputs.

## What it does
- Launches a fixed grid of PPO/GRPO experiments (3 tasks × 4 algo/filter combos × think/no-think).
- Uses multiple GPUs in parallel; each run uses a single GPU, and each GPU runs its own sequential queue.
- Inserts a fixed 30s cooldown between tasks on the same GPU.

## How to run
From repo root:

```bash
bash scripts/runs/run_profiling.sh --model_size 3B --samples 128 --steps 1
```

Arguments:
- `--model_size` (required, no default)
- `--samples` (collapse sample count, default `128`)
- `--steps` (training steps per run, default `1`)

Examples:
```bash
# 7B model, 64 samples, 2 steps
bash scripts/runs/run_profiling.sh --model_size 7B --samples 64 --steps 2

# 3B model, 16 samples
bash scripts/runs/run_profiling.sh --model_size 3B --samples 16
```

## Outputs
The script writes:
- Log file: `profiling_results_${MODEL_SIZE}_samples${SAMPLES}_${GPU}.log`
- Per-run results: `logs/profiling_results_${MODEL_SIZE}_samples${SAMPLES}_${GPU}/`

The log header includes:
```
GPU: 1xH200 | Model Size: 3B | Steps: 1 | Collapse: first_turn=true, multi_turn=true, num_samples=128
```

At the end, a grouped summary is appended for easier scanning.

## Notes
- GPU model is detected via `nvidia-smi`. If mixed models are present, the log suffix uses `mixed`.
- To stop a running profiling session, kill the driver processes (see `scripts/runs/run_profiling.sh`).
