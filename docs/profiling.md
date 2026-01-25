# Profiling Runs (run_profiling.sh)

This doc explains how to run the profiling script (multi-GPU optional) and where to find outputs.

## What it does
- Launches a fixed grid of PPO/GRPO experiments (3 tasks × 4 algo/filter combos × think/no-think).
- Uses multiple GPUs in parallel; each run uses a GPU group (size configurable), and each group runs its own sequential queue.
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
- `--gpus` (optional, comma list like `0,1,2,3`; default auto-detect via `nvidia-smi -L`)
- `--gpus-per-exp` (optional, GPUs per experiment, default `1`)

Examples:
```bash
# 7B model, 64 samples, 2 steps
bash scripts/runs/run_profiling.sh --model_size 7B --samples 64 --steps 2

# 3B model, 64 samples, 10 steps
bash scripts/runs/run_profiling.sh --model_size 3B --samples 64 --steps 10

# 3B model, 64 samples, 10 steps, 2 GPUs per experiment (auto-detect GPUs)
bash scripts/runs/run_profiling.sh --model_size 3B --samples 64 --steps 10 --gpus-per-exp 2

# 3B model, fixed GPU list, 2 GPUs per experiment
bash scripts/runs/run_profiling.sh --model_size 3B --samples 64 --steps 10 --gpus 0,1,2,3 --gpus-per-exp 2
```

## Outputs
The script writes:
- Log file: `profiling_results_${MODEL_SIZE}_samples${SAMPLES}_${GPU}.log`
- Per-run results: `logs/profiling_results_${MODEL_SIZE}_samples${SAMPLES}_${GPU}/`

The log header includes:
```
GPU per exp: 2xH200 | Model Size: 3B | Steps: 1 | Collapse: first_turn=true, multi_turn=true, num_samples=128
GPUS: 0 1 2 3 | groups: 0,1 2,3 | cooldown=30s
```

At the end, a grouped summary is appended for easier scanning.

## Notes
- GPU model is detected via `nvidia-smi`. If mixed models are present, the log suffix uses `mixed`.
- If `--gpus` is not provided, the script auto-detects GPUs; if detection fails it falls back to `0-7`.
- `--gpus-per-exp` must evenly divide the GPU list length.
- To stop a running profiling session, kill the driver processes (see `scripts/runs/run_profiling.sh`).
