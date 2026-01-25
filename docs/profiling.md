# Profiling Runs

This doc explains how to run `scripts/runs/run_profiling.sh` directly and how to use `run_all_profiling.sh` to iterate over the most common sample sizes.

## `scripts/runs/run_profiling.sh`

- Launches the fixed grid of PPO/GRPO experiments (3 tasks × 4 algo/filter combos × think/no-think) for a single model size.
- Auto-detects GPUs (falls back to `0-7` if detection fails), groups them into slots, and runs one sequential queue per slot with a 30 s cooldown between runs on the same GPU.
- Each run logs the GPU group, collapse settings, and a grouped summary at the end; per-run logs go into `logs/profiling_<GPU_LABEL>_details/`.

### How to run
From the repo root:

```bash
bash scripts/runs/run_profiling.sh --model_size 3B --samples 128 --steps 1
```

Arguments:
- `--model_size` (required, no default)
- `--samples` (collapse sample count, default `128`)
- `--steps` (training steps per run, default `1`)
- `--gpus` (optional, comma list like `0,1,2,3`; auto-detected via `nvidia-smi -L` if omitted)
- `--gpus-per-exp` (optional, GPUs per experiment, default `1`; must evenly divide the GPU list length)

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

### Outputs
- Log file: `logs/profiling_<GPU_LABEL>/profiling_results_${MODEL_SIZE}_samples${SAMPLES}_${GPU_LABEL}.log`
- Per-run detail logs: `logs/profiling_<GPU_LABEL>_details/<task-algo-filter-model-thinking>.log`
- Summary entries written to `logs/profiling_<GPU_LABEL>/profiling_results_${MODEL_SIZE}_samples${SAMPLES}_${GPU_LABEL}.log` at the end, one line per experiment.

## `run_all_profiling.sh`

- Wrapper that sequentially invokes `scripts/runs/run_profiling.sh` for each of the active sample counts (currently 32, 64, 128) using the 1.5 B/3 B/7 B models. 
- Accepts `--gpus-per-exp` (default `1`) and forwards it to every profiling invocation, ensuring consistent GPU grouping across the suite.
- Runs the profiling script in the current shell; killing `run_all_profiling.sh` stops launching new experiments, but any already-running `train.py`/Ray workers must be killed separately (see notes below).

### Usage

```bash
bash run_all_profiling.sh --gpus-per-exp 2
```

The script currently runs the 1.5 B/3 B/7 B models configurations (32/64/128 samples). To include other model sizes or sample counts, uncomment or add additional `bash scripts/runs/run_profiling.sh …` lines within `run_all_profiling.sh`.

## Notes
- GPU detection uses `nvidia-smi`; mixed GPU pools tag the logs with `mixed`.
- To abort any profiling session early, kill the `run_profiling.sh`/`run_all_profiling.sh` process and any child `python train.py …` jobs. Ray also spawns worker processes (e.g., `ray::WorkerDict.*`), which you can stop by killing the associated `python train.py` owner or using `ray stop`.
- Logs and per-run detail directories live under `logs/profiling_<GPU_LABEL>/` and `logs/profiling_<GPU_LABEL>_details/`.
