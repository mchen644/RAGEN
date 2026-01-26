# Main Table Runs

This doc covers the experiment scripts for the main performance table.

## Scripts Overview

| Script | Purpose | Variables |
|--------|---------|-----------|
| `run_main_table_diff_algo.sh` | Compare algorithms | PPO, DAPO, GRPO, DrGRPO |
| `run_main_table_diff_size.sh` | Compare model sizes | 0.5B, 1.5B, 3B, 7B |
| `run_main_table_diff_model.sh` | Compare model types | Instruct, Reasoning |

All scripts run experiments across 5 tasks (sokoban, frozenlake, webshop, metamathqa, countdown) with filter/nofilter settings.

---

## 1. Different Algorithms (`run_main_table_diff_algo.sh`)

Compares PPO/DAPO/GRPO/DrGRPO using Qwen2.5-3B.

```bash
bash scripts/runs/run_main_table_diff_algo.sh --steps 400
```

Options:
- `--steps` (default: `400`)
- `--tasks` (comma list; default: `sokoban,frozenlake,webshop,metamathqa,countdown`)
- `--algos` (comma list; default: `PPO,DAPO,GRPO,DrGRPO`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `1`)
- `--cooldown` (seconds; default: `30`)
- `--gpu-memory-utilization` (default: `0.3`)

Examples:
```bash
# Run only PPO and GRPO
bash scripts/runs/run_main_table_diff_algo.sh --algos PPO,GRPO

# Single task, quick sanity
bash scripts/runs/run_main_table_diff_algo.sh --steps 5 --tasks sokoban --algos PPO
```

Outputs:
- Per-task logs: `logs/diff_algo_<task>_Qwen2.5-3B/`
- Summary log: `logs/diff_algo_Qwen2.5-3B.log`

---

## 2. Different Model Sizes (`run_main_table_diff_size.sh`)

Compares Qwen2.5 models of different sizes using PPO.

```bash
bash scripts/runs/run_main_table_diff_size.sh --steps 400
```

Options:
- `--steps` (default: `400`)
- `--tasks` (comma list; default: `sokoban,frozenlake,webshop,metamathqa,countdown`)
- `--models` (comma list; default: `Qwen2.5-0.5B,Qwen2.5-1.5B,Qwen2.5-3B,Qwen2.5-7B`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `1`)
- `--cooldown` (seconds; default: `30`)
- `--gpu-memory-utilization` (default: `0.3`)

Examples:
```bash
# Run only 3B and 7B
bash scripts/runs/run_main_table_diff_size.sh --models Qwen2.5-3B,Qwen2.5-7B

# Quick test with smallest model
bash scripts/runs/run_main_table_diff_size.sh --steps 5 --models Qwen2.5-0.5B --tasks sokoban
```

Outputs:
- Per-task logs: `logs/diff_size_<task>/`
- Summary log: `logs/diff_size_PPO.log`

---

## 3. Different Model Types (`run_main_table_diff_model.sh`)

Compares different model types (Instruct, Reasoning) using PPO.

```bash
bash scripts/runs/run_main_table_diff_model.sh --steps 400
```

Options:
- `--steps` (default: `400`)
- `--tasks` (comma list; default: `sokoban,frozenlake,webshop,metamathqa,countdown`)
- `--models` (comma list; default: `Qwen2.5-3B-Instruct`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `1`)
- `--cooldown` (seconds; default: `30`)
- `--gpu-memory-utilization` (default: `0.3`)

Examples:
```bash
# Run with specific models
bash scripts/runs/run_main_table_diff_model.sh --models Qwen2.5-3B-Instruct

# Quick test
bash scripts/runs/run_main_table_diff_model.sh --steps 5 --tasks sokoban
```

Outputs:
- Per-task logs: `logs/diff_model_<task>/`
- Summary log: `logs/diff_model_PPO.log`

---

## Common Notes

- Filter settings: `filter` uses `top_p=0.9`, `nofilter` uses `top_p=1.0`
- All experiments log to wandb with project names like `ragen_multi_gpu_test`, `ragen_model_size`, `ragen_model_type`
- Use `--gpus-per-exp` for multi-GPU experiments (GPU count must be divisible by this value)
