#!/bin/bash
# 3B
bash scripts/runs/run_profiling.sh --model_size 3B --samples 32 --steps 10
bash scripts/runs/run_profiling.sh --model_size 3B --samples 64 --steps 10
bash scripts/runs/run_profiling.sh --model_size 3B --samples 128 --steps 10

# 1.5B
bash scripts/runs/run_profiling.sh --model_size 1.5B --samples 32 --steps 10
bash scripts/runs/run_profiling.sh --model_size 1.5B --samples 64 --steps 10
bash scripts/runs/run_profiling.sh --model_size 1.5B --samples 128 --steps 10

# 7B
bash scripts/runs/run_profiling.sh --model_size 7B --samples 32 --steps 10
bash scripts/runs/run_profiling.sh --model_size 7B --samples 64 --steps 10
bash scripts/runs/run_profiling.sh --model_size 7B --samples 128 --steps 10
