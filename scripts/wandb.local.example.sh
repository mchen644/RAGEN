#!/usr/bin/env bash
# Project-local W&B config template.
# Copy this file to scripts/wandb.local.sh, fill values, then run:
#   source scripts/wandb.local.sh

export WANDB_API_KEY="replace_with_your_wandb_api_key"
export WANDB_ENTITY="replace_with_your_wandb_team_name"  # Must be a team, not personal entity
export WANDB_PROJECT="ragen_profiling"
