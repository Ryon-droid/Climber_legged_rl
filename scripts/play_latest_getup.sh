#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/gpufree-data/.venvs/climber-validation/bin/python scripts/play_getup.py \
  --config configs/robot/cad_getup_stand_round10_ppo_refine.yaml \
  --checkpoint outputs/mujoco_runs/cad_getup_stand_round10_ppo_refine_20261029_180330/best/best_model.zip \
  --reset-mode folded --loop "$@"
