#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/climber/.venvs/climber_rl_20261104/bin/python scripts/play_getup.py \
  --config configs/robot/cad_getup_stand_round12_max_spring.yaml \
  --checkpoint outputs/mujoco_runs/cad_getup_stand_round12_max_spring_20261029_224155/best/best_model.zip \
  --reset-mode folded --loop "$@"
