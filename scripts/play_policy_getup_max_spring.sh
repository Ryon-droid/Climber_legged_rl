#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /root/gpufree-data/.venvs/climber-validation/bin/python scripts/play_getup.py \
  --config configs/robot/cad_getup_stand_round12_max_spring.yaml \
  --checkpoint outputs/mujoco_runs/cad_getup_stand_round9_imitation_seed_20261029_175259/stage_2.zip \
  --reset-mode folded --loop "$@"
