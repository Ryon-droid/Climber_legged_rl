#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec env MUJOCO_GL=glfw OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/gpufree-data/.venvs/climber-validation/bin/python scripts/play_getup.py \
  --config configs/robot/cad_getup_stand_round6_support.yaml \
  --checkpoint outputs/mujoco_runs/cad_getup_stand_round5_dynamic_20261029/best/best_model.zip \
  --reset-mode folded --loop "$@"
