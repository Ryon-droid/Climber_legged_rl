#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/climber/.venvs/climber_rl_20261104/bin/python scripts/play.py \
  --checkpoint outputs/mujoco_runs/cad_velocity_round1_20261029_185422/best/best_model.zip \
  --viewer --loop "$@"
