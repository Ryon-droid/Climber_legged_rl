#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/play.py \
  --checkpoint outputs/mujoco_runs/cad_velocity_round3_20261029_201813/final_policy.zip \
  --viewer --loop "$@"
