#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
training_name="cad_velocity_round4_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs/cad_velocity_round4
/home/climber/.venvs/climber_rl_20261104/bin/python -u scripts/train.py \
  --backend cad-dynamics --config configs/robot/cad_velocity_round4.yaml \
  --warm-start outputs/mujoco_runs/cad_velocity_round3_20261029_201813/final_policy.zip \
  --device cpu --run-name "$training_name" "$@" 2>&1 | tee "outputs/cad_velocity_round4/${training_name}.log"
