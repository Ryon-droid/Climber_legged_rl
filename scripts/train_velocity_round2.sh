#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
training_name="cad_velocity_round2_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs/cad_velocity_round2
/home/climber/.venvs/climber_rl_20261104/bin/python -u scripts/train.py \
  --backend cad-dynamics --config configs/robot/cad_velocity_round2.yaml \
  --warm-start outputs/mujoco_runs/cad_velocity_round1_20261029_185422/best/best_model.zip \
  --device cpu --run-name "$training_name" "$@" 2>&1 | tee "outputs/cad_velocity_round2/${training_name}.log"
