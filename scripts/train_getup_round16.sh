#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
training_name="cad_getup_stand_round16_hard_standing_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs/cad_getup_stand_round16_hard_standing
/home/climber/.venvs/climber_rl_20261104/bin/python -u scripts/train.py \
  --backend cad-dynamics --config configs/robot/cad_getup_stand_round16_hard_standing.yaml \
  --warm-start outputs/mujoco_runs/cad_getup_stand_round15_hold_20261030_001841/best/best_model.zip \
  --device cpu --run-name "$training_name" "$@" 2>&1 | tee "outputs/cad_getup_stand_round16_hard_standing/${training_name}.log"
