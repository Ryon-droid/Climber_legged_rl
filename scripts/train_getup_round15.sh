#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
training_name="cad_getup_stand_round15_hold_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs/cad_getup_stand_round15_hold
/root/gpufree-data/.venvs/climber-validation/bin/python -u scripts/train.py \
  --backend cad-dynamics --config configs/robot/cad_getup_stand_round15_hold.yaml \
  --warm-start outputs/mujoco_runs/cad_getup_stand_round14_replay_20261029_234655/best/best_model.zip \
  --device cpu --run-name "$training_name" "$@" 2>&1 | tee "outputs/cad_getup_stand_round15_hold/${training_name}.log"
