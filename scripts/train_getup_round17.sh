#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
training_name="cad_getup_stand_round17_light_push_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs/cad_getup_stand_round17_light_push
/root/gpufree-data/.venvs/climber-validation/bin/python -u scripts/train.py \
  --backend cad-dynamics --config configs/robot/cad_getup_stand_round17_light_push.yaml \
  --warm-start outputs/mujoco_runs/cad_getup_stand_round16_hard_standing_20261031_145045/best/best_model.zip \
  --device cpu --run-name "$training_name" "$@" 2>&1 | tee "outputs/cad_getup_stand_round17_light_push/${training_name}.log"
