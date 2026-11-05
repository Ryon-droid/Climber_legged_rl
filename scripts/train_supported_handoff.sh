#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
training_name="cad_getup_stand_round7_handoff_$(date +%Y%m%d_%H%M%S)"
training_log="outputs/cad_getup_stand_round7_handoff/${training_name}.log"
mkdir -p outputs/cad_getup_stand_round7_handoff
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/gpufree-data/.venvs/climber-validation/bin/python -u scripts/train.py \
  --backend cad-dynamics \
  --config configs/robot/cad_getup_stand_round7_handoff.yaml \
  --warm-start outputs/mujoco_runs/cad_getup_stand_round5_dynamic_20261029/best/best_model.zip \
  --device cpu --run-name "$training_name" "$@" 2>&1 | tee "$training_log"
