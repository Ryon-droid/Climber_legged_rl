#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
training_name="cad_getup_stand_round14_replay_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs/cad_getup_stand_round14_replay
/home/climber/.venvs/climber_rl_20261104/bin/python -u scripts/train.py \
  --backend cad-dynamics --config configs/robot/cad_getup_stand_round14_replay.yaml \
  --warm-start outputs/mujoco_runs/cad_getup_stand_round12_max_spring_20261029_224155/best/best_model.zip \
  --device cpu --run-name "$training_name" "$@" 2>&1 | tee "outputs/cad_getup_stand_round14_replay/${training_name}.log"
