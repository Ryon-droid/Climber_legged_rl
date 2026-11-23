#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
training_name="cad_getup_stand_round10_ppo_refine_$(date +%Y%m%d_%H%M%S)"
training_log="outputs/cad_getup_stand_round10_ppo_refine/${training_name}.log"
mkdir -p outputs/cad_getup_stand_round10_ppo_refine
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/climber/.venvs/climber_rl_20261104/bin/python -u scripts/train.py \
  --backend cad-dynamics --config configs/robot/cad_getup_stand_round10_ppo_refine.yaml \
  --warm-start outputs/mujoco_runs/cad_getup_stand_round9_imitation_seed_20261029_175259/initial_policy.zip \
  --device cpu --run-name "$training_name" "$@" 2>&1 | tee "$training_log"
