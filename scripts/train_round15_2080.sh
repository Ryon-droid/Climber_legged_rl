#!/usr/bin/env bash
set -euo pipefail
CLIMBER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CLIMBER_PYTHON="${CLIMBER_PYTHON:-/home/climber/.venvs/climber_rl_20261104/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$CLIMBER_ROOT"
steps="${1:-240000}"
envs="${2:-4}"
run_name="cad_balance_estimator_round15_2080_$(date +%Y%m%d_%H%M%S)"
config="configs/robot/cad_balance_estimator_round15.yaml"
parent="outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip"
"$CLIMBER_PYTHON" scripts/prepare_round15_2080.py
SECONDS=0
"$CLIMBER_PYTHON" -u scripts/train.py --backend cad-dynamics --config "$config" \
    --device cpu --num-envs "$envs" --total-timesteps "$steps" \
    --warm-start "$parent" --run-name "$run_name"
elapsed="$SECONDS"
"$CLIMBER_PYTHON" scripts/verify_2080_training.py \
    --config "$config" --parent "$parent" \
    --run-dir "outputs/mujoco_runs/$run_name" --expected-steps "$steps" \
    --elapsed-seconds "$elapsed"
