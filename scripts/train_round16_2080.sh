#!/usr/bin/env bash
set -euo pipefail
CLIMBER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CLIMBER_PYTHON="${CLIMBER_PYTHON:-/home/climber/.venvs/climber_rl_20261104/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$CLIMBER_ROOT"
stamp="$(date +%Y%m%d_%H%M%S)"
parent="outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip"
for trial in a b; do
    config="configs/robot/cad_balance_estimator_round16${trial}.yaml"
    run_name="cad_balance_estimator_round16${trial}_2080_${stamp}"
    SECONDS=0
    "$CLIMBER_PYTHON" -u scripts/train.py --backend cad-dynamics --config "$config" \
        --device cpu --num-envs 4 --total-timesteps 80000 \
        --warm-start "$parent" --run-name "$run_name"
    elapsed="$SECONDS"
    "$CLIMBER_PYTHON" scripts/verify_2080_training.py \
        --config "$config" --parent "$parent" --run-dir "outputs/mujoco_runs/$run_name" \
        --expected-steps 80000 --elapsed-seconds "$elapsed"
done
"$CLIMBER_PYTHON" -u scripts/assess_round16_2080.py --stamp "$stamp"
