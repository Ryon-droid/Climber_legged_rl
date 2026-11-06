#!/usr/bin/env bash
set -euo pipefail
CLIMBER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CLIMBER_PYTHON="${CLIMBER_PYTHON:-/home/climber/.venvs/climber_rl_20261104/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$CLIMBER_ROOT"
stamp="$(date +%Y%m%d_%H%M%S)"
echo "ROUND18_STAMP=$stamp"
"$CLIMBER_PYTHON" -u scripts/train_round18_warmup.py --stamp "$stamp"
"$CLIMBER_PYTHON" -u scripts/gate_round18_warmup.py --stamp "$stamp"
trained=0
for trial in a b; do
    if ! "$CLIMBER_PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if sys.argv[2] in r["eligible_trials"] else 1)' \
        "outputs/round18_warmup_${stamp}/gate.json" "$trial"; then
        echo "SKIP_PPO_TRIAL=$trial; warmup gate did not pass"
        continue
    fi
    config="configs/robot/cad_balance_estimator_round18${trial}.yaml"
    parent="outputs/mujoco_runs/cad_balance_estimator_round18${trial}_warmup_${stamp}/final_policy.zip"
    run_name="cad_balance_estimator_round18${trial}_2080_${stamp}"
    SECONDS=0
    "$CLIMBER_PYTHON" -u scripts/train.py --backend cad-dynamics --config "$config" \
        --device cpu --num-envs 4 --total-timesteps 40000 \
        --warm-start "$parent" --run-name "$run_name"
    elapsed="$SECONDS"
    "$CLIMBER_PYTHON" scripts/verify_2080_training.py \
        --config "$config" --parent "$parent" --run-dir "outputs/mujoco_runs/$run_name" \
        --expected-steps 40000 --elapsed-seconds "$elapsed"
    trained=$((trained+1))
done
if [ "$trained" -gt 0 ]; then
    "$CLIMBER_PYTHON" -u scripts/assess_round18_2080.py --stamp "$stamp"
else
    echo 'ROUND18_STOPPED_AT_WARMUP_GATE; baseline preserved; no PPO started'
fi
echo 'ROUND18_WORKFLOW_FINISHED'
