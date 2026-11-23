#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
training_name="cad_getup_stand_round9_imitation_seed_$(date +%Y%m%d_%H%M%S)"
training_log="outputs/cad_getup_stand_round9_imitation/${training_name}.log"
mkdir -p outputs/cad_getup_stand_round9_imitation
initial_checkpoint="$(/home/climber/.venvs/climber_rl_20261104/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
import yaml
config=Path('configs/robot/cad_getup_stand_round9_imitation.yaml')
manifest=Path('outputs/cad_getup_stand_round9_imitation/imitation_latest.json')
initial=yaml.safe_load(config.read_text())['imitation']['initial_checkpoint']
if manifest.exists():
    report=json.loads(manifest.read_text())
    candidate=Path(report['checkpoint'])
    if (report['config_sha256']==hashlib.sha256(config.read_bytes()).hexdigest()
            and candidate.is_file() and report['checkpoint_sha256']==hashlib.sha256(candidate.read_bytes()).hexdigest()):
        initial=str(candidate)
print(initial)
PY
)"
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/climber/.venvs/climber_rl_20261104/bin/python -u scripts/distill_getup_policy.py \
  --config configs/robot/cad_getup_stand_round9_imitation.yaml \
  --run-name "$training_name" --initial-checkpoint "$initial_checkpoint" \
  --focus-offset-m 0.36 --focus-weight 2 "$@" 2>&1 | tee "$training_log"
