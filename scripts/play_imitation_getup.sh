#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
checkpoint_path="$(/home/climber/.venvs/climber_rl_20261104/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
config=Path('configs/robot/cad_getup_stand_round9_imitation.yaml')
report=json.loads(Path('outputs/cad_getup_stand_round9_imitation/imitation_latest.json').read_text())
if report['config_sha256'] != hashlib.sha256(config.read_bytes()).hexdigest():
    raise SystemExit('Imitation config changed; rerun train_imitation_getup.sh before playback')
checkpoint=Path(report['checkpoint'])
if report['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
    raise SystemExit('Imitation checkpoint hash does not match its training report')
print(checkpoint)
PY
)"
exec env MUJOCO_GL=glfw OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/climber/.venvs/climber_rl_20261104/bin/python -u scripts/play_getup.py \
  --config configs/robot/cad_getup_stand_round9_imitation.yaml \
  --checkpoint "$checkpoint_path" --reset-mode folded --loop "$@"
