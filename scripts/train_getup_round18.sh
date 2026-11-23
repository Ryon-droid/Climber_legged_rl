#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
parent="outputs/mujoco_runs/cad_getup_stand_round17_light_push_20261031_152823"
/home/climber/.venvs/climber_rl_20261104/bin/python - "$parent" <<'CHECK_PARENT'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if json.loads((p/'status.json').read_text()).get('state')!='completed':
    raise SystemExit('第17轮尚未完成，请结束后再运行；当前训练不会被中断。')
if not (p/'best/best_model.zip').is_file():
    raise SystemExit('第17轮best不存在，不能开始第18轮。')
for run in Path('outputs/mujoco_runs').glob('cad_getup_stand_round18_multidirection_*'):
    metadata=run/'metadata.json'
    if not metadata.exists():
        continue
    pid=json.loads(metadata.read_text()).get('pid')
    process=Path('/proc')/str(pid)/'cmdline'
    try:
        command=process.read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (FileNotFoundError,ProcessLookupError):
        continue
    if 'scripts/train.py' in command and run.name in command:
        raise SystemExit('第18轮已在运行，避免重复启动：'+str(run))
CHECK_PARENT
training_name="cad_getup_stand_round18_multidirection_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs/cad_getup_stand_round18_multidirection
/home/climber/.venvs/climber_rl_20261104/bin/python -u scripts/train.py \
  --backend cad-dynamics --config configs/robot/cad_getup_stand_round18_multidirection.yaml \
  --warm-start "$parent/best/best_model.zip" \
  --device cpu --run-name "$training_name" "$@" 2>&1 | tee "outputs/cad_getup_stand_round18_multidirection/${training_name}.log"
