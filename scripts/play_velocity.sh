#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Resolve only real movement runs, never validation or get-up policies.
checkpoint="$(/home/climber/.venvs/climber_rl_20261104/bin/python - <<'PY'
from pathlib import Path
import json
folders=sorted(Path('outputs/mujoco_runs').glob('cad_velocity_round*_*'),key=lambda p:p.stat().st_mtime,reverse=True)
for folder in folders:
    path=folder/'best/best_model.zip'
    metadata=folder/'metadata.json'
    if path.is_file() and metadata.is_file() and json.loads(metadata.read_text()).get('requested_timesteps',0)>=500000:
        print(path)
        break
else:
    raise SystemExit('暂无正式移动训练的best模型。先运行 bash scripts/train_velocity.sh；不把站立或短验证权重当作移动模型。')
PY
)"
/home/climber/.venvs/climber_rl_20261104/bin/python scripts/play.py \
  --checkpoint "$checkpoint" --viewer --loop "$@"
