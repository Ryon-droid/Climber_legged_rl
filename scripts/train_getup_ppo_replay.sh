#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ " ${*} " != *" --help "* && " ${*} " != *" -h "* ]]; then
  /home/climber/.venvs/climber_rl_20261104/bin/python - <<'PY'
from pathlib import Path
import yaml
config = yaml.safe_load(Path('configs/robot/cad_getup_stand_round11_ppo_replay.yaml').read_text())
offsets = config['getup']['rearward_offsets_m']
if len(offsets) != 1:
    raise SystemExit('未启动训练：用户已确认实机起身只有一个固定趴姿。当前配置仍是三个历史仿真初态，请先确认固定趴姿尺寸，并统一更新初态、示范数据和模型检查；不能任选一个旧数值代替实测。')
PY
fi
training_name="cad_getup_stand_round11_ppo_replay_$(date +%Y%m%d_%H%M%S)"
training_log="outputs/cad_getup_stand_round11_ppo_replay/${training_name}.log"
mkdir -p outputs/cad_getup_stand_round11_ppo_replay
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/climber/.venvs/climber_rl_20261104/bin/python -u scripts/train.py \
  --backend cad-dynamics --config configs/robot/cad_getup_stand_round11_ppo_replay.yaml \
  --warm-start outputs/mujoco_runs/cad_getup_stand_round10_ppo_refine_20261029_180330/best/best_model.zip \
  --device cpu --run-name "$training_name" "$@" 2>&1 | tee "$training_log"
