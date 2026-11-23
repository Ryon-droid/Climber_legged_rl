#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
parent="outputs/mujoco_runs/cad_getup_stand_round20_tight_20261031_164134"
/home/climber/.venvs/climber_rl_20261104/bin/python - "$parent" <<'CHECK'
from pathlib import Path
import json,sys
p=Path(sys.argv[1])
if json.loads((p/"status.json").read_text()).get("state")!="completed":
    raise SystemExit("Parent round20 has not completed")
if not (p/"best/best_model.zip").is_file():
    raise SystemExit("Parent best checkpoint missing")
for entry in Path("/proc").glob("[0-9]*/cmdline"):
    try: parts=entry.read_bytes().split(bytes([0]))
    except OSError: continue
    if any(Path(x.decode(errors="replace")).name in ("train.py","train_distance_curriculum.py") for x in parts if x):
        raise SystemExit("Training is already active; do not start another run")
CHECK
training_name="cad_getup_stand_round21_hard_standing_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs/cad_getup_stand_round21_hard_standing
/home/climber/.venvs/climber_rl_20261104/bin/python -u scripts/train.py \
  --backend cad-dynamics --config configs/robot/cad_getup_stand_round21_hard_standing.yaml \
  --warm-start "$parent/best/best_model.zip" --device cpu --run-name "$training_name" "$@" \
  2>&1 | tee "outputs/cad_getup_stand_round21_hard_standing/${training_name}.log"
