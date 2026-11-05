#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
bash scripts/train_imitation_getup.sh \
  --updates 4000 --dagger-rounds 4 --dagger-updates 2000 \
  --learning-rate 0.00005 --focus-offset-m 0.37 --focus-weight 1.3 \
  --selection-horizon 20 --selection-standing-episodes 5 "$@"
