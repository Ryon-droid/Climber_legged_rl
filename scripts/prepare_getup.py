"""Prepare a combined-task model and a validated closed-chain folded reset bank."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from climber_legged_rl.cad_dynamics import build_dynamics
from climber_legged_rl.getup import prepare_pose_bank

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/robot/cad_getup_stand_push.yaml'))
    args = parser.parse_args()
    build_dynamics(args.config)
    print(json.dumps(prepare_pose_bank(args.config), ensure_ascii=False, indent=2))
    from climber_legged_rl.getup_curriculum import prepare_curriculum_bank
    curriculum=prepare_curriculum_bank(args.config)
    if curriculum is not None:
        print(json.dumps(curriculum,indent=2))
