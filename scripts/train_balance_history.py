"""Prepare and train a history standing comparison from a reviewed checkpoint."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['single', 'history', 'recovery', 'robust',
        'estimator', 'estimator_round2', 'estimator_round3', 'estimator_round4',
        'estimator_round5', 'estimator_round6', 'estimator_round7', 'estimator_round8', 'estimator_round9', 'estimator_round10', 'estimator_round11', 'estimator_round12', 'estimator_round13', 'estimator_round14',
        'gpu', 'gpu_estimator'], default='history')
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--total-timesteps', type=int)
    parser.add_argument('--num-envs', type=int)
    args = parser.parse_args()
    if args.parent is None:
        parents = {
            'estimator_round2': 'outputs/mujoco_runs/cad_balance_estimator_20261031_200736/best/best_model.zip',
            'estimator_round3': 'outputs/mujoco_runs/cad_balance_estimator_round2_20261031_203153/best/best_model.zip',
            'estimator_round4': 'outputs/mujoco_runs/cad_balance_estimator_round3_20261031_205118/best/best_model.zip',
            'estimator_round5': 'outputs/mujoco_runs/cad_balance_estimator_round4_20261031_210723/best/best_model.zip',
            'estimator_round6': 'outputs/mujoco_runs/cad_balance_estimator_round5_20261101_175824/best/best_model.zip',
            'estimator_round7': 'outputs/mujoco_runs/cad_balance_estimator_round6_20261101_184802/best/best_model.zip',
            'estimator_round8': 'outputs/mujoco_runs/cad_balance_estimator_round7_20261101_192356/best/best_model.zip',
            'estimator_round9': 'outputs/mujoco_runs/cad_balance_estimator_round8_20261101_194643/best/best_model.zip',
            'estimator_round10': 'outputs/mujoco_runs/cad_balance_estimator_round9_20261101_201637/best/best_model.zip',
            'estimator_round11': 'outputs/mujoco_runs/cad_balance_estimator_round10_20261101_204316/best/best_model.zip',
            'estimator_round12': 'outputs/mujoco_runs/cad_balance_estimator_round11_20261101_211204/best/best_model.zip',
            'estimator_round13': 'outputs/mujoco_runs/cad_balance_estimator_round12_20261101_214546/checkpoints/climber_200000_steps.zip',
            'estimator_round14': 'outputs/mujoco_runs/cad_balance_estimator_round13_20261101_221330/best/best_model.zip'}
        if args.variant in parents:
            args.parent = ROOT/parents[args.variant]
        elif 'estimator' in args.variant:
            args.parent = ROOT/'outputs/cad_balance_estimator/pretrained/pretrained_policy.zip'
        else:
            args.parent = ROOT/'outputs/mujoco_runs/cad_getup_stand_round22_guarded_balance_20261031_183529/best/best_model.zip'
    if not args.parent.is_file():
        parser.error('Parent checkpoint does not exist')
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            parts = [Path(p.decode(errors='replace')).name for p in entry.read_bytes().split(b'\0') if p]
        except OSError:
            continue
        if entry.parent.name == str(os.getpid()):
            continue
        if any(name in ('train.py', 'train_distance_curriculum.py', 'train_balance_history.py') for name in parts):
            parser.error('Another training process is active')
    from climber_legged_rl.cad_dynamics import build_dynamics
    from climber_legged_rl.getup import prepare_pose_bank
    from climber_legged_rl.mujoco_training import train
    from smoke_mujoco import run as smoke
    config = ROOT / f'configs/robot/cad_balance_{args.variant}.yaml'
    build_dynamics(config)
    prepare_pose_bank(config)
    smoke(config, steps=1000)
    if 'gpu' in args.variant:
        from climber_legged_rl.warp_backend import validate_gpu_report
        validate_gpu_report(config)
    train(config, total_steps=args.total_timesteps, num_envs=args.num_envs, device='cpu',
        run_name=f'cad_balance_{args.variant}_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
        warm_start=args.parent)


if __name__ == '__main__':
    main()
