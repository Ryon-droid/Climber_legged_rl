"""Build the measured 208 mm spring-stop model and belly-down reset without a parent policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'scripts')]
from climber_legged_rl.cad_dynamics import build_dynamics, validate_artifacts
from climber_legged_rl.getup import prepare_pose_bank
from climber_legged_rl.mujoco_model import load_config
from smoke_mujoco import run as smoke

CONFIG = ROOT/'configs/robot/cad_balance_spring208_coldstart.yaml'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--parent', type=Path, help='Optional parent checkpoint for transfer validation')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    source = ROOT/config['experiment']['source_model']
    if not source.is_file():
        raise SystemExit(f'Missing repaired CAD model {source}; run scripts/repair_cad_model.py first')
    xml, generated = build_dynamics(config_path)
    report = dict(simulation_only=True, prepared=True, cold_start=True,
                  config=str(config_path), config_sha256=sha(config_path),
                  source_model=str(source), source_model_sha256=sha(source),
                  model_sha256=sha(xml), total_mass_kg=generated['total_mass_kg'],
                  spring_travel=generated['spring_travel'], parent_checkpoint=None,
                  training_started=False)
    if config.get('getup', {}).get('enabled'):
        poses = prepare_pose_bank(config_path)
        report['new_pose'] = poses['poses'][0]
    if args.parent:
        from climber_legged_rl.mujoco_training import validate_policy_transfer
        validate_policy_transfer(args.parent, xml, config, warm_start=True)
        report['parent_checkpoint'] = str(args.parent)
        report['parent_checkpoint_sha256'] = sha(args.parent)
    smoke(config_path, steps=1000)
    validate_artifacts(config_path)
    report['smoke_passed'] = True
    report['strict_warm_start_validated'] = bool(args.parent)
    report['note'] = 'Simulation-only cold start; no hardware readiness is implied.'
    (xml.parent/'spring208_preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    summary = dict(prepared=True, model=str(xml), total_mass_kg=report['total_mass_kg'],
                   smoke_passed=True)
    if 'new_pose' in report:
        pose = report['new_pose']
        summary.update(rearward_offset_m=pose['rearward_offset_m'],
                       spring_eye_distances_m=pose['spring_eye_distances_m'],
                       closure_error_m=pose['closure_error_m'],
                       base_bottom_m=pose['base_bottom_m'], wheel_bottom_m=pose['wheel_bottom_m'])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
