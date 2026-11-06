"""Prepare round 15 from an unchanged, verified round-14 physical model."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'scripts')]
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from climber_legged_rl.cad_dynamics import validate_artifacts, input_hashes
from climber_legged_rl.getup import prepare_pose_bank
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_training import validate_policy_transfer
from climber_legged_rl.standing_rehearsal import load_standing_rehearsal
from smoke_mujoco import run as smoke


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_cfg = ROOT/'configs/robot/cad_balance_estimator_round14.yaml'
    cfg = ROOT/'configs/robot/cad_balance_estimator_round15.yaml'
    old = load_config(old_cfg)
    config = load_config(cfg)
    for key in old.keys() | config.keys():
        if key not in ('training', 'experiment'):
            assert old.get(key) == config.get(key), key
    old_experiment = dict(old['experiment'])
    new_experiment = dict(config['experiment'])
    old_experiment.pop('output_directory')
    new_experiment.pop('output_directory')
    assert old_experiment == new_experiment
    report, _ = validate_artifacts(old_cfg)
    source_folder = ROOT/old['experiment']['output_directory']
    folder = ROOT/config['experiment']['output_directory']
    folder.mkdir(exist_ok=False)
    xml = folder/'robot_experiment.xml'
    xml.write_bytes((source_folder/'robot_experiment.xml').read_bytes())
    report.update(generated_model=str(xml), config_sha256=sha(cfg), input_sha256=input_hashes(cfg))
    report['round15_provenance'] = dict(source_config=str(old_cfg), source_report_sha256=sha(source_folder/'model_report.json'),
        source_model_sha256=sha(source_folder/'robot_experiment.xml'),
        physical_model_and_nontraining_settings_identical=True,
        note='Reuse the verified immutable physical model; only training sampler/rehearsal settings and seed change.')
    (folder/'model_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    prepare_pose_bank(cfg)
    smoke(cfg, steps=1000)
    validate_artifacts(cfg)
    checkpoint = ROOT/'outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip'
    validate_policy_transfer(checkpoint, xml, config, warm_start=True)
    load_standing_rehearsal(config['training']['standing_rehearsal'], config, xml)
    print('ROUND15_MODEL_PARENT_AND_REHEARSAL_VALIDATED', flush=True)


if __name__ == '__main__':
    main()
