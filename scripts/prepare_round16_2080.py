"""Prepare paired round-16 trials without rebuilding the validated CAD dynamics."""
import argparse
import copy
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
from collect_standing_rehearsal import collect
from smoke_mujoco import run as smoke


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_cfg = ROOT/'configs/robot/cad_balance_estimator_round15.yaml'
    old = load_config(old_cfg)
    source = ROOT/old['experiment']['output_directory']
    original_report, _ = validate_artifacts(old_cfg)
    parent = ROOT/'outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip'
    configurations = []
    for trial in ('a', 'b'):
        cfg = ROOT/f'configs/robot/cad_balance_estimator_round16{trial}.yaml'
        config = load_config(cfg)
        for key in old.keys() | config.keys():
            if key not in ('training', 'experiment', 'balance_recovery'):
                assert old.get(key) == config.get(key), key
        expected = copy.deepcopy(old['balance_recovery'])
        if trial == 'b':
            expected['heading_weight'] = 2.0
        assert expected == config['balance_recovery']
        ex = dict(config['experiment']); ex['output_directory'] = old['experiment']['output_directory']
        assert ex == old['experiment']
        folder = ROOT/config['experiment']['output_directory']
        folder.mkdir(exist_ok=True)
        xml = folder/'robot_experiment.xml'
        if xml.exists():
            assert sha(xml) == sha(source/'robot_experiment.xml')
        else:
            xml.write_bytes((source/'robot_experiment.xml').read_bytes())
        prepare_pose_bank(cfg)
        validate_policy_transfer(parent, xml, config, warm_start=True)
        if not (folder/'rehearsal').exists():
            teachers = json.loads((ROOT/'outputs/cad_balance_estimator_round13/rehearsal/standing.json').read_text())['checkpoints']
            checkpoints = {}
            for name, entry in teachers.items():
                path = ROOT/entry['path']
                assert sha(path) == entry['sha256'], name
                checkpoints[name] = path
            collect(cfg, checkpoints['stable'], checkpoints['recovery'], folder/'rehearsal',
                    initial_seconds=4.0, push_tail_seconds=4.0)
        report = copy.deepcopy(original_report)
        report.update(generated_model=str(xml), config_sha256=sha(cfg), input_sha256=input_hashes(cfg))
        report['round16_provenance'] = dict(source_config=str(old_cfg), source_model_sha256=sha(xml),
            physical_model_identical=True, trial=trial, parent_checkpoint=str(parent),
            parent_sha256=sha(parent), changed_reward='balance_recovery.heading_weight' if trial == 'b' else None)
        (folder/'model_report.json').write_text(json.dumps(report, indent=2)+'\n')
        smoke(cfg, steps=1000)
        validate_artifacts(cfg)
        _, metadata = load_standing_rehearsal(config['training']['standing_rehearsal'], config, xml)
        configurations.append(config)
        print(json.dumps(dict(trial=trial, prepared=True, rehearsal=metadata)), flush=True)
    a, b = copy.deepcopy(configurations)
    a['experiment'].pop('output_directory'); b['experiment'].pop('output_directory')
    a['training']['standing_rehearsal'].pop('dataset'); b['training']['standing_rehearsal'].pop('dataset')
    b['balance_recovery']['heading_weight'] = a['balance_recovery']['heading_weight']
    assert a == b, 'Trials must differ only in heading reward and data/output location'
    print('ROUND16_PAIRED_MODEL_PARENT_REHEARSAL_VALIDATED', flush=True)


if __name__ == '__main__':
    main()
