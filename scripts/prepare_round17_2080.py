"""Prepare round-17 paired rehearsal experiments with immutable CAD physics."""
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
import numpy as np
from climber_legged_rl.cad_dynamics import validate_artifacts, input_hashes
from climber_legged_rl.getup import prepare_pose_bank
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_training import validate_policy_transfer
from climber_legged_rl.standing_rehearsal import load_standing_rehearsal
from smoke_mujoco import run as smoke
from reweight_standing_rehearsal import reweighted_arrays


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original_cfg = ROOT/'configs/robot/cad_balance_estimator_round16a.yaml'
    original = load_config(original_cfg)
    original_report, _ = validate_artifacts(original_cfg)
    source_folder = ROOT/original['experiment']['output_directory']
    source_data = ROOT/original['training']['standing_rehearsal']['dataset']
    parent = ROOT/'outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip'
    diagnostics = json.loads((ROOT/'outputs/round17_preflight_20261105/teacher_comparison.json').read_text())
    assert all(diagnostics[f'teacher_standing_{seed}']['episode']['getup_metrics']['combined_success']
               for seed in (261409, 261106, 261704)), 'Recovery teacher must reproduce before training'
    configs = []
    for trial in ('a', 'b'):
        cfg = ROOT/f'configs/robot/cad_balance_estimator_round17{trial}.yaml'
        config = load_config(cfg)
        for key in original.keys() | config.keys():
            if key not in ('training', 'experiment'):
                assert original.get(key) == config.get(key), key
        experiment = dict(config['experiment'])
        experiment['output_directory'] = original['experiment']['output_directory']
        assert experiment == original['experiment']
        old_training, new_training = copy.deepcopy(original['training']), copy.deepcopy(config['training'])
        old_training['seed'] = new_training['seed'] = 146
        for values in (old_training, new_training):
            values['standing_rehearsal'].pop('dataset')
            values['standing_rehearsal'].pop('dataset_sha256')
        assert old_training == new_training
        folder = ROOT/config['experiment']['output_directory']
        folder.mkdir(exist_ok=True)
        xml = folder/'robot_experiment.xml'
        if xml.exists():
            assert sha(xml) == sha(source_folder/'robot_experiment.xml')
        else:
            xml.write_bytes((source_folder/'robot_experiment.xml').read_bytes())
        report = copy.deepcopy(original_report)
        report.update(generated_model=str(xml), config_sha256=sha(cfg), input_sha256=input_hashes(cfg))
        report['round17_provenance'] = dict(source_config=str(original_cfg), source_model_sha256=sha(xml),
            physical_model_identical=True, reward_and_control_identical=True, trial=trial,
            parent_checkpoint=str(parent), parent_sha256=sha(parent),
            recovery_rehearsal_mass=.1 if trial == 'a' else .3)
        (folder/'model_report.json').write_text(json.dumps(report, indent=2)+'\n')
        prepare_pose_bank(cfg)
        smoke(cfg, steps=1000)
        validate_artifacts(cfg)
        validate_policy_transfer(parent, xml, config, warm_start=True)
        arrays, metadata = load_standing_rehearsal(config['training']['standing_rehearsal'], config, xml)
        if trial == 'b':
            source_report = json.loads(source_data.with_suffix('.json').read_text())
            with np.load(source_data, allow_pickle=False) as source_arrays:
                expected, _ = reweighted_arrays(source_arrays, source_report, .3)
            np.testing.assert_array_equal(arrays[0], expected['observations'])
            np.testing.assert_array_equal(arrays[1], expected['actions'])
            np.testing.assert_allclose(arrays[2], expected['weights']/expected['weights'].sum(), rtol=1e-12)
        configs.append(config)
        print(json.dumps(dict(trial=trial, prepared=True, rehearsal=metadata)), flush=True)
    a, b = copy.deepcopy(configs)
    for values in (a, b):
        values['experiment'].pop('output_directory')
        values['training']['standing_rehearsal'].pop('dataset')
        values['training']['standing_rehearsal'].pop('dataset_sha256')
    assert a == b, 'Paired experiments differ beyond rehearsal weights'
    print('ROUND17_PAIRED_MODEL_PARENT_REHEARSAL_VALIDATED', flush=True)


if __name__ == '__main__':
    main()
