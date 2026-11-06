"""Prepare disjoint train/selection/test prone banks with identical measured physics."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'scripts')]
import yaml
from climber_legged_rl.cad_dynamics import input_hashes, validate_artifacts
from climber_legged_rl.getup import prepare_pose_bank
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_training import validate_policy_transfer
from smoke_mujoco import run as smoke

PARENT = ROOT/'outputs/mujoco_runs/spring208_rising_distill_20261106_031028/final_policy.zip'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = load_config(ROOT/'configs/robot/cad_balance_spring208.yaml')
    source = ROOT/original['experiment']['output_directory']
    nominal = original['getup']['rearward_offsets_m'][0]
    # Each group contains a fresh bounded solve, not perturbed active joints with
    # unsolved passive joints. Selection and test offsets never enter PPO resets.
    offsets = {'train': [nominal-.006, nominal-.003, nominal],
               'test': [nominal-.0075, nominal-.0045, nominal-.0015],
               'nominal': [nominal]}
    for split, poses in offsets.items():
        config = copy.deepcopy(original)
        config['experiment']['output_directory'] = f'outputs/cad_spring208_phase1_{split}'
        config['getup']['rearward_offsets_m'] = poses
        if split != 'nominal':
            config['getup']['folded_reset_noise'] = dict(enabled=True,
                roll_half_width_rad=.004363323129985824,
                pitch_half_width_rad=.004363323129985824,
                base_speed_half_width_m_s=.002, base_rate_half_width_rad_s=.01)
        config['getup']['stand_probability'] = .10
        config['push']['episode_probability'] = 0.
        train = config['training']
        train.update(total_timesteps=40000, num_envs=4, seed=208341,
            n_steps=512, batch_size=256, n_epochs=2, learning_rate=3e-6,
            clip_range=.02, target_kl=.003, vf_coef=.001,
            freeze_rising_actor=False, freeze_standing_actor=True, freeze_action_std=True,
            action_std=[.015]*4+[.15]*2, evaluation_steps=20000, evaluation_workers=4,
            evaluation_standing_episodes=8, evaluation_push_episodes=0,
            evaluate_initial_policy=True, evaluation_all_poses=True,
            evaluation_hard_standing_seeds=[], evaluation_hard_push_seeds=[])
        train['hard_standing_resets']['enabled'] = False
        train['hard_push_resets']['enabled'] = False
        # No mechanics migration is needed: this parent already uses 208 mm.
        train.pop('allow_spring_limit_warm_start', None)
        train.pop('spring_limit_migration', None)
        cfg = ROOT/f'configs/robot/cad_spring208_phase1_{split}.yaml'
        cfg.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        folder = ROOT/config['experiment']['output_directory']
        folder.mkdir(exist_ok=True)
        xml = folder/'robot_experiment.xml'
        if xml.exists() and sha(xml) != sha(source/'robot_experiment.xml'):
            raise RuntimeError('Existing output has unexpected physical XML')
        xml.write_bytes((source/'robot_experiment.xml').read_bytes())
        report = json.loads((source/'model_report.json').read_text())
        report.update(generated_model=str(xml), generated_sha256=sha(xml),
            config_sha256=sha(cfg), input_sha256=input_hashes(cfg))
        report['phase1_provenance'] = dict(parent_checkpoint=str(PARENT),
            parent_checkpoint_sha256=sha(PARENT), physical_model_identical=True,
            split=split, training_offsets_m=offsets['train'], heldout_offsets_m=offsets['test'])
        (folder/'model_report.json').write_text(json.dumps(report, indent=2)+'\n')
        bank = prepare_pose_bank(cfg)
        smoke(cfg, steps=1000)
        validate_artifacts(cfg)
        validate_policy_transfer(PARENT, xml, config, warm_start=True)
        print(json.dumps(dict(split=split, config=str(cfg), prepared=True,
            poses=[dict(offset=p['rearward_offset_m'], spring=p['spring_eye_distances_m'],
                closure=p['closure_error_m']) for p in bank['poses']])), flush=True)
    assert not set(offsets['train']) & set(offsets['test'])
    print('SPRING208_PHASE1_READY', flush=True)


if __name__ == '__main__':
    main()
