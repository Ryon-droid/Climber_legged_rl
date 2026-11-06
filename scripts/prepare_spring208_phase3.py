"""Broaden valid prone training poses and select with repeated reset seeds."""
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'scripts')]
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from climber_legged_rl.cad_dynamics import input_hashes, validate_artifacts
from climber_legged_rl.getup import prepare_pose_bank
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_training import load_warm_start, validate_policy_transfer
from smoke_mujoco import run as smoke

PARENT = ROOT/'outputs/mujoco_runs/spring208_phase1_20261106_1303/best/best_model.zip'
PARENT_SHA = '28a44dcfc15928016ec809137bade5b127a31a5f2cc880ee96ca590b5bf4c454'
STD = [.0015]*4+[.015]*2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_case(job):
    scale, offset, seed = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    cfg = ROOT/'configs/robot/cad_spring208_phase3_train.yaml'
    config = load_config(cfg)
    env = make_mujoco_env(ROOT/config['experiment']['output_directory']/'robot_experiment.xml', cfg)
    try:
        env.pose_bank = [p for p in env.pose_bank if abs(p['rearward_offset_m']-offset) < 1e-10]
        if not env.pose_bank:
            raise RuntimeError('Missing calibration pose')
        model = PPO.load(PARENT, device='cpu')
        with torch.no_grad():
            model.policy.log_std.copy_(torch.log(torch.tensor(STD)*scale))
        obs, _ = env.reset(seed=seed, options={'reset_mode': 'folded'})
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                return dict(scale=scale, offset_m=offset, seed=seed,
                    success=bool(info['combined_success']), acquired=bool(info['getup_acquired']),
                    reason=info['termination_reason'] or 'timeout')
    finally:
        env.close()


def main():
    torch.set_num_threads(1)
    if sha(PARENT) != PARENT_SHA:
        raise RuntimeError('Unexpected phase-1 checkpoint')
    previous = load_config(ROOT/'configs/robot/cad_spring208_phase2_train.yaml')
    nominal = max(previous['getup']['rearward_offsets_m'])
    offsets = dict(train=[nominal-.0012*i for i in reversed(range(7))],
        test=[nominal-.0069, nominal-.0041, nominal-.0017], nominal=[nominal])
    old_test = load_config(ROOT/'configs/robot/cad_spring208_phase1_test.yaml')['getup']['rearward_offsets_m']
    old_test += load_config(ROOT/'configs/robot/cad_spring208_phase2_test.yaml')['getup']['rearward_offsets_m']
    if set(offsets['test']) & set(offsets['train']+old_test):
        raise RuntimeError('Fresh assessment geometry overlaps earlier sets')
    source = ROOT/previous['experiment']['output_directory']
    for split, poses in offsets.items():
        config = copy.deepcopy(previous)
        config['experiment']['output_directory'] = f'outputs/cad_spring208_phase3_{split}'
        config['getup']['rearward_offsets_m'] = poses
        if split == 'nominal':
            config['getup'].pop('folded_reset_noise', None)
        config['training'].update(seed=208343, total_timesteps=200000, action_std=STD,
            learning_rate=5e-7, evaluation_steps=40000, checkpoint_steps=40000,
            evaluation_pose_episodes=8, evaluation_seed=431000, evaluation_rank_worst_pose=True)
        cfg = ROOT/f'configs/robot/cad_spring208_phase3_{split}.yaml'
        if cfg.exists():
            raise RuntimeError('Refusing to overwrite an existing phase-3 config')
        cfg.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        folder = ROOT/config['experiment']['output_directory']
        folder.mkdir(exist_ok=False)
        xml = folder/'robot_experiment.xml'
        xml.write_bytes((source/'robot_experiment.xml').read_bytes())
        report = json.loads((source/'model_report.json').read_text())
        report.pop('phase1_provenance', None)
        report.pop('phase2_provenance', None)
        report.update(generated_model=str(xml), generated_sha256=sha(xml),
            config_sha256=sha(cfg), input_sha256=input_hashes(cfg),
            phase3_provenance=dict(parent_checkpoint=str(PARENT), parent_sha256=sha(PARENT),
                physical_model_identical=True, split=split, training_offsets_m=offsets['train'],
                fresh_assessment_offsets_m=offsets['test'], previous_test_used_for_diagnosis_only=True))
        (folder/'model_report.json').write_text(json.dumps(report, indent=2)+'\n')
        prepare_pose_bank(cfg)
        smoke(cfg, steps=1000)
        validate_artifacts(cfg)
        validate_policy_transfer(PARENT, xml, config, warm_start=True)
        print(json.dumps(dict(prepared=split, physical_xml_sha256=sha(xml))), flush=True)
    cfg = ROOT/'configs/robot/cad_spring208_phase3_train.yaml'
    config = load_config(cfg)
    xml = ROOT/config['experiment']['output_directory']/'robot_experiment.xml'
    vector = DummyVecEnv([lambda: make_mujoco_env(xml, cfg)])
    try:
        model, _, _ = load_warm_start(PARENT, vector, config['training'], 'cpu', 208343)
        model.configure_estimator(config['training']['velocity_estimator'])
        actor = model.policy.mlp_extractor.policy_net
        for module in (actor.standing_net, actor.standing_head, actor.velocity_estimator):
            if any(p.requires_grad for p in module.parameters()):
                raise RuntimeError('Protected module is trainable')
        if model.policy.log_std.requires_grad or not all(p.requires_grad for p in actor.rising_net.parameters()):
            raise RuntimeError('Invalid exploration/rising freeze configuration')
    finally:
        vector.close()
    jobs = [(scale, offset, seed) for scale in (10., 1.)
            for offset in offsets['train'] for seed in range(438000, 438002)]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        rows = list(pool.map(calibration_case, jobs))
    summary = {str(scale):dict(count=sum(r['scale']==scale for r in rows),
        success=sum(r['success'] for r in rows if r['scale']==scale),
        acquired=sum(r['acquired'] for r in rows if r['scale']==scale)) for scale in (10., 1.)}
    report = dict(parent_sha256=sha(PARENT), summary=summary, episodes=rows,
        note='Exploratory small training-set calibration, excluded from final assessment seeds.')
    (ROOT/'outputs/cad_spring208_phase3_train/exploration_calibration.json').write_text(json.dumps(report, indent=2)+'\n')
    print('EXPLORATION_CALIBRATION '+json.dumps(summary), flush=True)
    if summary['1.0']['success'] < summary['10.0']['success'] or summary['1.0']['success'] == 0:
        raise RuntimeError('Lower exploration noise did not produce a usable calibration signal')
    print('SPRING208_PHASE3_READY', flush=True)


if __name__ == '__main__':
    main()
