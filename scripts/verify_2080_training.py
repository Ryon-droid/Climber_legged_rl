"""Verify a completed real CAD PPO update and checkpoint portability."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_training import validate_policy_transfer
from climber_legged_rl.velocity_estimation import VelocityEstimationPPO


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--expected-steps', type=int, required=True)
    parser.add_argument('--elapsed-seconds', type=float)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/robot/cad_balance_estimator_round14.yaml')
    parser.add_argument('--parent', type=Path, default=ROOT/'outputs/mujoco_runs/cad_balance_estimator_round13_20261101_221330/best/best_model.zip')
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = args.run_dir.resolve()
    cfg = args.config.resolve()
    config = load_config(cfg)
    active_xml = ROOT / config['experiment']['output_directory'] / 'robot_experiment.xml'
    validate_artifacts(cfg)
    status = json.loads((folder / 'status.json').read_text())
    metadata = json.loads((folder / 'metadata.json').read_text())
    assert status['state'] == 'completed', status
    assert status['timesteps'] >= args.expected_steps
    assert metadata['num_envs'] >= 1
    parent = args.parent.resolve()
    checkpoint = folder / 'final_policy.zip'
    validate_policy_transfer(checkpoint, active_xml, config)
    previous = PPO.load(parent, device='cpu')
    ordinary = PPO.load(checkpoint, device='cpu')
    restored = VelocityEstimationPPO.load(checkpoint, device='cpu')
    before = previous.policy.mlp_extractor.policy_net.state_dict()
    after = restored.policy.mlp_extractor.policy_net.state_dict()
    frozen = {}
    for prefix in ('rising_', 'velocity_estimator.'):
        keys = [k for k in before if k.startswith(prefix)]
        assert keys
        frozen[prefix] = all(torch.equal(before[k], after[k]) for k in keys)
    standing_keys = [k for k in before if k.startswith('standing_')]
    updated = [k for k in standing_keys if not torch.equal(before[k], after[k])]
    assert updated and all(frozen.values())
    assert all(torch.isfinite(v).all() for v in restored.policy.state_dict().values())
    assert not restored.estimator_trainable
    assert restored.separate_actor_critic_gradient_clipping
    rehearsal = restored.configure_standing_rehearsal(config['training']['standing_rehearsal'],
                                                     config, active_xml, 143, restore_optimizer=True)
    assert restored._standing_rehearsal.optimizer.state
    env = make_mujoco_env(active_xml, cfg)
    reward_steps = 0
    max_difference = 0.0
    try:
        obs, _ = env.reset(seed=263200, options={'reset_mode': 'standing'})
        for i in range(200):
            a = ordinary.predict(obs, deterministic=True)[0]
            b = restored.predict(obs, deterministic=True)[0]
            np.testing.assert_array_equal(a, b)
            max_difference = max(max_difference, float(np.max(np.abs(a-b))))
            obs, reward, ended, truncated, info = env.step(a)
            assert np.isfinite(obs).all() and np.isfinite(reward)
            reward_steps += 1
            if ended or truncated:
                obs, _ = env.reset(seed=263201+i, options={'reset_mode': 'standing'})
    finally:
        env.close()
    with (folder / 'progress.csv').open() as stream:
        progress = list(csv.DictReader(stream))
    fps = [float(row['time/fps']) for row in progress if row.get('time/fps')]
    groups = json.loads((folder / 'evaluation_by_reset.json').read_text())
    evaluation = {}
    for name, group in groups.items():
        evaluation[name] = dict(episodes=len(group['episodes']),
                                successes=sum(bool(e['combined_success']) for e in group['episodes']),
                                mean_episode_seconds=group['mean_episode_seconds'])
    report = dict(training_verified=True, simulation_only=True, hostname=platform.node(),
                  physics_backend=metadata['backend'], policy_device=metadata['device'],
                  python=platform.python_version(), torch=torch.__version__, mujoco=mujoco.__version__,
                  timesteps=status['timesteps'], num_envs=metadata['num_envs'],
                  training_and_evaluation_seconds=args.elapsed_seconds,
                  latest_reported_training_fps=fps[-1] if fps else None,
                  max_reported_training_fps=max(fps) if fps else None,
                  standing_tensor_count_changed=len(updated), frozen_weights_unchanged=frozen,
                  ordinary_and_custom_reload_identical=True, reload_max_action_difference=max_difference,
                  finite_reload_rollout_steps=reward_steps, rehearsal_optimizer_restored=True,
                  rehearsal=rehearsal, evaluation=evaluation,
                  checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                  config_sha256=sha(cfg), model_sha256=sha(active_xml),
                  source_snapshot_manifest=str(ROOT/'outputs/relocation_2080_20261104/manifest.json'),
                  limitation='Bounded simulation continuation, not converged training or hardware validation.')
    try:
        report['gpu'] = subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        report['gpu'] = None
    destination = folder / 'device_verification.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print('VERIFIED_REPORT=' + str(destination), flush=True)


if __name__ == '__main__':
    main()
