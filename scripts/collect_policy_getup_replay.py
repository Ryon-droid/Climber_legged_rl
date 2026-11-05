"""Collect successful pure-policy rollouts for training-only actor rehearsal."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import numpy as np
import torch
from stable_baselines3 import PPO
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import ROOT, load_config
from climber_legged_rl.mujoco_training import validate_policy_transfer


def collect(config_path, checkpoint, output, standing_episodes=20, seed=99800, push_episode_probability=None):
    config = load_config(config_path)
    model_path = ROOT / config['experiment']['output_directory'] / 'robot_experiment.xml'
    validate_artifacts(config_path)
    validate_policy_transfer(checkpoint, model_path, config, warm_start=False)
    if output.exists():
        raise ValueError('Replay output already exists; use a new directory')
    if not config['getup'].get('policy_control', {}).get('enabled'):
        raise ValueError('Replay source must use pure policy control')
    policy = PPO.load(checkpoint, device='cpu')
    env = make_mujoco_env(model_path, config_path)
    if push_episode_probability is not None:
        if not env.push.enabled or not 0 <= push_episode_probability <= 1:
            env.close()
            raise ValueError('Push override requires enabled pushes and probability in [0,1]')
        env.push.config = dict(env.push.config,episode_probability=float(push_episode_probability))
    observations, actions, weights, outcomes = [], [], [], []
    poses = env.pose_bank
    specs = [('folded', p, seed) for p in poses]
    specs += [('standing', None, seed+i) for i in range(standing_episodes)]
    try:
        for mode, pose, episode_seed in specs:
            env.pose_bank = [pose] if pose is not None else poses
            obs, _ = env.reset(seed=episode_seed, options={'reset_mode': mode})
            episode = []
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                weight = 12. if mode == 'folded' and env.steps*env.dt < 1.6 else 4. if mode == 'folded' else 1.
                episode.append((obs.copy(), np.clip(action, -1., 1.), weight))
                obs, _, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            success = bool(info['combined_success'])
            outcome = dict(mode=mode, seed=episode_seed, success=success,
                rearward_offset_m=None if pose is None else pose['rearward_offset_m'],
                getup_seconds=info['getup_seconds'], reason=info['termination_reason'] or 'timeout',
                final_displacement_m=info['planar_displacement_m'], samples=len(episode))
            outcomes.append(outcome)
            print(json.dumps(outcome), flush=True)
            if mode == 'folded' and not success:
                raise RuntimeError('Source policy failed a prone rollout; no replay dataset saved')
            if success:
                for observation, action, weight in episode:
                    observations.append(observation); actions.append(action); weights.append(weight)
    finally:
        env.close()
    if not any(r['mode'] == 'standing' and r['success'] for r in outcomes):
        raise RuntimeError('No successful standing trajectories collected')
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / 'demonstrations.npz'
    np.savez_compressed(dataset, observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32), weights=np.asarray(weights, dtype=np.float64))
    (output / 'config.yaml').write_bytes(Path(config_path).read_bytes())
    report = dict(simulation_only=True, source_kind='successful_pure_policy_rollouts',
        checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        config=str(config_path), config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(), expert_outcomes=outcomes,
        stages=[dict(samples=len(observations), source='successful policy episodes only')],
        runtime_reference_assistance=False, seed=seed, push_episode_probability_override=push_episode_probability)
    (output / 'imitation_report.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--standing-episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=99800)
    parser.add_argument('--push-episode-probability',type=float,help='Explicitly override push sampling during collection')
    args = parser.parse_args()
    if args.standing_episodes < 1:
        parser.error('--standing-episodes must be positive')
    torch.set_num_threads(1)
    print(json.dumps(collect(args.config, args.checkpoint, args.output, args.standing_episodes, args.seed,args.push_episode_probability), indent=2))
