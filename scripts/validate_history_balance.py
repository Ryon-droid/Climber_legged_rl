"""Check real CAD migration, a short PPO update and portable checkpoint loading."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT / 'outputs/mujoco_runs/'
        'cad_getup_stand_round22_guarded_balance_20261031_183529/best/best_model.zip')
    parser.add_argument('--steps', type=int, default=4096)
    args = parser.parse_args()
    if not 256 <= args.steps <= 8192:
        parser.error('--steps must be in [256,8192]')
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv
    from climber_legged_rl.mujoco_env import make_mujoco_env
    from climber_legged_rl.mujoco_model import load_config
    from climber_legged_rl.mujoco_training import load_warm_start, validate_policy_transfer
    torch.set_num_threads(1)
    config_path = ROOT / 'configs/robot/cad_balance_history.yaml'
    config = load_config(config_path)
    folder = ROOT / config['experiment']['output_directory']
    model_path = folder / 'robot_experiment.xml'
    validate_policy_transfer(args.parent, model_path, config, warm_start=True)
    output = folder / 'validation'
    output.mkdir(exist_ok=True)
    for name in ('robot_experiment.xml', 'getup_poses.json'):
        shutil.copyfile(folder / name, output / name)
    shutil.copyfile(config_path, output / 'config.yaml')
    vector = DummyVecEnv([lambda: make_mujoco_env(model_path, config_path) for _ in range(2)])
    settings = dict(config['training'], n_steps=128, batch_size=256)
    policy, _, _ = load_warm_start(args.parent, vector, settings, 'cpu', 130)
    parent = PPO.load(args.parent, device='cpu')
    parent_folder = args.parent.parent.parent if args.parent.parent.name in ('best', 'checkpoints') else args.parent.parent
    old_env = make_mujoco_env(parent_folder / 'robot_experiment.xml', parent_folder / 'config.yaml')
    new_env = make_mujoco_env(model_path, config_path)
    comparisons = []
    try:
        for mode, seed in [('folded', 112000), ('standing', 112000), ('standing', 99905)]:
            old_env.push.enabled = new_env.push.enabled = False
            old_obs, _ = old_env.reset(seed=seed, options={'reset_mode': mode})
            new_obs, _ = new_env.reset(seed=seed, options={'reset_mode': mode})
            error = position_error = 0.
            while True:
                a, _ = parent.predict(old_obs, deterministic=True)
                b, _ = policy.predict(new_obs, deterministic=True)
                error = max(error, float(np.max(np.abs(a - b))))
                old_obs, _, old_done, old_truncated, old_info = old_env.step(a)
                new_obs, _, done, truncated, info = new_env.step(b)
                position_error = max(position_error, float(np.max(np.abs(old_env.data.qpos - new_env.data.qpos))))
                if done or truncated or old_done or old_truncated:
                    if (old_done, old_truncated) != (done, truncated):
                        raise AssertionError(f'Migrated rollout termination differs: {mode=} {seed=} '
                            f'{error=} {position_error=} old={old_info.get("episode_metrics")} '
                            f'new={info.get("episode_metrics")}')
                    break
            comparisons.append(dict(mode=mode, seed=seed, max_action_error=error,
                max_qpos_error=position_error, seconds=info['episode_metrics']['seconds'],
                parent_success=old_info['combined_success'], migrated_success=info['combined_success']))
        if any(c['max_action_error'] > 1e-5 or c['max_qpos_error'] > 1e-5 for c in comparisons):
            raise AssertionError('Initial history migration changed the physical rollout')
        actor = policy.policy.mlp_extractor.policy_net
        frozen = {k: v.detach().clone() for k, v in actor.state_dict().items() if k.startswith('rising_')}
        standing = {k: v.detach().clone() for k, v in actor.state_dict().items() if k.startswith('standing_')}
        policy.set_logger(configure(str(output), ['csv']))
        policy.learn(total_timesteps=args.steps)
        frozen_equal = all(torch.equal(value, actor.state_dict()[key]) for key, value in frozen.items())
        standing_changed = any(not torch.equal(value, actor.state_dict()[key]) for key, value in standing.items())
        if not frozen_equal or not standing_changed:
            raise AssertionError('Frozen rising or trainable standing contract failed')
        policy.save(output / 'validation_policy.zip')
        loaded = PPO.load(output / 'validation_policy.zip', device='cpu')
        obs, _ = new_env.reset(seed=112001, options={'reset_mode': 'standing'})
        np.testing.assert_array_equal(policy.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
        summary = dict(simulation_only=True, formal_training=False, timesteps=policy.num_timesteps,
            observation_shape=list(new_env.observation_space.shape), comparisons=comparisons,
            frozen_rising_parameters_equal=frozen_equal, standing_parameters_changed=standing_changed,
            ordinary_ppo_reload=True, model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            parent_checkpoint=str(args.parent), note='Integration validation; no claim of improved balance success rate')
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        old_env.close()
        new_env.close()
        vector.close()


if __name__ == '__main__':
    main()
