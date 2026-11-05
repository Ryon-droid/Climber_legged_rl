"""Collect teacher episodes, fit a sensor-only velocity estimator and save a warm start.

Validation splits whole episodes, never neighboring frames of the same trajectory.
Control weights remain unchanged during this supervised preparation.
"""
import argparse
from copy import deepcopy
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

PARENT = ROOT / 'outputs/mujoco_runs/cad_getup_stand_round22_guarded_balance_20261031_183529/best/best_model.zip'


def prepare(config_path, parent=PARENT, episodes=16, epochs=80, output=None, dataset=None):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from climber_legged_rl.cad_dynamics import build_dynamics
    from climber_legged_rl.getup import prepare_pose_bank
    from climber_legged_rl.mujoco_env import make_mujoco_env
    from climber_legged_rl.mujoco_model import load_config
    from climber_legged_rl.mujoco_training import load_warm_start, validate_policy_transfer
    from climber_legged_rl.velocity_estimation import estimator_update
    from smoke_mujoco import run as smoke
    config_path, parent = Path(config_path).resolve(), Path(parent).resolve()
    config = load_config(config_path)
    if not config['training'].get('velocity_estimator', {}).get('enabled'):
        raise ValueError('Choose a velocity_estimator enabled configuration')
    if episodes < 8 or epochs < 1:
        raise ValueError('Require at least eight complete episodes and positive epochs')
    torch.set_num_threads(1)
    build_dynamics(config_path)
    prepare_pose_bank(config_path)
    smoke(config_path, steps=1000)
    folder = ROOT / config['experiment']['output_directory']
    model_path = folder / 'robot_experiment.xml'
    validate_policy_transfer(parent, model_path, config, warm_start=True)
    output = Path(output) if output else folder / 'pretrained'
    if (output / 'pretrained_policy.zip').exists():
        raise FileExistsError(f'Pretrained checkpoint already exists: {output}; choose another --output')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('robot_experiment.xml', 'getup_poses.json'):
        shutil.copyfile(folder / name, output / name)
    shutil.copyfile(config_path, output / 'config.yaml')
    teacher = PPO.load(parent, device='cpu')
    vector = DummyVecEnv([lambda: make_mujoco_env(model_path, config_path)])
    student, _, _ = load_warm_start(parent, vector, config['training'], 'cpu', 130)
    student.configure_estimator(config['training']['velocity_estimator'])
    env = make_mujoco_env(model_path, config_path)
    observations, episode_ids, records = [], [], []
    frozen = {k: v.clone() for k, v in student.policy.state_dict().items() if 'velocity_estimator' not in k}
    try:
        if dataset is None:
            for episode in range(episodes):
                mode = 'folded' if episode % 2 == 0 else 'standing'
                seed = 210000 + episode
                obs, _ = env.reset(seed=seed, options={'reset_mode': mode})
                count = 0
                while True:
                    observations.append(obs.copy())
                    episode_ids.append(episode)
                    # Underlying history retains the teacher's original observations;
                    # only the outer student wrapper masks privileged velocity.
                    teacher_history = np.concatenate(env.env._history)
                    dimensions = teacher.observation_space.shape[0]
                    if dimensions == 36:
                        teacher_obs = teacher_history[-36:]
                    elif dimensions == len(teacher_history):
                        teacher_obs = teacher_history
                    else:
                        raise ValueError('Teacher must use 36 inputs or matching original history')
                    action, _ = teacher.predict(teacher_obs, deterministic=True)
                    obs, _, done, truncated, info = env.step(action)
                    count += 1
                    if done or truncated:
                        break
                records.append(dict(episode=episode, seed=seed, mode=mode, frames=count,
                    reason=info['episode_metrics']['reason'], success=info['combined_success']))
                print(f'VELOCITY_TEACHER episode={episode+1}/{episodes} mode={mode} frames={count}', flush=True)
            observations = np.array(observations, np.float32)
            episode_ids = np.array(episode_ids)
        else:
            dataset = Path(dataset)
            provenance = json.loads((dataset.parent/'pretrain_report.json').read_text())
            if (provenance['teacher_sha256'] != hashlib.sha256(parent.read_bytes()).hexdigest()
                    or provenance['model_sha256'] != hashlib.sha256(model_path.read_bytes()).hexdigest()
                    or provenance['dataset_sha256'] != hashlib.sha256(dataset.read_bytes()).hexdigest()):
                raise ValueError('Cached dataset provenance does not match teacher/model')
            with np.load(dataset, allow_pickle=False) as data:
                observations = data['observations'].copy()
                episode_ids = data['episode_ids'].copy()
            if (observations.ndim != 2 or observations.shape[1] != 183
                    or len(episode_ids) != len(observations) or not np.isfinite(observations).all()):
                raise ValueError('Invalid cached velocity dataset')
            records = provenance['episodes']
        # Every fourth pair of prone/standing episodes is held out together.
        heldout = episode_ids % 8 >= 6
        train_obs, validation_obs = observations[~heldout], observations[heldout]
        np.savez_compressed(output / 'teacher_velocity.npz', observations=observations,
            episode_ids=episode_ids, heldout=heldout)
        actor = student.policy.mlp_extractor.policy_net
        def metrics(data):
            with torch.no_grad():
                predictions = actor.estimate_velocity(torch.from_numpy(data)).numpy()
            error = predictions - data[:, -3:]
            return dict(rmse_xyz_m_s=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                rmse_m_s=float(np.sqrt(np.mean(error**2))))
        initial = metrics(validation_obs)
        zero_rmse = float(np.sqrt(np.mean(validation_obs[:, -3:]**2)))
        best_error = initial['rmse_m_s']
        best_weights = deepcopy(actor.velocity_estimator.state_dict())
        best_optimizer = deepcopy(student.policy.estimator_optimizer.state_dict())
        best_epoch = 0
        for epoch in range(epochs):
            estimator_update(student.policy, train_obs, epochs=1, batch_size=512)
            result = metrics(validation_obs)
            if result['rmse_m_s'] < best_error:
                best_error, best_epoch = result['rmse_m_s'], epoch + 1
                best_weights = deepcopy(actor.velocity_estimator.state_dict())
                best_optimizer = deepcopy(student.policy.estimator_optimizer.state_dict())
        actor.velocity_estimator.load_state_dict(best_weights)
        student.policy.estimator_optimizer.load_state_dict(best_optimizer)
        final = metrics(validation_obs)
        for key, value in frozen.items():
            if not torch.equal(value, student.policy.state_dict()[key]):
                raise AssertionError('Supervised training changed a controller/critic parameter: ' + key)
        student.num_timesteps = 0
        student.save(output / 'pretrained_policy.zip')
        report = dict(simulation_only=True, formal_training=False,
            teacher_checkpoint=str(parent), teacher_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
            model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(), episodes=records,
            dataset_sha256=hashlib.sha256((output/'teacher_velocity.npz').read_bytes()).hexdigest(),
            training_frames=len(train_obs), heldout_frames=len(validation_obs),
            split='whole episodes: episode_id % 8 >= 6 held out', epochs=epochs, best_epoch=best_epoch,
            initial_heldout=initial, heldout=final, zero_velocity_baseline_rmse_m_s=zero_rmse,
            beats_zero_baseline=final['rmse_m_s'] < zero_rmse,
            control_parameters_unchanged=True, training_input_dimensions=observations.shape[1],
            runtime_actor_input_dimensions=actor.actor_dimensions,
            note='Estimator accuracy is measured on teacher episodes; closed-loop success must be evaluated separately')
        (output/'pretrain_report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k:v for k,v in report.items() if k != 'episodes'}, indent=2), flush=True)
        return output / 'pretrained_policy.zip'
    finally:
        env.close()
        vector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/robot/cad_balance_estimator.yaml')
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dataset', type=Path, help='Reuse a provenance-checked teacher_velocity.npz')
    args = parser.parse_args()
    prepare(args.config, args.parent, args.episodes, args.epochs, args.output, args.dataset)


if __name__ == '__main__':
    main()
