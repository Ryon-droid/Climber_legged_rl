"""Evaluate estimator closed-loop behavior and run a bounded PPO/checkpoint smoke test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/robot/cad_balance_estimator.yaml')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/cad_balance_estimator/pretrained/pretrained_policy.zip')
    parser.add_argument('--steps', type=int, default=1024)
    args = parser.parse_args()
    if not 256 <= args.steps <= 8192:
        parser.error('Require 256..8192 integration-validation steps')
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv
    from climber_legged_rl.cad_dynamics import build_dynamics
    from climber_legged_rl.getup import prepare_pose_bank
    from climber_legged_rl.mujoco_env import make_mujoco_env
    from climber_legged_rl.mujoco_model import load_config
    from climber_legged_rl.mujoco_training import load_warm_start, validate_policy_transfer, evaluate
    from climber_legged_rl.velocity_estimation import VelocityEstimationPPO
    from smoke_mujoco import run as smoke
    torch.set_num_threads(1)
    cfg = load_config(args.config)
    folder = ROOT/cfg['experiment']['output_directory']
    xml = folder/'robot_experiment.xml'
    build_dynamics(args.config)
    prepare_pose_bank(args.config)
    smoke(args.config, steps=1000)
    validate_policy_transfer(args.checkpoint, xml, cfg, warm_start=True)
    output = folder/'validation'
    output.mkdir(parents=True, exist_ok=True)
    for name in ('robot_experiment.xml', 'getup_poses.json'):
        shutil.copyfile(folder/name, output/name)
    shutil.copyfile(args.config, output/'config.yaml')
    before = {mode: evaluate(args.checkpoint, args.config, xml, episodes=2, seed=112000,
        reset_mode=mode, push_episode_probability=0.) for mode in ('folded', 'standing')}
    gpu = cfg['experiment'].get('physics_backend') == 'mujoco_warp'
    factories = [lambda: make_mujoco_env(xml, args.config)] * 2
    if gpu:
        from climber_legged_rl.warp_backend import WarpVecEnv
        vector = WarpVecEnv(factories, cfg.get('gpu_physics', {}))
    else:
        vector = DummyVecEnv(factories)
    try:
        settings = dict(cfg['training'], n_steps=128, batch_size=256)
        learner, _, _ = load_warm_start(args.checkpoint, vector, settings, 'cpu', 130)
        learner.configure_estimator(settings['velocity_estimator'])
        rehearsal = learner.configure_standing_rehearsal(settings.get('standing_rehearsal', {}),
                                                        cfg, xml, 130)
        actor = learner.policy.mlp_extractor.policy_net
        frozen = {k: v.clone() for k, v in actor.state_dict().items() if k.startswith('rising_')}
        standing = {k: v.clone() for k, v in actor.state_dict().items() if k.startswith('standing_')}
        estimator = {k: v.clone() for k, v in actor.velocity_estimator.state_dict().items()}
        learner.set_logger(configure(str(output), ['csv']))
        vector.set_options([{'reset_mode': 'folded'}, {'reset_mode': 'standing'}])
        learner.learn(args.steps)
        frozen_unchanged = all(torch.equal(v, actor.state_dict()[k]) for k, v in frozen.items())
        if learner.policy.freeze_rising_actor:
            assert frozen_unchanged
        else:
            assert not frozen_unchanged
        assert any(not torch.equal(v, actor.state_dict()[k]) for k, v in standing.items())
        estimator_changed = any(not torch.equal(v, actor.velocity_estimator.state_dict()[k]) for k, v in estimator.items())
        assert estimator_changed == learner.estimator_trainable
        learner.save(output/'validation_policy.zip')
        loaded = PPO.load(output/'validation_policy.zip', device='cpu')
        resumed = VelocityEstimationPPO.load(output/'validation_policy.zip', device='cpu')
        assert resumed.policy.estimator_optimizer.state_dict()['state']
        obs = vector.reset()
        np.testing.assert_array_equal(learner.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
        with torch.no_grad():
            # Host and deployment actor see no privileged training tail.
            runtime = torch.from_numpy(obs[:, :180].copy())
            expected = torch.clamp(actor(torch.from_numpy(obs)), -1., 1.)
            torch.testing.assert_close(torch.clamp(actor(runtime), -1., 1.), expected, rtol=0, atol=0)
        after = {mode: evaluate(output/'validation_policy.zip', args.config, xml, episodes=2,
            seed=112000, reset_mode=mode, push_episode_probability=0.) for mode in ('folded', 'standing')}
        report = dict(simulation_only=True, formal_training=False, timesteps=learner.num_timesteps,
            physics_backend='mujoco_warp' if gpu else 'mujoco',
            training_input_dimensions=183, runtime_actor_input_dimensions=180,
            rising_actor_frozen=learner.policy.freeze_rising_actor,
            rising_weights_unchanged=frozen_unchanged, standing_weights_updated=True,
            estimator_trainable=learner.estimator_trainable, estimator_weights_updated=estimator_changed,
            standing_rehearsal=rehearsal,
            ordinary_ppo_reload=True, estimator_optimizer_restored=True, runtime_without_labels=True,
            model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(), before=before, after=after,
            note='Bounded integration test; this is not a formal success-rate comparison')
        (output/'summary.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k not in ('before', 'after')}, indent=2))
    finally:
        vector.close()


if __name__ == '__main__':
    main()
