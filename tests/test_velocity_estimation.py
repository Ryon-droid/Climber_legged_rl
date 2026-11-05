"""Information boundaries, estimator learning, migration and checkpoint contracts."""
from pathlib import Path
import sys
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from climber_legged_rl.velocity_estimation import (VelocityEstimatedPolicy,
    VelocityEstimationPPO, VelocitySupervision, copy_estimation_parent, estimator_update)
from climber_legged_rl.history_policy import HistoryStandingPolicy, copy_history_parent


class StudentEnv(gym.Env):
    def __init__(self, dimensions=183):
        self.observation_space = gym.spaces.Box(-100., 100., (dimensions,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (6,), dtype=np.float32)
        self.n = 0

    def _kinematics(self):
        return np.zeros(3), np.array([.2, -.1, .3]), np.zeros(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.n = 0
        return self.np_random.normal(size=self.observation_space.shape).astype(np.float32), {}

    def step(self, action):
        self.n += 1
        obs = self.np_random.normal(size=self.observation_space.shape).astype(np.float32)
        if len(obs) == 183:
            obs[179] = 1.
            obs[-3:] = obs[:3] * .2
        return obs, float(-np.square(action - .2).sum()), False, self.n == 16, {}


def model(freeze=True):
    torch.manual_seed(2)
    return VelocityEstimationPPO(VelocityEstimatedPolicy, StudentEnv(), n_steps=32,
        batch_size=16, n_epochs=1, learning_rate=.001, seed=2, device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[16, 8], vf=[16, 8]), freeze_rising_actor=freeze))


class VelocityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_masks_every_frame_without_mutating_history_and_labels_physical_velocity(self):
        env = VelocitySupervision(StudentEnv(180))
        obs, _ = env.reset(seed=1)
        np.testing.assert_array_equal(obs[:180].reshape(5, 36)[:, 28:30], np.zeros((5, 2)))
        np.testing.assert_allclose(obs[-3:], [.2, -.1, .3])
        original = np.ones(180, np.float32)
        env._augment(original)
        np.testing.assert_array_equal(original, np.ones(180))
        self.assertEqual(env.observation_space.shape, (183,))

    def test_privileged_labels_and_velocity_slots_cannot_change_actor(self):
        policy = model().policy
        x = torch.randn(12, 183)
        y = x.clone()
        y[:, -3:] += 20.
        for i in range(5):
            y[:, i * 36 + 28:i * 36 + 30] += 20.
        with torch.no_grad():
            actor = policy.mlp_extractor.policy_net
            torch.testing.assert_close(actor(x), actor(y), rtol=0, atol=0)
            torch.testing.assert_close(actor(x[:, :180]), actor(x), rtol=0, atol=0)
            torch.testing.assert_close(actor.estimate_velocity(x), actor.estimate_velocity(y), rtol=0, atol=0)
            self.assertFalse(torch.equal(policy.predict_values(x), policy.predict_values(y)))

    def test_supervised_learning_only_changes_estimator_and_reduces_heldout_error(self):
        policy = model().policy
        x = torch.randn(1024, 183)
        x[:, -3:] = x[:, :3] * .2
        actor = policy.mlp_extractor.policy_net
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        with torch.no_grad():
            initial = (actor.estimate_velocity(x[768:]) - x[768:, -3:]).square().mean()
        estimator_update(policy, x[:768], epochs=30, batch_size=128)
        with torch.no_grad():
            final = (actor.estimate_velocity(x[768:]) - x[768:, -3:]).square().mean()
        self.assertLess(final, initial / 5)
        for key, value in before.items():
            if 'velocity_estimator' not in key:
                torch.testing.assert_close(value, policy.state_dict()[key], rtol=0, atol=0)
        policy.zero_grad(set_to_none=True)
        actor(x).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in actor.velocity_estimator.parameters()))

    def test_standard_ppo_reload_and_estimator_optimizer_resume(self):
        learner = model()
        learner.configure_estimator({'epochs': 2, 'batch_size': 16})
        frozen = {k: v.clone() for k, v in learner.policy.state_dict().items() if 'rising_' in k}
        learner.learn(64)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'policy.zip'
            learner.save(path)
            ordinary = PPO.load(path, device='cpu')
            resumed = VelocityEstimationPPO.load(path, env=StudentEnv(), device='cpu')
            x, _ = StudentEnv().reset(seed=11)
            np.testing.assert_array_equal(learner.predict(x, deterministic=True)[0], ordinary.predict(x, deterministic=True)[0])
            state = resumed.policy.estimator_optimizer.state_dict()
            self.assertGreater(len(state['state']), 0)
            resumed.learn(32, reset_num_timesteps=False)
            for key, value in frozen.items():
                torch.testing.assert_close(value, resumed.policy.state_dict()[key], rtol=0, atol=0)

    def test_ordinary_and_history_parents_migrate_control_weights(self):
        parent = PPO('MlpPolicy', StudentEnv(36), device='cpu', policy_kwargs=dict(net_arch=dict(pi=[16, 8], vf=[16, 8])))
        student = model()
        copy_estimation_parent(parent.policy, student.policy)
        actor = student.policy.mlp_extractor.policy_net
        torch.testing.assert_close(parent.policy.action_net.weight, actor.standing_head.weight)
        history = PPO(HistoryStandingPolicy, StudentEnv(180), device='cpu', policy_kwargs=parent.policy_kwargs)
        copy_history_parent(parent.policy, history.policy)
        copy_estimation_parent(history.policy, student.policy)
        torch.testing.assert_close(history.policy.mlp_extractor.policy_net.standing_head.weight, actor.standing_head.weight)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.zip'
            parent.save(path)
            # Loading a non-estimator parent must not require estimator metadata.
            loaded = VelocityEstimationPPO.load(path, device='cpu')
            self.assertEqual(loaded.observation_space.shape, (36,))

    def test_adaptation_can_train_rising_branch_and_keeps_setting_on_reload(self):
        learner = model(freeze=False)
        actor = learner.policy.mlp_extractor.policy_net
        self.assertTrue(all(p.requires_grad for p in actor.rising_net.parameters()))
        x = torch.randn(16, 183)
        x[:, 179] = .5
        optimizer = learner.policy.optimizer
        before = actor.rising_head.weight.detach().clone()
        optimizer.zero_grad()
        actor(x).square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, actor.rising_head.weight))
        with tempfile.TemporaryDirectory() as folder:
            learner.save(Path(folder)/'policy.zip')
            loaded = PPO.load(Path(folder)/'policy.zip', device='cpu')
            self.assertFalse(loaded.policy.freeze_rising_actor)
            self.assertTrue(loaded.policy.mlp_extractor.policy_net.rising_head.weight.requires_grad)

    def test_warm_start_can_freeze_both_estimator_and_rising_without_changing_actions(self):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from climber_legged_rl.mujoco_training import load_warm_start
        parent = model(freeze=False)
        parent.configure_estimator({'epochs': 1, 'batch_size': 16})
        parent.learn(32)
        settings = dict(learning_rate=.001, ent_coef=0., n_steps=32, batch_size=16,
            n_epochs=1, gamma=.99, gae_lambda=.95, clip_range=.2, max_grad_norm=.5,
            history_standing_actor=True, freeze_rising_actor=True,
            velocity_estimator={'enabled': True, 'trainable': False, 'epochs': 1, 'batch_size': 16})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'parent.zip'
            parent.save(path)
            vector = DummyVecEnv([lambda: StudentEnv()])
            try:
                learner, _, _ = load_warm_start(path, vector, settings, 'cpu', 11)
                learner.configure_estimator(settings['velocity_estimator'])
                observations = np.random.default_rng(2).normal(size=(32, 183)).astype(np.float32)
                observations[:16, 179] = .5
                observations[16:, 179] = 1.
                np.testing.assert_array_equal(parent.predict(observations, deterministic=True)[0],
                    learner.predict(observations, deterministic=True)[0])
                actor = learner.policy.mlp_extractor.policy_net
                frozen = {k: v.clone() for k, v in actor.state_dict().items()
                    if k.startswith(('rising_', 'velocity_estimator.'))}
                standing = actor.standing_head.weight.detach().clone()
                learner.learn(64)
                for key, value in frozen.items():
                    torch.testing.assert_close(value, actor.state_dict()[key], rtol=0, atol=0)
                self.assertFalse(torch.equal(standing, actor.standing_head.weight))
                learner.save(Path(folder)/'frozen.zip')
                loaded = VelocityEstimationPPO.load(Path(folder)/'frozen.zip', env=StudentEnv(), device='cpu')
                self.assertFalse(loaded.estimator_trainable)
                self.assertTrue(loaded.policy.freeze_rising_actor)
                self.assertTrue(all(not p.requires_grad for p in loaded.policy.mlp_extractor.policy_net.velocity_estimator.parameters()))
                loaded.learn(32, reset_num_timesteps=False)
                for key, value in frozen.items():
                    torch.testing.assert_close(value, loaded.policy.mlp_extractor.policy_net.state_dict()[key], rtol=0, atol=0)
            finally:
                vector.close()


if __name__ == '__main__':
    unittest.main()
