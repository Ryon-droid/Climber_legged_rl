from pathlib import Path
import sys
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from climber_legged_rl.phase_split_policy import PhaseSplitStandingPolicy, copy_parent_policy
from climber_legged_rl.mujoco_training import load_warm_start


class PhaseEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-10., 10., (36,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (6,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(36, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        obs = np.full(36, .1, dtype=np.float32)
        obs[-1] = .45 if self.steps < 3 else 1.
        return obs, float(-np.square(action - .25).sum()), False, self.steps == 8, {}


class PhaseSplitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = PhaseEnv()
        self.kwargs = dict(net_arch=dict(pi=[16, 8], vf=[12]), activation_fn=torch.nn.ReLU)
        self.parent = PPO('MlpPolicy', self.env, n_steps=8, batch_size=8, n_epochs=1,
                          policy_kwargs=self.kwargs, seed=8, device='cpu')
        self.split = PPO(PhaseSplitStandingPolicy, self.env, n_steps=8, batch_size=8,
                         n_epochs=1, policy_kwargs=self.kwargs, seed=9, device='cpu')
        copy_parent_policy(self.parent.policy, self.split.policy)

    def tearDown(self):
        self.env.close()

    def test_conversion_preserves_all_actions_and_values_in_both_phases(self):
        observations = torch.randn(32, 36)
        observations[:, -1] = torch.tensor([0., .45, .9, 1.] * 8)
        with torch.no_grad():
            a = self.parent.policy.get_distribution(observations).distribution.mean
            b = self.split.policy.get_distribution(observations).distribution.mean
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(self.parent.policy.predict_values(observations),
                self.split.policy.predict_values(observations), rtol=0, atol=0)

    def test_standing_gradients_cannot_modify_rising_branch(self):
        observations = torch.randn(16, 36)
        observations[:, -1] = 1.
        rising_observations = observations.clone()
        rising_observations[:, -1] = .9
        old_rising = self.split.policy.get_distribution(rising_observations).distribution.mean.detach().clone()
        old_standing = self.split.policy.get_distribution(observations).distribution.mean.detach().clone()
        for _ in range(4):
            loss = (self.split.policy.get_distribution(observations).distribution.mean - .5).square().mean()
            self.split.policy.optimizer.zero_grad()
            loss.backward()
            self.split.policy.optimizer.step()
        torch.testing.assert_close(self.split.policy.get_distribution(rising_observations).distribution.mean,
            old_rising, rtol=0, atol=0)
        self.assertFalse(torch.equal(old_standing, self.split.policy.get_distribution(observations).distribution.mean))
        actor = self.split.policy.mlp_extractor.policy_net
        self.assertTrue(all(not p.requires_grad for net in (actor.rising_net, actor.rising_head) for p in net.parameters()))

    def test_real_ppo_update_save_and_reload_preserve_frozen_branch(self):
        actor = self.split.policy.mlp_extractor.policy_net
        before = {k: v.clone() for k, v in actor.state_dict().items() if k.startswith('rising_')}
        self.split.learn(total_timesteps=16)
        for key, value in before.items():
            torch.testing.assert_close(actor.state_dict()[key], value, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'policy.zip'
            self.split.save(path)
            loaded = PPO.load(path, device='cpu')
            observations = np.random.default_rng(2).normal(size=(12, 36)).astype(np.float32)
            observations[:, -1] = np.tile([0., .9, 1.], 4)
            np.testing.assert_array_equal(self.split.predict(observations, deterministic=True)[0],
                loaded.predict(observations, deterministic=True)[0])
            self.assertIsInstance(loaded.policy, PhaseSplitStandingPolicy)
            self.assertTrue(all(not p.requires_grad for p in loaded.policy.mlp_extractor.policy_net.rising_head.parameters()))

    def test_warm_start_conversion_and_subsequent_warm_start_keep_architecture(self):
        settings = dict(learning_rate=1e-5, ent_coef=0., n_steps=8, batch_size=8, n_epochs=1,
            gamma=.99, gae_lambda=.95, clip_range=.1, max_grad_norm=.5,
            phase_split_standing_actor=True, reset_optimizer_on_warm_start=True)
        vector = DummyVecEnv([PhaseEnv])
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'parent.zip'
                self.parent.save(path)
                converted, _, expanded = load_warm_start(path, vector, settings, 'cpu', 10)
                self.assertFalse(expanded)
                self.assertIsInstance(converted.policy, PhaseSplitStandingPolicy)
                converted.save(path)
                continued, _, _ = load_warm_start(path, vector, settings, 'cpu', 11)
                self.assertIsInstance(continued.policy, PhaseSplitStandingPolicy)
                obs = np.zeros((1, 36), dtype=np.float32)
                np.testing.assert_array_equal(converted.predict(obs, deterministic=True)[0],
                    continued.predict(obs, deterministic=True)[0])
        finally:
            vector.close()


if __name__ == '__main__':
    unittest.main()
