"""History timing, checkpoint migration and recovery shaping contracts."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from climber_legged_rl.observation_history import ObservationHistory, wrap_observations
from climber_legged_rl.balance_rewards import BalanceRecoveryReward
from climber_legged_rl.history_policy import HistoryStandingPolicy, copy_history_parent
from climber_legged_rl.phase_split_policy import PhaseSplitStandingPolicy, copy_parent_policy
from climber_legged_rl.mujoco_training import load_warm_start


class FrameEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-10., 10., (36,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (6,), dtype=np.float32)
        self.model = SimpleNamespace(geom_friction=np.ones((5, 3)), geom_bodyid=np.array([1, 1, 2, 2, 0]))
        self.wheel_geoms = [0, 2]
        self.wheel_geom_ids = {0, 1, 2, 3}
        self.wheel_ids = [1, 2]
        self.ground_id = 4
        self.max_steps = 8
        self.pose_bank = [1, 2]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(36, dtype=np.float32), {}

    def step(self, action):
        self.last_action = action.copy()
        self.steps += 1
        obs = np.full(36, self.steps / 10., dtype=np.float32)
        obs[-1] = .45 if self.steps < 3 else 1.
        return obs, float(-np.square(action - .25).sum()), False, self.steps == self.max_steps, {}


def wrapped(settings=None):
    return ObservationHistory(FrameEnv(), settings or {'observations': {'history_frames': 5}})


class HistoryEnvironmentTests(unittest.TestCase):
    def test_old_config_returns_original_env_without_changes(self):
        env = FrameEnv()
        self.assertIs(wrap_observations(env, {}), env)

    def test_oldest_to_newest_once_per_step_and_reset_clears_episode(self):
        env = wrapped()
        initial, _ = env.reset(seed=1)
        np.testing.assert_array_equal(initial, np.zeros(180))
        for n in range(1, 4):
            obs, *_ = env.step(np.full(6, .1 * n))
        np.testing.assert_allclose(obs.reshape(5, 36)[:, 0], [0., 0., .1, .2, .3])
        np.testing.assert_allclose(obs[-36 + 18:-36 + 24], np.full(6, .3))
        self.assertEqual(obs[-1], 1.)
        obs, _ = env.reset(seed=2)
        np.testing.assert_array_equal(obs, np.zeros(180))

    def test_action_delay_and_command_memory_have_explicit_timing(self):
        env = wrapped({'observations': {'history_frames': 5}, 'robustness': {'action_delay_steps': [2, 2]}})
        env.reset(seed=1)
        for command, applied in ((.1, 0.), (.2, 0.), (.3, .1), (.4, .2)):
            obs, *_ = env.step(np.full(6, command))
            np.testing.assert_allclose(env.unwrapped.last_action, np.full(6, applied))
            np.testing.assert_allclose(obs[-36 + 18:-36 + 24], np.full(6, command))
        env.reset(seed=1)
        env.step(np.ones(6))
        np.testing.assert_array_equal(env.unwrapped.last_action, np.zeros(6))

    def test_noise_and_domain_randomization_are_seeded_and_restore_nominal_friction(self):
        config = {'observations': {'history_frames': 5}, 'robustness': {
            'observation_noise': {'angular_velocity': .1}, 'wheel_friction_multiplier': [.8, 1.2],
            'motor_strength': [.9, 1.1]}}
        env = wrapped(config)
        a, info_a = env.reset(seed=3)
        friction = env.model.geom_friction.copy()
        env.step(np.ones(6))
        b, info_b = env.reset(seed=3)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(env.model.geom_friction, friction)
        np.testing.assert_array_equal(friction[0], friction[1])
        np.testing.assert_array_equal(friction[2], friction[3])
        self.assertAlmostEqual(friction[4, 0], min(friction[0, 0], friction[2, 0]))
        self.assertEqual(info_a, info_b)
        self.assertTrue((np.abs(a.reshape(5, 36)[:, :3]) <= .025).all())
        np.testing.assert_array_equal(a.reshape(5, 36)[:, 3:], np.zeros((5, 33)))

    def test_task_overrides_forward_to_inner_env(self):
        env = wrapped()
        env.max_steps = 2
        env.pose_bank = [2]
        self.assertEqual(env.unwrapped.max_steps, 2)
        self.assertEqual(env.unwrapped.pose_bank, [2])

    def test_invalid_pipeline_settings_are_rejected(self):
        for settings in ({'observations': {'history_frames': True}},
                         {'robustness': {'action_delay_steps': [2, 1]}},
                         {'robustness': {'observation_noise': {'typo': 1.}}},
                         {'robustness': {'motor_strength': [-1., 1.]}}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                wrapped(settings)


class HistoryPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.kwargs = dict(net_arch=dict(pi=[16, 8], vf=[12]), activation_fn=torch.nn.ReLU)
        self.parent = PPO('MlpPolicy', FrameEnv(), n_steps=8, batch_size=8, n_epochs=1,
                          policy_kwargs=self.kwargs, seed=8, device='cpu')
        self.env = wrapped()
        self.history = PPO(HistoryStandingPolicy, self.env, n_steps=8, batch_size=8,
                           n_epochs=1, policy_kwargs=self.kwargs, seed=9, device='cpu')
        copy_history_parent(self.parent.policy, self.history.policy)

    def tearDown(self):
        self.env.close()
        self.parent.env.close()

    def test_migration_preserves_actions_and_values_with_arbitrary_old_frames(self):
        observations = torch.randn(32, 180)
        observations[:, -1] = torch.tensor([0., .45, .9, 1.] * 8)
        latest = observations[:, -36:]
        with torch.no_grad():
            torch.testing.assert_close(self.parent.policy.get_distribution(latest).distribution.mean,
                self.history.policy.get_distribution(observations).distribution.mean, rtol=0, atol=1e-7)
            torch.testing.assert_close(self.parent.policy.predict_values(latest),
                self.history.policy.predict_values(observations), rtol=0, atol=1e-7)

    def test_phase_split_parent_keeps_its_two_distinct_branches(self):
        split = PPO(PhaseSplitStandingPolicy, FrameEnv(), n_steps=8, batch_size=8,
                    n_epochs=1, policy_kwargs=self.kwargs, seed=7)
        copy_parent_policy(self.parent.policy, split.policy)
        with torch.no_grad():
            split.policy.mlp_extractor.policy_net.standing_head.bias.add_(.1)
        copy_history_parent(split.policy, self.history.policy)
        obs = np.random.default_rng(1).normal(size=(16, 180)).astype(np.float32)
        obs[:, -1] = np.tile([.5, 1.], 8)
        np.testing.assert_allclose(split.predict(obs[:, -36:], deterministic=True)[0],
            self.history.predict(obs, deterministic=True)[0], rtol=0, atol=1e-7)

    def test_ppo_save_load_and_export_keep_frozen_rising_outputs(self):
        obs = torch.randn(16, 180)
        obs[:, -1] = .5
        actor = self.history.policy.mlp_extractor.policy_net
        before = {k: v.clone() for k, v in actor.state_dict().items() if k.startswith('rising_')}
        rising = self.history.policy.get_distribution(obs).distribution.mean.detach().clone()
        standing_weight = actor.standing_net[0].weight.detach().clone()
        self.history.learn(total_timesteps=32)
        for key, value in before.items():
            torch.testing.assert_close(actor.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(self.history.policy.get_distribution(obs).distribution.mean, rising, rtol=0, atol=0)
        self.assertFalse(torch.equal(standing_weight, actor.standing_net[0].weight))
        self.assertGreater(int(torch.count_nonzero(actor.standing_net.past_projection.weight)), 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.zip'
            self.history.save(path)
            loaded = PPO.load(path, device='cpu')
            np.testing.assert_array_equal(loaded.predict(obs.numpy(), deterministic=True)[0],
                self.history.predict(obs.numpy(), deterministic=True)[0])
            traced = torch.jit.trace(loaded.policy.mlp_extractor.policy_net, obs[:1])
            torch.testing.assert_close(traced(obs), actor(obs), rtol=0, atol=1e-7)

    def test_warm_start_migrates_and_reloads_history_checkpoint(self):
        settings = dict(learning_rate=1e-5, ent_coef=0., n_steps=8, batch_size=8, n_epochs=1,
            gamma=.99, gae_lambda=.95, clip_range=.1, max_grad_norm=.5,
            history_standing_actor=True, reset_optimizer_on_warm_start=True)
        vector = DummyVecEnv([wrapped])
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'parent.zip'
                self.parent.save(path)
                converted, _, expanded = load_warm_start(path, vector, settings, 'cpu', 10)
                self.assertTrue(expanded)
                converted.save(path)
                continued, _, expanded = load_warm_start(path, vector, settings, 'cpu', 11)
                self.assertFalse(expanded)
                np.testing.assert_array_equal(converted.predict(np.zeros(180), deterministic=True)[0],
                    continued.predict(np.zeros(180), deterministic=True)[0])
        finally:
            vector.close()


class RecoveryRewardTests(unittest.TestCase):
    @staticmethod
    def settings():
        settings = {'enabled': True}
        for name in ('position', 'velocity', 'heading', 'yaw_rate'):
            settings[name + '_scale'] = .1
            settings[name + '_weight'] = 1.
        return settings

    def test_attitude_huber_has_continuous_signal_and_excludes_yaw_rate(self):
        settings = self.settings()
        settings.update(tilt_scale=.1, tilt_weight=3., tilt_rate_scale=.5, tilt_rate_weight=.5)
        reward = BalanceRecoveryReward(settings)
        zero = reward.terms([0, 0], [0, 0], 0., 0., tilt=0., tilt_rate=0.)
        self.assertEqual(sum(zero.values()), 0.)
        small = reward.terms([0, 0], [0, 0], 0., 0., tilt=.05, tilt_rate=.25)
        self.assertAlmostEqual(small['recovery_tilt'], -.375)
        self.assertAlmostEqual(small['recovery_tilt_rate'], -.0625)
        large = reward.terms([0, 0], [0, 0], 0., 0., tilt=.3, tilt_rate=1.)
        self.assertAlmostEqual(large['recovery_tilt'], -7.5)
        self.assertAlmostEqual(large['recovery_tilt_rate'], -.75)
        yaw = reward.terms([0, 0], [0, 0], 0., 1., tilt=.3, tilt_rate=1.)
        self.assertEqual(yaw['recovery_tilt_rate'], large['recovery_tilt_rate'])
        self.assertLess(yaw['recovery_yaw_rate'], large['recovery_yaw_rate'])

    def test_old_configuration_ignores_attitude_measurements(self):
        reward = BalanceRecoveryReward(self.settings())
        expected = reward.terms([.2, 0], [.1, .1], .2, .3)
        self.assertEqual(expected, reward.terms([.2, 0], [.1, .1], .2, .3, tilt=.4, tilt_rate=1.))
        self.assertNotIn('recovery_tilt', expected)

    def test_invalid_attitude_settings_and_missing_measurements_fail(self):
        for extra in ({'tilt_scale': .1}, {'tilt_weight': 1.},
                      {'tilt_scale': 0., 'tilt_weight': 1.},
                      {'tilt_rate_scale': .5, 'tilt_rate_weight': float('nan')}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                BalanceRecoveryReward(dict(self.settings(), **extra))
        reward = BalanceRecoveryReward(dict(self.settings(), tilt_scale=.1, tilt_weight=1.))
        for tilt in (None, -.1, float('nan'), float('inf')):
            with self.subTest(tilt=tilt), self.assertRaises(ValueError):
                reward.terms([0, 0], [0, 0], 0., 0., tilt=tilt)

    def test_broad_penalty_distinguishes_large_errors_and_is_zero_at_target(self):
        settings = {'enabled': True}
        for name in ('position', 'velocity', 'heading', 'yaw_rate'):
            settings[name + '_scale'] = .1
            settings[name + '_weight'] = 1.
        reward = BalanceRecoveryReward(settings)
        self.assertEqual(sum(reward.terms([0, 0], [0, 0], 0., 0.).values()), 0.)
        small = reward.terms([.08, 0], [0, 0], 0., 0.)['recovery_position']
        large = reward.terms([.2, 0], [0, 0], 0., 0.)['recovery_position']
        self.assertLess(large, small)
        self.assertLess(small, 0.)
        self.assertEqual(BalanceRecoveryReward({}).terms([], [], 0, 0), {})


if __name__ == '__main__':
    unittest.main()
