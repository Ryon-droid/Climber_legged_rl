"""Seeded force episodes must stay training-only and preserve the force law."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from climber_legged_rl.push_reset_replay import PushResetReplay


class ResetEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)
        self.push = SimpleNamespace(enabled=True, config=dict(episode_probability=.4,
            force_N=[2., 15.], duration_seconds=[.05, .1]))
        self.calls = []
        self.fail = False

    def reset(self, *, seed=None, options=None):
        self.calls.append((seed, options, dict(self.push.config)))
        if self.fail:
            raise RuntimeError('reset failed')
        return np.zeros(1, dtype=np.float32), {}


class PushReplayTests(unittest.TestCase):
    def make(self, seed=12):
        return PushResetReplay(ResetEnv(), dict(probability=.2, seeds=[261106]), seed)

    def test_forces_push_only_during_selected_resets(self):
        env = self.make()
        original = env.unwrapped.push.config
        flags = []
        for _ in range(100):
            _, info = env.reset()
            flags.append(info['training_hard_push_reset'])
            seed, options, settings = env.unwrapped.calls[-1]
            if flags[-1]:
                self.assertEqual((seed, options), (261106, {'reset_mode': 'standing'}))
                self.assertEqual(settings, dict(original, episode_probability=1.))
            else:
                self.assertEqual((seed, options, settings), (None, None, original))
            self.assertIs(env.unwrapped.push.config, original)
        self.assertTrue(0 < sum(flags) < 100)

    def test_explicit_evaluation_resets_bypass_replay(self):
        env = self.make()
        for seed, options in [(23, None), (None, {'reset_mode': 'folded'}), (0, {})]:
            _, info = env.reset(seed=seed, options=options)
            self.assertFalse(info['training_hard_push_reset'])
            self.assertEqual(env.unwrapped.calls[-1][:2], (seed, options))
            self.assertEqual(env.unwrapped.calls[-1][2]['episode_probability'], .4)

    def test_worker_choices_are_reproducible(self):
        a, b = self.make(), self.make()
        for _ in range(40):
            a.reset()
            b.reset()
        self.assertEqual(a.unwrapped.calls, b.unwrapped.calls)

    def test_original_config_restored_when_reset_fails(self):
        env = self.make()
        env.unwrapped.fail = True
        original = env.unwrapped.push.config
        for _ in range(100):
            with self.assertRaises(RuntimeError):
                env.reset()
            self.assertIs(env.unwrapped.push.config, original)
        self.assertTrue(any(c[2]['episode_probability'] == 1. for c in env.unwrapped.calls))

    def test_invalid_settings_are_rejected(self):
        for probability, seeds in [(0., [1]), (1., [1]), (float('nan'), [1]),
                                    (.2, []), (.2, [-1]), (.2, [1, 1]), (.2, [True])]:
            with self.subTest(probability=probability, seeds=seeds), self.assertRaises(ValueError):
                PushResetReplay(ResetEnv(), dict(probability=probability, seeds=seeds), 1)
        env = ResetEnv()
        env.push.enabled = False
        with self.assertRaises(ValueError):
            PushResetReplay(env, dict(probability=.2, seeds=[1]), 1)
