from pathlib import Path
import sys
import unittest
import gymnasium as gym
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from climber_legged_rl.standing_reset_replay import StandingResetReplay


class ResetEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)
        self.calls = []

    def reset(self, *, seed=None, options=None):
        self.calls.append((seed, options))
        return np.zeros(1, dtype=np.float32), {}


class StandingReplayTests(unittest.TestCase):
    def test_scaled_initial_states_restore_settings_and_keep_full_difficulty(self):
        class ScalingEnv(ResetEnv):
            def __init__(self):
                super().__init__()
                self.config = {'task': dict(reset_tilt_std_rad=.008, reset_velocity_std=.005, other=10)}
                self.reset_tasks = []
                self.fail = False
            def reset(self, **kwargs):
                self.reset_tasks.append(dict(self.config['task']))
                if self.fail:
                    raise RuntimeError('reset failed')
                return super().reset(**kwargs)
        settings = dict(probability=.9, seeds=[261409], initial_state_scale_range=[.5, .95],
                        full_state_probability=.25)
        env = StandingResetReplay(ScalingEnv(), settings, 12)
        task = env.unwrapped.config['task']
        original = dict(task)
        scales = []
        for _ in range(100):
            _, info = env.reset()
            self.assertEqual(task, original)
            scale = info['training_hard_standing_state_scale']
            if info['training_hard_standing_reset']:
                scales.append(scale)
            actual = env.unwrapped.reset_tasks[-1]
            self.assertAlmostEqual(actual['reset_tilt_std_rad'], .008 * scale)
            self.assertAlmostEqual(actual['reset_velocity_std'], .005 * scale)
            self.assertEqual(actual['other'], 10)
        self.assertTrue(any(.5 <= s <= .95 for s in scales))
        self.assertIn(1., scales)
        for seed, options in [(261409, {'reset_mode': 'standing'}), (None, {'reset_mode': 'folded'})]:
            _, info = env.reset(seed=seed, options=options)
            self.assertFalse(info['training_hard_standing_reset'])
            self.assertEqual(info['training_hard_standing_state_scale'], 1.)
            self.assertEqual(env.unwrapped.reset_tasks[-1], original)
        env.unwrapped.fail = True
        before = len(env.unwrapped.reset_tasks)
        for _ in range(20):
            with self.assertRaises(RuntimeError):
                env.reset()
            self.assertEqual(task, original)
        self.assertTrue(any(t['reset_tilt_std_rad'] < .008 for t in env.unwrapped.reset_tasks[before:]))

    def test_invalid_initial_state_curriculum_rejected(self):
        for bounds, probability in [([0., 1.], .25), ([.5, 1.1], .25), ([1., .5], .25),
                                    ([.5], .25), ([float('nan'), 1.], .25), ([.5, 1.], -1.),
                                    ([.5, 1.], float('nan'))]:
            with self.subTest(bounds=bounds, probability=probability), self.assertRaises(ValueError):
                StandingResetReplay(ResetEnv(), dict(probability=.3, seeds=[1],
                    initial_state_scale_range=bounds, full_state_probability=probability), 12)

    def make(self, seed=12):
        return StandingResetReplay(ResetEnv(), dict(probability=.3, seeds=[5, 11, 14]), seed)

    def test_explicit_seed_and_mode_are_preserved(self):
        env = self.make()
        _, info = env.reset(seed=55, options={'reset_mode': 'folded'})
        self.assertEqual(env.unwrapped.calls[-1], (55, {'reset_mode': 'folded'}))
        self.assertFalse(info['training_hard_standing_reset'])

    def test_hard_seeds_force_standing_and_keep_random_resets(self):
        env = self.make()
        for _ in range(100):
            env.reset()
        hard = [call for call in env.unwrapped.calls if call[0] is not None]
        self.assertTrue(0 < len(hard) < 100)
        for seed, options in hard:
            self.assertIn(seed, [5, 11, 14])
            self.assertEqual(options, {'reset_mode': 'standing'})

    def test_worker_schedule_is_reproducible(self):
        a, b = self.make(), self.make()
        for _ in range(40):
            a.reset(); b.reset()
        self.assertEqual(a.unwrapped.calls, b.unwrapped.calls)

    def test_weighted_replay_emphasizes_failure_and_keeps_other_seeds(self):
        settings = dict(probability=.3, seeds=[5, 11, 14], seed_probabilities=[.05, .05, .9])
        a = StandingResetReplay(ResetEnv(), settings, 12)
        b = StandingResetReplay(ResetEnv(), settings, 12)
        chosen = []
        for _ in range(1000):
            _, info = a.reset()
            b.reset()
            if info['training_hard_standing_reset']:
                chosen.append(info['training_hard_standing_seed'])
        self.assertEqual(a.unwrapped.calls, b.unwrapped.calls)
        self.assertEqual(set(chosen), {5, 11, 14})
        self.assertGreater(chosen.count(14)/len(chosen), .8)
        before = len(a.unwrapped.calls)
        _, info = a.reset(seed=77, options={'reset_mode': 'folded'})
        self.assertEqual(a.unwrapped.calls[before], (77, {'reset_mode': 'folded'}))
        self.assertIsNone(info['training_hard_standing_seed'])

    def test_unweighted_schedule_preserves_original_rng_calls(self):
        env = self.make()
        rng = np.random.default_rng(12)
        expected = []
        for _ in range(50):
            replay = rng.random() < .3
            expected.append((int(rng.choice([5, 11, 14])), {'reset_mode': 'standing'}) if replay else (None, None))
            env.reset()
        self.assertEqual(env.unwrapped.calls, expected)

    def test_invalid_seed_probabilities_rejected(self):
        for probabilities in ([1.], [0., .5, .5], [-.1, .5, .6], [.2, .2, .2],
                              [float('nan'), .5, .5], [[.3, .3, .4]]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                StandingResetReplay(ResetEnv(), dict(probability=.3, seeds=[5, 11, 14],
                    seed_probabilities=probabilities), 12)

    def test_invalid_settings_rejected(self):
        for probability, seeds in [(0., [1]), (1., [1]), (float('nan'), [1]), (.3, []), (.3, [-1])]:
            with self.subTest(probability=probability, seeds=seeds), self.assertRaises(ValueError):
                StandingResetReplay(ResetEnv(), dict(probability=probability, seeds=seeds), 1)
