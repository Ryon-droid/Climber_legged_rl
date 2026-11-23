"""Verify real pulse limits, episode boundaries, and exact-replay accounting."""
import copy
import unittest

import numpy as np

from test_exact_push_replay import ForceEnv
from climber_legged_rl.push_reset_replay import ExactPushResetReplay
from climber_legged_rl.recovery_force_curriculum import RecoveryForceCurriculum


class StepForceEnv(ForceEnv):
    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0., False, False, {}


class ForceCurriculumTests(unittest.TestCase):
    def make(self, push=True):
        base = StepForceEnv()
        replay = ExactPushResetReplay(base, [dict(seed=17, force_N=[18., 18.], push=push)], .99, 12)
        return RecoveryForceCurriculum(replay, ((0, 12.), (2, 15.), (4, 18.)))

    def assert_pulses(self, env, cap):
        forces = [np.linalg.norm(env.unwrapped.push.force(i, 2000)) for i in range(2000)]
        self.assertGreater(max(forces), 0.)
        self.assertLessEqual(max(forces), cap + 1e-10)
        self.assertGreaterEqual(len(env.unwrapped.push.events), 2)

    def test_replay_cap_applies_to_first_and_later_pulses_and_only_changes_at_reset(self):
        env = self.make()
        for cap in (12., 15., 18.):
            _, info = env.reset()
            self.assertTrue(info['training_hard_push_reset'])
            self.assertEqual(info['training_exact_force_replay'], cap == 18.)
            self.assertEqual(info['training_force_scaled_replay'], cap < 18.)
            self.assert_pulses(env, cap)
            for _ in range(2):
                *_, info = env.step(np.zeros(1))
                self.assertEqual(info['training_force_cap_N'], cap)
            self.assertEqual(env.unwrapped.push.config['force_N'], [cap, cap])
        self.assertEqual(env.cases[0]['force_N'], [18., 18.])

    def test_explicit_resets_use_capped_bands_and_restore_original_at_final_stage(self):
        env = self.make()
        original = copy.deepcopy(env.original)
        _, info = env.reset(seed=41)
        self.assertFalse(info['training_hard_push_reset'])
        self.assertEqual(env.unwrapped.push.config['force_bands'][0]['force_N'], [2., 12.])
        env.transitions = 4
        env.reset(seed=41)
        self.assertEqual(env.unwrapped.push.config, original)
        self.assertEqual(env.original, original)

    def test_no_push_replay_remains_no_push(self):
        env = self.make(push=False)
        env.reset()
        for step in range(2000):
            np.testing.assert_array_equal(env.unwrapped.push.force(step, 2000), np.zeros(3))

    def test_invalid_schedule_rejected(self):
        for schedule in ((), ((1, 12.),), ((0, 12.), (0, 15.)), ((0, 19.),),
                         ((0, 15.), (2, 12.)), ((0, float('nan')),)):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                RecoveryForceCurriculum(self.make().env, schedule)


if __name__ == '__main__':
    unittest.main()
