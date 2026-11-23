"""Repeated disturbances get targeted training without leaking holdout seeds."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_spring208_push_balance as trainer
from test_push_balance_selection import result


def event(recovered, x=-18.):
    return dict(recovered=recovered, force_world_N=[x, 0., 0.])


class RepeatRecoveryTests(unittest.TestCase):
    def test_standing_retention_starts_from_accepted_parent_and_trains_eighteen_before_first_check(self):
        with patch.object(trainer.ab, 'setup'):
            try:
                trainer.setup('balance', 'standing_retention18')
                self.assertEqual(trainer.PARENT.parent.name, 'push_directional_recovery18_20261116')
                self.assertEqual(trainer.PARENT_SHA, 'ec7fee00619e4240a2c6d93c1a247ec74f972a47b34deb3a8ad8e2a1478f696e')
                self.assertEqual((trainer.STEPS, trainer.INTERVAL, trainer.MAX_SECONDS), (196608, 49152, 2400))
                self.assertEqual((trainer.TRAIN_SEED, trainer.common.SELECT_SEED, trainer.common.TEST_SEED), (1520000, 1530000, 1540000))
                self.assertEqual(trainer.ab.LEARNING_RATE, 3e-6)
                self.assertEqual(trainer.force_schedule(), ((0, 15.), (3072, 18.)))
                self.assertLess(trainer.force_schedule()[-1][0] * 8, trainer.INTERVAL)
                self.assertEqual(trainer.training_push_probability(), .65)
                self.assertEqual(trainer.retention_weights()['standing'], 7)
            finally:
                trainer.setup('balance', 'directional_recovery18')

    def test_standing_retention_push_probability_and_force_cap_survive_spawn(self):
        import tempfile
        import json
        from stable_baselines3.common.vec_env import SubprocVecEnv
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'hard_training_resets.json').write_text(json.dumps(dict(cases=[
                dict(seed=17, force_N=[18., 18.], push=True),
                dict(seed=18, force_N=[18., 18.], push=False)], probability=.5)))
            with patch.object(trainer, 'PROFILE', 'standing_retention18'):
                create = trainer.factory(folder, 17, 'balance')
            vector = SubprocVecEnv([create], start_method='spawn')
            try:
                vector.reset()
                self.assertEqual(vector.get_attr('original')[0]['episode_probability'], .65)
                self.assertEqual(vector.get_attr('push')[0].config['force_N'][1], 15.)
                self.assertEqual(vector.get_attr('control')[0]['wheel_kv'], .4)
                self.assertEqual(vector.get_attr('torque_limits')[0].tolist(), [20.]*4+[2.]*2)
            finally:
                vector.close()

    def test_short_round_resets_budget_and_reaches_full_force_before_halfway(self):
        with patch.object(trainer.ab, 'setup'):
            try:
                trainer.setup('balance', 'short_directional18')
                self.assertEqual(trainer.PARENT.name, 'selected_policy.zip')
                self.assertEqual(trainer.PARENT.parent.name, 'push_directional_recovery18_20261116')
                self.assertEqual(trainer.PARENT_SHA, 'ec7fee00619e4240a2c6d93c1a247ec74f972a47b34deb3a8ad8e2a1478f696e')
                self.assertEqual((trainer.STEPS, trainer.INTERVAL, trainer.MAX_SECONDS), (196608, 49152, 1500))
                self.assertEqual((trainer.TRAIN_SEED, trainer.common.SELECT_SEED, trainer.common.TEST_SEED), (1490000, 1500000, 1510000))
                self.assertEqual(trainer.ab.LEARNING_RATE, 3e-6)
                self.assertEqual(trainer.ab.ACTION_STD, [.03]*4 + [.18]*2)
                schedule = trainer.force_schedule()
                self.assertEqual(schedule, ((0, 12.), (6144, 15.), (12288, 18.)))
                self.assertLessEqual(schedule[-1][0] * trainer.ab.NUM_ENVS, trainer.STEPS // 2)
            finally:
                trainer.setup('balance', 'directional_recovery18')
            self.assertEqual((trainer.STEPS, trainer.INTERVAL, trainer.MAX_SECONDS), (491520, 98304, 2100))

    def test_short_profile_spawn_and_episode_reset_advance_force_cap(self):
        import tempfile
        import json
        import numpy as np
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from climber_legged_rl.recovery_force_curriculum import RecoveryForceCurriculum
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'hard_training_resets.json').write_text(json.dumps(dict(cases=[
                dict(seed=17, force_N=[18., 18.], push=True)], probability=.5)))
            with patch.object(trainer, 'PROFILE', 'short_directional18'):
                create = trainer.factory(folder, 17, 'balance')
                self.assertEqual(trainer.failure_sampling_weight([event(True), event(False)]), 6)
                self.assertEqual(trainer.replay_probability(), .5)
            vector = SubprocVecEnv([create], start_method='spawn')
            try:
                vector.reset()
                self.assertEqual(vector.get_attr('control')[0]['wheel_kv'], .4)
                self.assertEqual(vector.get_attr('probability')[0], .5)
                np.testing.assert_array_equal(vector.get_attr('torque_limits')[0], [20.]*4+[2.]*2)
                self.assertEqual(vector.get_attr('push')[0].config['force_N'][1], 12.)
                self.assertEqual([b['probability'] for b in vector.get_attr('original')[0]['force_bands']], [.1, .2, .4, .3])
            finally:
                vector.close()
            env = create()
            try:
                wrapper = env
                while not isinstance(wrapper, RecoveryForceCurriculum):
                    wrapper = wrapper.env
                wrapper.env.probability = 1.
                for transitions, cap in ((0, 12.), (6144, 15.), (12288, 18.)):
                    wrapper.transitions = transitions
                    _, info = env.reset()
                    self.assertEqual(info['training_force_cap_N'], cap)
                    self.assertEqual(env.unwrapped.push.config['force_N'][1], cap)
                    self.assertEqual(info['training_exact_force_replay'], cap == 18.)
            finally:
                env.close()

    def test_prioritize_second_push_and_delayed_falls_without_multiplying_weights(self):
        with patch.object(trainer, 'PROFILE', 'repeat_recovery18'):
            self.assertEqual(trainer.failure_sampling_weight([event(True, 18.), event(False)]), 6)
            self.assertEqual(trainer.failure_sampling_weight([event(True), event(True)]), 4)
            self.assertEqual(trainer.failure_sampling_weight([event(False)]), 3)
            self.assertEqual(trainer.failure_sampling_weight([event(False, 18.)]), 1)
            self.assertEqual(trainer.failure_sampling_weight([]), 1)

    def test_existing_profiles_keep_prior_weights(self):
        for profile, expected in [('rated_balance18', 1), ('recovery_curriculum18', 3)]:
            with patch.object(trainer, 'PROFILE', profile):
                self.assertEqual(trainer.failure_sampling_weight([event(True), event(False)]), expected)

    def test_lower_force_retention_is_required_for_this_profile(self):
        baseline, after = result(), result()
        after['summary']['exact18']['success'] += 1
        after['summary']['accepted15']['success'] -= 1
        with patch.object(trainer, 'PROFILE', 'repeat_recovery18'):
            self.assertFalse(trainer.eligible(after, baseline))
            self.assertGreater(trainer.score(after), trainer.score(baseline))
        with patch.object(trainer, 'PROFILE', 'recovery_curriculum18'):
            self.assertTrue(trainer.eligible(after, baseline))

    def test_schedule_and_all_motor_limits_are_captured_by_spawned_factory(self):
        from stable_baselines3.common.vec_env import SubprocVecEnv
        import numpy as np
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'hard_training_resets.json').write_text(json.dumps(dict(cases=[
                dict(seed=17, force_N=[18., 18.], push=True)])))
            with patch.object(trainer, 'PROFILE', 'repeat_recovery18'):
                create = trainer.factory(folder, 17, 'balance')
                self.assertEqual(trainer.force_schedule(), ((0, 15.), (12288, 16.5), (24576, 18.)))
            vector = SubprocVecEnv([create], start_method='spawn')
            try:
                vector.reset()
                np.testing.assert_array_equal(vector.get_attr('torque_limits')[0], [20.] * 4 + [2.] * 2)
                push = vector.get_attr('push')[0]
                self.assertEqual(push.config['force_N'][1], 15.)
                self.assertTrue(all(b['force_N'][1] <= 15. for b in push.config.get('force_bands', [])))
            finally:
                vector.close()

    def test_balanced_reward_and_distribution_survive_spawn(self):
        import tempfile
        import json
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from climber_legged_rl.early_recovery_reward import BALANCED_RECOVERY_SETTINGS
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'hard_training_resets.json').write_text(json.dumps(dict(cases=[
                dict(seed=17, force_N=[15., 18.], push=True)], probability=.35)))
            with patch.object(trainer, 'PROFILE', 'balanced_recovery18'):
                create = trainer.factory(folder, 17, 'balance')
            vector = SubprocVecEnv([create], start_method='spawn')
            try:
                vector.reset()
                self.assertEqual(vector.get_attr('settings')[0], BALANCED_RECOVERY_SETTINGS)
                push = vector.get_attr('push')[0]
                original = vector.get_attr('original')[0]
                self.assertEqual([b['probability'] for b in original['force_bands']], [.1, .2, .4, .3])
                self.assertEqual(push.config['force_N'][1], 15.)
            finally:
                vector.close()

    def test_wheel_response_gain_is_captured_and_early_penalty_is_disabled(self):
        import tempfile
        import json
        import numpy as np
        from stable_baselines3.common.vec_env import SubprocVecEnv
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'hard_training_resets.json').write_text(json.dumps(dict(cases=[
                dict(seed=17, force_N=[15., 18.], push=True)], probability=.35)))
            with patch.object(trainer, 'PROFILE', 'wheel_response18'):
                self.assertIsNone(trainer.early_reward_settings())
                self.assertEqual(trainer.force_schedule(), ((0, 12.), (12288, 15.), (36864, 18.)))
                create = trainer.factory(folder, 17, 'balance')
            vector = SubprocVecEnv([create], start_method='spawn')
            try:
                vector.reset()
                self.assertEqual(vector.get_attr('control')[0]['wheel_kv'], .4)
                vector.step(np.zeros((1,6)))
                self.assertEqual(vector.get_attr('control')[0]['wheel_kv'], .4)
                np.testing.assert_array_equal(vector.get_attr('torque_limits')[0], [20.]*4+[2.]*2)
                self.assertEqual(vector.get_attr('push')[0].config['force_N'][1], 12.)
            finally:
                vector.close()

    def test_directional_replay_profile_survives_spawn_with_gain_and_force_cap(self):
        import tempfile
        import json
        import numpy as np
        from stable_baselines3.common.vec_env import SubprocVecEnv
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'hard_training_resets.json').write_text(json.dumps(dict(cases=[
                dict(seed=17, force_N=[15., 18.], push=True),
                dict(seed=18, force_N=[15., 18.], push=True, successful_retention=True)], probability=.5)))
            with patch.object(trainer, 'PROFILE', 'directional_recovery18'):
                self.assertIsNone(trainer.early_reward_settings())
                self.assertEqual(trainer.replay_probability(), .5)
                self.assertEqual(trainer.failure_sampling_weight([event(True), event(False)]), 6)
                self.assertIn(trainer.PROFILE, trainer.BALANCED_SELECTION_PROFILES)
                self.assertEqual(trainer.force_schedule(), ((0, 12.), (12288, 15.), (36864, 18.)))
                create = trainer.factory(folder, 17, 'balance')
            vector = SubprocVecEnv([create], start_method='spawn')
            try:
                vector.reset()
                self.assertEqual(vector.get_attr('probability')[0], .5)
                self.assertEqual(vector.get_attr('control')[0]['wheel_kv'], .4)
                vector.step(np.zeros((1, 6)))
                np.testing.assert_array_equal(vector.get_attr('torque_limits')[0], [20.]*4+[2.]*2)
                self.assertEqual(vector.get_attr('push')[0].config['force_N'][1], 12.)
                self.assertEqual(vector.get_attr('original')[0]['force_bands'][2]['probability'], .4)
            finally:
                vector.close()


if __name__ == '__main__':
    unittest.main()
