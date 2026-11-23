"""Exercise the actual phase-3 motor limiter without PPO training."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'source'), str(ROOT / 'scripts')]
from climber_legged_rl.motor_authority import apply_standing_motor_limit
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config


class StandingMotorAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ROOT / 'configs/robot/cad_spring208_phase3_nominal.yaml'
        config = load_config(self.cfg)
        xml = ROOT / config['experiment']['output_directory'] / 'robot_experiment.xml'
        if not xml.exists():
            self.skipTest('Phase-3 model assets are required')
        self.env = make_mujoco_env(xml, self.cfg)
        self.addCleanup(self.env.close)

    def test_override_changes_neither_state_nor_canonical_config_or_gains(self):
        e = self.env.unwrapped
        e.reset(seed=12, options={'reset_mode': 'standing'})
        state, config, control = e.data.qpos.copy(), copy.deepcopy(e.config), dict(e.control)
        metadata = apply_standing_motor_limit(e, 20.)
        np.testing.assert_array_equal(e.data.qpos, state)
        self.assertEqual(e.config, config)
        self.assertEqual(e.control, control)
        self.assertEqual(e.getup['policy_control']['standing_torque_limit_Nm'], 10.)
        self.assertEqual(metadata['standing_leg_torque_limit_Nm'], 20.)
        self.assertTrue(metadata['runtime_override'])

    def test_standing_limit_persists_after_resets_and_steps(self):
        e = self.env.unwrapped
        apply_standing_motor_limit(e, 20.)
        for seed in (12, 13):
            e.reset(seed=seed, options={'reset_mode': 'standing'})
            for _ in range(4):
                np.testing.assert_array_equal(e.torque_limits, [20.] * 4 + [2.] * 2)
                e.step(np.zeros(6))
                np.testing.assert_array_equal(e.torque_limits, [20.] * 4 + [2.] * 2)

    def test_rising_still_has_peak_window_then_rated_limit(self):
        e = self.env.unwrapped
        apply_standing_motor_limit(e, 20.)
        e.reset(seed=12, options={'reset_mode': 'folded'})
        for seconds, limit in ((0., 40.), (.59, 40.), (.6, 20.), (1.1, 20.)):
            e.steps = round(seconds / e.dt)
            e._set_motor_envelope()
            np.testing.assert_array_equal(e.torque_limits, [limit] * 4 + [2.] * 2)
        self.assertEqual(e.control['leg_kp'], e.envelope['standing_leg_kp'])
        self.assertFalse(e.acquired)
        e.acquired = True
        e._set_motor_envelope()
        np.testing.assert_array_equal(e.torque_limits, [20.] * 4 + [2.] * 2)

    def test_invalid_limits_cannot_change_environment(self):
        e = self.env.unwrapped
        for limit in (-1., 0., 20.01, 40., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                apply_standing_motor_limit(e, limit)
            self.assertEqual(e.envelope['standing_torque_limit_Nm'], 10.)

    def test_historical_default_is_still_ten(self):
        e = self.env.unwrapped
        metadata = apply_standing_motor_limit(e)
        e.reset(seed=12, options={'reset_mode': 'standing'})
        e.step(np.zeros(6))
        np.testing.assert_array_equal(e.torque_limits, [10.] * 4 + [2.] * 2)
        self.assertFalse(metadata['runtime_override'])

    def test_wheel_response_override_only_applies_after_acquisition(self):
        e = self.env.unwrapped
        config = copy.deepcopy(e.config)
        metadata = apply_standing_motor_limit(e, 20., standing_wheel_kv=.4)
        self.assertFalse(metadata['pd_gains_unchanged'])
        for mode, expected in [('standing', .4), ('folded', .2), ('standing', .4)]:
            e.reset(seed=12, options={'reset_mode': mode})
            e.step(np.zeros(6))
            self.assertEqual(e.control['wheel_kv'], expected)
        self.assertEqual(e.config, config)
        np.testing.assert_array_equal(e.torque_limits, [20.] * 4 + [2.] * 2)
        e.reset(seed=12, options={'reset_mode': 'folded'})
        e.acquired = True
        e._set_motor_envelope()
        self.assertEqual(e.control['wheel_kv'], .4)

    def test_bad_wheel_gains_are_rejected_atomically(self):
        e = self.env.unwrapped
        envelope, control = dict(e.envelope), dict(e.control)
        for gain in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                apply_standing_motor_limit(e, 20., standing_wheel_kv=gain)
            self.assertEqual(e.envelope, envelope)
            self.assertEqual(e.control, control)

    def test_actual_physics_clips_at_twenty_instead_of_ten(self):
        e = self.env.unwrapped
        # Discarded low-level saturation probe, not a policy rollout or training.
        for limit in (10., 20.):
            apply_standing_motor_limit(e, limit)
            e.reset(seed=12, options={'reset_mode': 'standing'})
            observed = []
            observe = e._observe_motor_torque
            def measured(torque):
                observed.append(torque.copy())
                observe(torque)
            e._observe_motor_torque = measured
            try:
                e._advance_physics(e.data.qpos[e.q_indices[:4]] + 2., np.zeros(2))
            finally:
                e._observe_motor_torque = observe
            actual = np.abs(np.array(observed))
            self.assertEqual(len(observed), e.control['decimation'])
            self.assertLessEqual(actual[:, :4].max(), limit + 1e-9)
            self.assertGreaterEqual(actual[:, :4].max(), limit - 1e-9)
            self.assertLessEqual(actual[:, 4:].max(), 2. + 1e-9)

    def test_training_worker_receives_captured_limit_and_replay_retains_it(self):
        import train_spring208_push_balance as trainer
        from stable_baselines3.common.vec_env import SubprocVecEnv
        previous = trainer.PROFILE
        try:
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                (folder / 'hard_training_resets.json').write_text(json.dumps(dict(cases=[
                    dict(seed=12, force_N=[18., 18.], push=True)])))
                trainer.PROFILE = 'rated_balance18'
                create = trainer.factory(folder, 12, 'balance')
                # Spawned workers import globals anew; the closure must carry its limit.
                trainer.PROFILE = 'balance_first18'
                vector = SubprocVecEnv([create], start_method='spawn')
                try:
                    vector.reset()
                    for _ in range(4):
                        vector.step(np.zeros((1, 6)))
                        np.testing.assert_array_equal(vector.get_attr('torque_limits')[0],
                                                      [20.] * 4 + [2.] * 2)
                finally:
                    vector.close()
        finally:
            trainer.PROFILE = previous

    def test_evaluator_records_and_obeys_requested_limit(self):
        from train_spring208_push import evaluate_case
        checkpoint = ROOT / 'outputs/push_wheel_recovery18_20261113_2020/selected_policy.zip'
        if not checkpoint.exists():
            self.skipTest('Accepted checkpoint required for evaluator integration')
        spec = dict(config=str(self.cfg), group='standing', mode='standing', push=False,
                    seed=12, balance_curriculum_stage='balance', standing_leg_torque_limit_Nm=20.)
        row = evaluate_case((str(checkpoint), spec, True))
        self.assertEqual(row['motor_envelope']['standing_leg_torque_limit_Nm'], 20.)
        self.assertEqual(row['warnings'], 0)
        self.assertGreater(row['spring_sample_count'], 2)
        self.assertIn('strict_success', row)
        self.assertTrue(np.all(np.array(row['peak_torque_Nm']) <= [20.] * 4 + [2.] * 2))

    def test_evaluator_applies_and_records_wheel_response(self):
        from train_spring208_push import evaluate_case
        checkpoint = ROOT / 'outputs/push_recovery_curriculum18_20261115/selected_policy.zip'
        spec = dict(config=str(self.cfg), group='standing', mode='standing', push=False,
                    seed=12, balance_curriculum_stage='balance',
                    standing_leg_torque_limit_Nm=20., standing_wheel_kv=.4)
        row = evaluate_case((str(checkpoint), spec, True))
        self.assertEqual(row['motor_envelope']['standing_wheel_kv'], .4)
        self.assertEqual(row['motor_envelope']['rising_wheel_kv'], .2)
        self.assertEqual(row['warnings'], 0)
        self.assertTrue(row['success'])
        self.assertTrue(np.all(np.array(row['peak_torque_Nm']) <= [20.] * 4 + [2.] * 2))


class MotorProfileTests(unittest.TestCase):
    def test_only_new_profile_requests_twenty(self):
        import train_spring208_push_balance as trainer
        previous = trainer.PROFILE
        try:
            for profile in ('balance_first18', 'sustained_balance18', 'rated_balance18', 'reactive_balance18'):
                trainer.PROFILE = profile
                expected = {'standing_leg_torque_limit_Nm': 20.} if profile in ('rated_balance18', 'reactive_balance18') else {}
                self.assertEqual(trainer.motor_options(), expected)
                self.assertEqual(trainer.sustained_profile(), profile != 'balance_first18')
        finally:
            trainer.PROFILE = previous


if __name__ == '__main__':
    unittest.main()
