"""The full-policy task must execute actor actions throughout the physical rollout."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))


class FullPolicyGetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from climber_legged_rl.cad_dynamics import build_dynamics
        from climber_legged_rl.getup import prepare_pose_bank
        cls.temp = tempfile.TemporaryDirectory()
        cls.config = Path(cls.temp.name)/'config.yaml'
        config = yaml.safe_load((ROOT/'configs/robot/cad_getup_stand_round8_policy.yaml').read_text())
        config['experiment']['output_directory'] = cls.temp.name
        cls.config.write_text(yaml.safe_dump(config))
        cls.model, _ = build_dynamics(cls.config)
        prepare_pose_bank(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        from climber_legged_rl.mujoco_env import make_mujoco_env
        self.env = make_mujoco_env(self.model, self.config)

    def tearDown(self):
        self.env.close()

    def test_actor_actions_are_executed_from_first_step_and_during_standing(self):
        for mode in ('folded', 'standing'):
            observation, _ = self.env.reset(seed=1, options={'reset_mode': mode})
            self.assertEqual(observation.shape, (36,))
            action = np.array([.4, -.3, .2, -.1, .6, -.5])
            _, _, _, _, info = self.env.step(action)
            np.testing.assert_array_equal(self.env.action, action)
            self.assertTrue(info['policy_controls_all_motors'])
            self.assertEqual(info['control_mode'], 'single_actor_getup_and_standing')

    def test_rise_target_comes_only_from_actor_and_measured_joint_positions(self):
        e = self.env
        e.reset(seed=1, options={'reset_mode': 'folded'})
        before = e.data.qpos.copy()
        e.rise_waypoints[:] = np.nan  # Prepared references must never drive the robot.
        action = np.array([.1, -.1, .2, -.2, 0., 0.])
        target = np.clip(before[e.q_indices[:4]] + .6*action[:4], e.leg_low, e.leg_high)
        for step in (0, 50, 150):
            e.steps = step
            np.testing.assert_allclose(e._leg_targets(action), target)
        np.testing.assert_array_equal(e.data.qpos, before)
        self.assertFalse(e.acquired)
        e.steps = 0
        _, _, _, _, info = e.step(np.zeros(6))
        self.assertFalse(info['getup_acquired'])
        np.testing.assert_array_equal(e.action, np.zeros(6))

    def test_motor_envelope_changes_limits_without_replacing_actions_or_state(self):
        e = self.env
        e.reset(seed=1, options={'reset_mode': 'folded'})
        before = e.data.qpos.copy()
        for seconds, limit, kp in ((0., 40., 80.), (.6, 20., 80.), (1.02, 20., 80.), (9., 20., 80.)):
            e.steps = round(seconds/e.dt)
            e._set_motor_envelope()
            np.testing.assert_array_equal(e.torque_limits[:4], [limit]*4)
            self.assertEqual(e.control['leg_kp'], kp)
            self.assertFalse(e.acquired)
            np.testing.assert_array_equal(e.data.qpos, before)
        e.reset(seed=1, options={'reset_mode': 'standing'})
        np.testing.assert_array_equal(e.torque_limits, [10., 10., 10., 10., 2., 2.])
        action = np.array([.5, -.5, 1., -1., 0., 0.])
        np.testing.assert_allclose(e._leg_targets(action), np.clip(e.nominal_legs+.25*action[:4], e.leg_low, e.leg_high))

    def test_optional_imitation_gain_schedule_only_changes_pd_not_actor_targets(self):
        e = self.env
        e.reset(seed=1, options={'reset_mode': 'folded'})
        e.envelope['rise_gain_seconds'] = 1.02
        before = e.data.qpos.copy()
        action = np.array([.1, -.1, .2, -.2, .3, -.3])
        target = e._leg_targets(action).copy()
        e.steps = round(1.02/e.dt)
        e._set_motor_envelope()
        self.assertEqual(e.control['leg_kp'], 20.)
        np.testing.assert_array_equal(e.torque_limits[:4], [20.]*4)
        np.testing.assert_array_equal(e._leg_targets(action), target)
        np.testing.assert_array_equal(e.data.qpos, before)
        self.assertFalse(e.acquired)

    def test_physical_limits_and_success_criteria_remain_in_force(self):
        e = self.env
        e.reset(seed=1, options={'reset_mode': 'folded'})
        e.max_steps = 1
        _, _, _, truncated, info = e.step(np.ones(6))
        self.assertTrue(truncated)
        self.assertFalse(info['combined_success'])
        self.assertFalse(info['getup_metrics']['guided_reference'])
        self.assertAlmostEqual(float(e.model.body_mass.sum()), 16.463456595727138)
        self.assertEqual(e.model.neq, 6)
        self.assertEqual(e.model.nmesh, 511)
        self.assertEqual(e.getup['leg_tolerance_m'], .01)
        self.assertEqual(e.config['task']['stationary_max_displacement_m'], .05)
        self.assertTrue(np.all(e.peak_torque_seen <= [40., 40., 40., 40., 2., 2.]))
        self.assertTrue(np.all(e.over_rated_seconds <= e.dt + 1e-12))


if __name__ == '__main__':
    unittest.main()
