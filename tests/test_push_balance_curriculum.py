"""Relax position without accepting running, spinning, falling, or brief settling."""
import copy
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from climber_legged_rl.push_balance_curriculum import STAGES, RecoveryAccounting, PushBalanceCurriculum


def stable_info(**extra):
    return dict(dict(planar_displacement_m=2., heading_error_rad=math.radians(90),
        planar_speed_m_s=.01, yaw_rate_rad_s=.01, tilt_rad=.01, tilt_rate_rad_s=.01,
        wheel_contacts=2, termination_reason=None), **extra)


class NativeStub(gym.Env):
    """A stable native-strict failure at a distant location."""
    def __init__(self, reason=None):
        self.action_space = gym.spaces.Box(-1., 1., (6,))
        self.observation_space = gym.spaces.Box(-1., 1., (1,))
        self.dt = .01
        self.push = SimpleNamespace(config=dict(settle_seconds=.5), events=[])
        self.reason = reason

    def reset(self, **kwargs):
        self.steps = 0
        self._total_reward = 0.
        return np.zeros(1), {}

    def _kinematics(self):
        return np.zeros(3), np.zeros(3), np.array([0., 0., -1.])

    def step(self, action):
        self.steps += 1
        done = self.steps == 60 and self.reason is not None
        truncated = self.steps == 60 and not done
        terms = dict(upright=6., position_hold=10., heading_hold=1.5,
                     recovery_position=-2., recovery_heading=-.5)
        reward = sum(terms.values()) - (80. if done else 0.)
        self._total_reward += reward
        info = stable_info(termination_reason=self.reason if done else None)
        info.update(reward_terms=terms, getup_acquired=True)
        if done or truncated:
            info.update(combined_success=False, push_metrics=dict(push_count=0),
                        getup_metrics=dict(combined_success=False),
                        episode_metrics=dict(reward=self._total_reward))
        return np.zeros(1), reward, done, truncated, info


class BalanceCurriculumTests(unittest.TestCase):
    def test_sustained_reward_grows_with_stable_time_without_changing_success_rules(self):
        env = PushBalanceCurriculum(NativeStub(), sustained_reward=2.)
        env.reset()
        _, first, _, _, _ = env.step(np.zeros(6))
        self.assertAlmostEqual(first, 6.01)
        for _ in range(59):
            _, reward, _, _, info = env.step(np.zeros(6))
        self.assertAlmostEqual(reward, 6.6)
        self.assertTrue(info['combined_success'])
        self.assertFalse(info['strict_success'])
        self.assertAlmostEqual(info['episode_metrics']['reward'], 360. + .01*sum(range(1,61)))

    def test_sustained_reward_resets_on_fall_and_rejects_invalid_weights(self):
        env = PushBalanceCurriculum(NativeStub(reason='base_contact'), sustained_reward=2.)
        env.reset()
        for _ in range(60):
            _, reward, done, _, info = env.step(np.zeros(6))
        self.assertTrue(done)
        self.assertEqual(info['reward_terms']['sustained_balance'], 0.)
        self.assertEqual(reward, -74.)
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                PushBalanceCurriculum(NativeStub(), sustained_reward=value)

    def test_reward_and_monitor_success_change_together_without_changing_native_success(self):
        env = PushBalanceCurriculum(NativeStub())
        env.reset()
        for _ in range(60):
            _, reward, done, truncated, info = env.step(np.zeros(6))
        self.assertEqual(reward, 6.)
        self.assertEqual(info['episode_metrics']['reward'], 360.)
        self.assertTrue(info['combined_success'])
        self.assertFalse(info['strict_success'])
        self.assertFalse(done)
        self.assertTrue(truncated)
        self.assertEqual(info['strict_push_metrics'], dict(push_count=0))

    def test_fall_penalty_and_native_strict_stage_are_preserved(self):
        env = PushBalanceCurriculum(NativeStub(reason='base_contact'))
        env.reset()
        for _ in range(60):
            _, reward, done, truncated, info = env.step(np.zeros(6))
        self.assertEqual(reward, 6. - 80.)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertFalse(info['combined_success'])
        env = PushBalanceCurriculum(NativeStub(), 'strict')
        env.reset()
        for _ in range(60):
            _, reward, _, _, info = env.step(np.zeros(6))
        self.assertEqual(reward, 15.)
        self.assertFalse(info['combined_success'])
        self.assertFalse(info['strict_success'])

    def test_balance_ignores_position_and_heading_but_later_stages_require_them(self):
        info = stable_info()
        self.assertTrue(STAGES['balance'].stable(info))
        self.assertFalse(STAGES['return'].stable(info))
        info.update(planar_displacement_m=.10, heading_error_rad=math.radians(10))
        self.assertTrue(STAGES['return'].stable(info))
        self.assertFalse(STAGES['strict'].stable(info))
        info.update(planar_displacement_m=.04, heading_error_rad=math.radians(4))
        self.assertTrue(STAGES['strict'].stable(info))

    def test_balance_rejects_moving_spinning_tilt_loss_of_support_and_fall(self):
        for patch in (dict(planar_speed_m_s=.2), dict(yaw_rate_rad_s=.3),
                      dict(tilt_rad=math.radians(15)), dict(tilt_rate_rad_s=.5),
                      dict(wheel_contacts=1), dict(termination_reason='base_contact'),
                      dict(planar_speed_m_s=float('nan'))):
            self.assertFalse(STAGES['balance'].stable(stable_info(**patch)))

    def test_post_pulse_requires_continuous_settling_before_deadline(self):
        accounting = RecoveryAccounting(STAGES['balance'], .01, .5)
        event = dict(start_step=1, end_step=10, deadline_step=120, recovered=True,
                     recovery_seconds=.01, settled_steps=50, impulse_Ns=1.)
        untouched = copy.deepcopy(event)
        for step in range(1, 59):
            accounting.observe(step, [event], stable_info())
        self.assertFalse(accounting.events[0]['recovered'])
        accounting.observe(59, [event], stable_info(wheel_contacts=0))
        for step in range(60, 109):
            accounting.observe(step, [event], stable_info())
        self.assertFalse(accounting.events[0]['recovered'])
        accounting.observe(109, [event], stable_info())
        self.assertTrue(accounting.success(True, 2., math.pi))
        self.assertFalse(accounting.success(False, 2., math.pi))
        self.assertEqual(event, untouched)

    def test_late_recovery_and_final_motion_do_not_pass(self):
        accounting = RecoveryAccounting(STAGES['balance'], .01, .5)
        event = dict(start_step=1, end_step=10, deadline_step=30, impulse_Ns=1.)
        for step in range(1, 101):
            accounting.observe(step, [event], stable_info())
        self.assertFalse(accounting.success(True, 2., 1.))
        accounting.events[0]['recovered'] = True
        accounting.observe(101, [event], stable_info(planar_speed_m_s=.5))
        self.assertFalse(accounting.success(True, 2., 1.))

    def test_intermediate_transient_envelope_tightens(self):
        self.assertTrue(STAGES['balance'].within_transient(2., math.pi))
        self.assertFalse(STAGES['return'].within_transient(.31, 0.))
        self.assertTrue(STAGES['return'].within_transient(.25, .5))
        self.assertFalse(STAGES['strict'].within_transient(.25, .5))


if __name__ == '__main__':
    unittest.main()
