"""Exact replay must preserve later pulses and restore ordinary reset settings."""
import copy
import unittest
import numpy as np
from test_push_reset_replay import ResetEnv
from climber_legged_rl.push_reset_replay import ExactPushResetReplay
from climber_legged_rl.push_disturbance import PushDisturbance


class ForceEnv(ResetEnv):
    def __init__(self):
        super().__init__()
        config=dict(enabled=True,force_N=[2.,18.],duration_seconds=[.05,.1],
            interval_seconds=[8.,12.],first_push_seconds=3.,recovery_seconds=4.,
            settle_seconds=.5,max_transient_displacement_m=.2,
            max_transient_heading_rad=.35,recovery_tilt_rad=.15,
            episode_probability=.5,direction_mode='balanced_sectors',direction_sectors=8,
            direction_jitter_deg=10.,elevation_deg=[-15.,15.],
            force_bands=[dict(force_N=[2.,18.],probability=1.)])
        self.push=PushDisturbance(config,.01)

    def reset(self,*,seed=None,options=None):
        observation,info=super().reset(seed=seed,options=options)
        self.push.reset(np.random.default_rng(seed))
        return observation,info


class ExactReplayTests(unittest.TestCase):
    def test_replays_all_pulses_and_explicit_reset_restores_training_distribution(self):
        base=ForceEnv();original=copy.deepcopy(base.push.config)
        case=dict(seed=17,force_N=[15.,18.])
        env=ExactPushResetReplay(base,[case],.9,12)
        # Discover one case using the original, unbanded force law.
        cfg=dict(original,force_N=case['force_N'],episode_probability=1.)
        cfg.pop('force_bands')
        reference=PushDisturbance(cfg,.01);reference.reset(np.random.default_rng(17))
        expected=[reference.force(step,2000).copy() for step in range(2000)]
        for _ in range(20):
            _,info=env.reset()
            if info['training_exact_force_replay']:break
        else:self.fail('No replay selected')
        actual=[base.push.force(step,2000).copy() for step in range(2000)]
        np.testing.assert_array_equal(actual,expected)
        self.assertGreaterEqual(len(base.push.events),2)
        self.assertNotIn('force_bands',base.push.config)
        _,info=env.reset(seed=41,options={'reset_mode':'standing'})
        self.assertFalse(info['training_exact_force_replay'])
        self.assertEqual(base.push.config,original)

    def test_rejects_force_above_current_stage(self):
        with self.assertRaisesRegex(ValueError,'stage force'):
            ExactPushResetReplay(ForceEnv(),[dict(seed=1,force_N=[18.,20.])],.2,1)

    def test_reset_failure_restores_training_distribution(self):
        base=ForceEnv();original=copy.deepcopy(base.push.config)
        env=ExactPushResetReplay(base,[dict(seed=17,force_N=[15.,18.])],.9,12)
        base.fail=True
        with self.assertRaises(RuntimeError):env.reset()
        self.assertEqual(base.push.config,original)

    def test_no_push_hard_reset_never_applies_a_pulse(self):
        base=ForceEnv()
        env=ExactPushResetReplay(base,[dict(seed=17,force_N=[2.,15.],push=False)],.9,12)
        _,info=env.reset()
        self.assertTrue(info['training_exact_force_replay'])
        for step in range(2000):np.testing.assert_array_equal(base.push.force(step,2000),np.zeros(3))
        self.assertEqual(base.push.events,[])


if __name__=='__main__':unittest.main()
