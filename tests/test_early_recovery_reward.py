"""Reward changes must preserve termination, success, actions and real physics."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'source')]
from climber_legged_rl.early_recovery_reward import EarlyRecoveryReward, BALANCED_RECOVERY_SETTINGS
from climber_legged_rl.push_balance_curriculum import PushBalanceCurriculum
from test_push_balance_curriculum import NativeStub


class RewardStub(NativeStub):
    def __init__(self):
        super().__init__()
        self.tilt = 0.
        self.acquired = True
        self.settled = 1.

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info.update(tilt_rad=self.tilt, getup_acquired=self.acquired, balance_settled_seconds=self.settled)
        info['reward_terms'].update(virtual_leg_length=2., linear_tracking=2.)
        self._total_reward += 4.
        if done or truncated:
            info['episode_metrics']['reward'] = self._total_reward
        return obs, reward+4., done, truncated, info


class EarlyRewardTests(unittest.TestCase):
    def test_milder_settings_reduce_risk_without_mutating_other_instances(self):
        oldbase, newbase = RewardStub(), RewardStub()
        old, new = EarlyRecoveryReward(oldbase), EarlyRecoveryReward(newbase, BALANCED_RECOVERY_SETTINGS)
        old.reset(); new.reset()
        oldbase.tilt = newbase.tilt = math.radians(12.)
        *_, oldinfo = old.step(np.zeros(6))
        *_, newinfo = new.step(np.zeros(6))
        self.assertLess(oldinfo['reward_terms']['early_recovery_risk'], newinfo['reward_terms']['early_recovery_risk'])
        self.assertLess(newinfo['reward_terms']['early_recovery_risk'], 0.)
        new.settings['risk_weight'] = 0.
        self.assertEqual(old.settings['risk_weight'], 2.)
        self.assertEqual(BALANCED_RECOVERY_SETTINGS['risk_weight'], .25)

    def test_upright_quiet_reward_is_unchanged(self):
        base = RewardStub(); env = EarlyRecoveryReward(base); env.reset()
        _, reward, _, _, info = env.step(np.zeros(6))
        self.assertEqual(reward, sum(info['reward_terms'].values()))
        self.assertFalse(info['early_recovery_active'])
        self.assertEqual(info['reward_terms']['early_recovery_risk'], 0.)

    def test_outward_tilt_is_penalized_before_same_angle_recovery(self):
        base = RewardStub(); env = EarlyRecoveryReward(base); env.reset()
        base.tilt = math.radians(2.); env.step(np.zeros(6))
        base.tilt = math.radians(3.); *_, outward = env.step(np.zeros(6))
        base.tilt = math.radians(4.); env.step(np.zeros(6))
        base.tilt = math.radians(3.); *_, inward = env.step(np.zeros(6))
        self.assertLess(outward['reward_terms']['early_recovery_risk'], inward['reward_terms']['early_recovery_risk'])
        self.assertEqual(inward['reward_terms']['early_recovery_risk'], 0.)

    def test_transient_relaxation_restores_after_continuous_settling(self):
        base = RewardStub(); env = EarlyRecoveryReward(base); env.reset()
        base.push.events = [dict(end_step=3)]; base.settled = 0.
        *_, info = env.step(np.zeros(6))
        self.assertEqual(info['reward_terms']['virtual_leg_length'], .5)
        self.assertEqual(info['reward_terms']['linear_tracking'], .5)
        base.steps = 10; base.settled = .5
        *_, info = env.step(np.zeros(6))
        self.assertFalse(info['early_recovery_active'])
        self.assertEqual(info['reward_terms']['virtual_leg_length'], 2.)

    def test_unacquired_getup_reward_is_unchanged_and_reset_forgets_tilt(self):
        base = RewardStub(); env = EarlyRecoveryReward(base); env.reset()
        base.acquired = False; base.tilt = math.radians(30.)
        *_, info = env.step(np.zeros(6))
        self.assertEqual(info['reward_terms']['early_recovery_risk'], 0.)
        env.reset(); self.assertIsNone(env.previous_tilt)

    def test_fall_and_success_accounting_and_episode_total_remain_consistent(self):
        base = RewardStub(); base.reason = 'base_contact'
        env = EarlyRecoveryReward(base); env.reset(); total = 0.
        for _ in range(60):
            _, reward, done, truncated, info = env.step(np.zeros(6)); total += reward
        self.assertTrue(done); self.assertFalse(truncated)
        self.assertFalse(info['combined_success'])
        self.assertEqual(reward, sum(info['reward_terms'].values())-80.)
        self.assertAlmostEqual(info['episode_metrics']['reward'], total)

    def test_matched_real_rollout_preserves_every_state_and_success(self):
        from climber_legged_rl.mujoco_env import make_mujoco_env
        from climber_legged_rl.mujoco_model import load_config
        from climber_legged_rl.motor_authority import apply_standing_motor_limit
        from stable_baselines3 import PPO
        import torch
        torch.set_num_threads(1)
        root = Path(__file__).resolve().parents[1]
        cfg = root/'configs/robot/cad_spring208_phase3_nominal.yaml'
        model_path = root/load_config(cfg)['experiment']['output_directory']/'robot_experiment.xml'
        checkpoint = root/'outputs/push_recovery_curriculum18_20261115/selected_policy.zip'
        policy = PPO.load(checkpoint, device='cpu')
        envs = []
        try:
            for wrapped in (False, True):
                env = make_mujoco_env(model_path, cfg)
                apply_standing_motor_limit(env, 20.)
                env.unwrapped.push.config = dict(env.unwrapped.push.config, episode_probability=1., force_N=[18.,18.])
                env.unwrapped.push.config.pop('force_bands', None)
                env = PushBalanceCurriculum(env, 'balance', sustained_reward=2.)
                envs.append(EarlyRecoveryReward(env, BALANCED_RECOVERY_SETTINGS) if wrapped else env)
            a, _ = envs[0].reset(seed=1331008, options={'reset_mode':'standing'})
            b, _ = envs[1].reset(seed=1331008, options={'reset_mode':'standing'})
            changed = False
            while True:
                np.testing.assert_array_equal(a, b)
                action, _ = policy.predict(a, deterministic=True)
                a, ra, da, ta, ia = envs[0].step(action)
                b, rb, db, tb, ib = envs[1].step(action)
                np.testing.assert_array_equal(envs[0].unwrapped.data.qpos, envs[1].unwrapped.data.qpos)
                self.assertEqual((da,ta), (db,tb)); changed |= ra != rb
                if da or ta:
                    self.assertEqual(ia['combined_success'], ib['combined_success'])
                    self.assertEqual(ia['push_metrics'], ib['push_metrics'])
                    break
            self.assertTrue(changed)
        finally:
            for env in envs: env.close()


if __name__ == '__main__':
    unittest.main()
