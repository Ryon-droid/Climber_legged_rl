"""Catch-first rewards must preserve the plant and restore quiet-stance costs."""
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts')]
import train_spring208_push_balance as trainer
from climber_legged_rl.recovery_clearance_reward import RecoveryClearanceReward
from test_early_recovery_reward import RewardStub


class VelocityStub(RewardStub):
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info['reward_terms']['recovery_velocity'] = -4.
        self._total_reward -= 4.
        if done or truncated:
            info['episode_metrics']['reward'] = self._total_reward
        return obs, reward-4., done, truncated, info


class CatchRecoveryTests(unittest.TestCase):
    def setUp(self):
        with patch.object(trainer.ab, 'setup'):
            trainer.setup('balance', 'catch_recovery24')

    def tearDown(self):
        with patch.object(trainer.ab, 'setup'):
            trainer.setup('balance', 'focused_recovery24')

    def test_stop_cost_relaxes_during_pulse_then_returns_when_settled(self):
        base = VelocityStub()
        env = RecoveryClearanceReward(base, trainer.clearance_reward_settings())
        env.reset()
        base.push.events = [dict(end_step=3)]
        base.settled = 1.
        with patch('climber_legged_rl.recovery_clearance_reward.minimum_base_clearance', return_value=.045):
            _, reward, _, _, info = env.step(np.zeros(6))
            self.assertTrue(info['recovery_clearance_active'])
            self.assertEqual(info['reward_terms']['recovery_velocity'], -1.)
            self.assertEqual(info['reward_terms']['virtual_leg_length'], .5)
            self.assertAlmostEqual(reward, sum(info['reward_terms'].values()))
            base.steps = 10
            *_, info = env.step(np.ones(6))
            self.assertFalse(info['recovery_clearance_active'])
            self.assertEqual(info['reward_terms']['recovery_velocity'], -4.)
            self.assertEqual(info['reward_terms']['virtual_leg_length'], 2.)

    def test_new_holdouts_exclude_all_discovery_and_old_test_ranges(self):
        assessor = trainer.Assessor.__new__(trainer.Assessor)
        assessor.pool = MagicMock()
        assessor.pool.map.return_value = []
        splits = []
        for split in ('selection', 'test'):
            with patch.object(trainer, 'sha', return_value='unused'):
                assessor.evaluate(Path('unused.zip'), split)
            specs = [job[1] for job in assessor.pool.map.call_args.args[1]]
            seeds = {s['seed'] for s in specs}
            self.assertTrue(all(s['measure_recovery_quality'] for s in specs))
            self.assertFalse(seeds & set(range(1640000, 1651256)))
            splits.append(seeds)
        self.assertFalse(splits[0] & splits[1])

    def test_real_failed_trajectory_identical_with_catch_reward(self):
        from stable_baselines3 import PPO
        import torch
        from climber_legged_rl.mujoco_env import make_mujoco_env
        from climber_legged_rl.mujoco_model import load_config
        from climber_legged_rl.motor_authority import apply_standing_motor_limit
        from climber_legged_rl.push_balance_curriculum import PushBalanceCurriculum
        torch.set_num_threads(1)
        cfg = trainer.CFG
        xml = trainer.ROOT/load_config(cfg)['experiment']['output_directory']/'robot_experiment.xml'
        policy = PPO.load(trainer.PARENT, device='cpu')
        envs = []
        try:
            for wrapped in (False, True):
                env = make_mujoco_env(xml, cfg)
                apply_standing_motor_limit(env, **trainer.motor_options())
                env.unwrapped.push.config = dict(env.unwrapped.push.config,
                    episode_probability=1., force_N=[24., 24.])
                env.unwrapped.push.config.pop('force_bands', None)
                env = PushBalanceCurriculum(env, 'balance', sustained_reward=2.)
                envs.append(RecoveryClearanceReward(env, trainer.clearance_reward_settings()) if wrapped else env)
            a, _ = envs[0].reset(seed=1581065, options={'reset_mode':'standing'})
            b, _ = envs[1].reset(seed=1581065, options={'reset_mode':'standing'})
            total = 0.
            changed = False
            while True:
                np.testing.assert_array_equal(a, b)
                action = policy.predict(a, deterministic=True)[0]
                a, ra, da, ta, ia = envs[0].step(action)
                b, rb, db, tb, ib = envs[1].step(action)
                total += rb
                changed |= ra != rb
                np.testing.assert_array_equal(envs[0].unwrapped.data.qpos, envs[1].unwrapped.data.qpos)
                self.assertEqual((da, ta), (db, tb))
                if da or ta:
                    self.assertEqual(ib['termination_reason'], 'base_contact')
                    self.assertEqual(ia['combined_success'], ib['combined_success'])
                    self.assertEqual(ia['push_metrics'], ib['push_metrics'])
                    self.assertAlmostEqual(ib['episode_metrics']['reward'], total)
                    break
            self.assertTrue(changed)
        finally:
            for env in envs:
                env.close()


if __name__ == '__main__':
    unittest.main()
