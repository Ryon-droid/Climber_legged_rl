"""Verify the reward cannot hide falls, alter physics, or leak reset history."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'source')]
from climber_legged_rl.recovery_clearance_reward import RecoveryClearanceReward
from test_early_recovery_reward import RewardStub

class ClearanceTests(unittest.TestCase):
    def test_approaching_ground_costs_more_than_receding_at_same_gap(self):
        base = RewardStub(); env = RecoveryClearanceReward(base); env.reset()
        with patch('climber_legged_rl.recovery_clearance_reward.minimum_base_clearance', side_effect=[.04,.03,.02,.03]):
            env.step(np.zeros(6))
            *_, closing = env.step(np.zeros(6))
            env.step(np.zeros(6))
            *_, receding = env.step(np.zeros(6))
        self.assertLess(closing['reward_terms']['recovery_clearance'], receding['reward_terms']['recovery_clearance'])

    def test_curvature_cost_is_bounded_and_history_does_not_cross_resets(self):
        base = RewardStub(); env = RecoveryClearanceReward(base); env.reset()
        with patch('climber_legged_rl.recovery_clearance_reward.minimum_base_clearance', return_value=.045):
            for action in (np.zeros(6), np.ones(6), -np.ones(6)):
                *_, info = env.step(action)
            self.assertAlmostEqual(info['reward_terms']['recovery_action_curvature'], -2.5)
            env.reset(); *_, info = env.step(np.zeros(6))
            self.assertEqual(info['reward_terms']['recovery_action_curvature'], 0.)

    def test_real_policy_trajectory_and_termination_are_identical(self):
        from climber_legged_rl.mujoco_env import make_mujoco_env
        from climber_legged_rl.mujoco_model import load_config
        from climber_legged_rl.motor_authority import apply_standing_motor_limit
        from climber_legged_rl.push_balance_curriculum import PushBalanceCurriculum
        from stable_baselines3 import PPO
        import torch
        torch.set_num_threads(1)
        root = Path(__file__).resolve().parents[1]
        cfg = root/'configs/robot/cad_spring208_phase3_nominal.yaml'
        xml = root/load_config(cfg)['experiment']['output_directory']/'robot_experiment.xml'
        policy = PPO.load(root/'outputs/push_focused_recovery24_20261116/selected_policy.zip', device='cpu')
        envs=[]
        try:
            for wrapped in (False, True):
                env = make_mujoco_env(xml, cfg)
                apply_standing_motor_limit(env, 20., .4)
                env.unwrapped.push.config=dict(env.unwrapped.push.config, episode_probability=1., force_N=[24.,24.])
                env.unwrapped.push.config.pop('force_bands',None)
                env=PushBalanceCurriculum(env,'balance',sustained_reward=2.)
                envs.append(RecoveryClearanceReward(env) if wrapped else env)
            a,_=envs[0].reset(seed=1600216,options={'reset_mode':'standing'})
            b,_=envs[1].reset(seed=1600216,options={'reset_mode':'standing'})
            changed=False; total=0.
            while True:
                np.testing.assert_array_equal(a,b)
                action=policy.predict(a,deterministic=True)[0]
                a,ra,da,ta,ia=envs[0].step(action)
                b,rb,db,tb,ib=envs[1].step(action)
                total+=rb;changed |= ra!=rb
                np.testing.assert_array_equal(envs[0].unwrapped.data.qpos,envs[1].unwrapped.data.qpos)
                self.assertEqual((da,ta),(db,tb))
                if da or ta:
                    self.assertEqual(ia['termination_reason'],'base_contact')
                    self.assertEqual(ia['combined_success'],ib['combined_success'])
                    self.assertEqual(ia['push_metrics'],ib['push_metrics'])
                    self.assertAlmostEqual(ib['episode_metrics']['reward'],total)
                    break
            self.assertTrue(changed)
        finally:
            for env in envs:env.close()

if __name__=='__main__':unittest.main()
