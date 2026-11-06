"""Physical reset validity, disjoint geometry, and protected optimizer settings."""
from pathlib import Path
import sys
import unittest
import mujoco
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import closure_error, load_config
from climber_legged_rl.mujoco_training import load_warm_start
from climber_legged_rl.policy_getup import ClimberPolicyGetupEnv
from climber_legged_rl.spring_geometry import eye_center_distance

CONFIG = ROOT/'configs/robot/cad_spring208_phase1_train.yaml'
PARENT = ROOT/'outputs/mujoco_runs/spring208_rising_distill_20261106_031028/final_policy.zip'


class Spring208Phase1Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = load_config(CONFIG)
        self.xml = ROOT/self.config['experiment']['output_directory']/'robot_experiment.xml'
        self.env = make_mujoco_env(self.xml, CONFIG)
        self.addCleanup(self.env.close)

    def test_seeded_resets_are_closed_in_range_and_repeatable(self):
        samples = []
        for seed in range(20):
            obs, info = self.env.reset(seed=seed, options={'reset_mode': 'folded'})
            qpos, qvel = self.env.data.qpos.copy(), self.env.data.qvel.copy()
            self.assertIn('folded_reset_noise', info)
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertLess(np.linalg.norm(closure_error(self.env.model, self.env.data), axis=1).max(), 1e-4)
            np.testing.assert_array_equal(qvel[6:], 0.)
            self.assertLessEqual(abs(qvel[:3]).max(), .002)
            self.assertLessEqual(abs(qvel[3:6]).max(), .01)
            for side in ('left', 'right'):
                self.assertTrue(.155 <= eye_center_distance(self.env.model, self.env.data, side) <= .208)
            bottoms = [(v @ self.env.data.geom_xmat[g].reshape(3, 3).T+
                        self.env.data.geom_xpos[g])[:, 2].min() for g, v in self.env.unwrapped.meshes]
            self.assertAlmostEqual(min(bottoms), .0005, places=8)
            obs2, _ = self.env.reset(seed=seed, options={'reset_mode': 'folded'})
            np.testing.assert_array_equal(qpos, self.env.data.qpos)
            np.testing.assert_array_equal(qvel, self.env.data.qvel)
            np.testing.assert_array_equal(obs, obs2)
            samples.append(qpos)
        self.assertGreater(np.max(np.ptp(samples, axis=0)), 1e-3)

    def test_standing_reset_unchanged_by_opt_in_noise(self):
        old = ClimberPolicyGetupEnv(self.xml, CONFIG)
        self.addCleanup(old.close)
        for seed in (0, 42, 261409):
            _, info = self.env.reset(seed=seed, options={'reset_mode': 'standing'})
            old.reset(seed=seed, options={'reset_mode': 'standing'})
            np.testing.assert_array_equal(self.env.data.qpos, old.data.qpos)
            np.testing.assert_array_equal(self.env.data.qvel, old.data.qvel)
            self.assertNotIn('folded_reset_noise', info)

    def test_train_and_test_geometries_are_disjoint_same_physics(self):
        test_path = ROOT/'configs/robot/cad_spring208_phase1_test.yaml'
        test = load_config(test_path)
        self.assertFalse(set(self.config['getup']['rearward_offsets_m']) & set(test['getup']['rearward_offsets_m']))
        other_xml = ROOT/test['experiment']['output_directory']/'robot_experiment.xml'
        self.assertEqual(other_xml.read_bytes(), self.xml.read_bytes())
        validate_artifacts(CONFIG)
        validate_artifacts(test_path)

    def test_warm_start_freezes_standing_estimator_and_std_but_trains_rising(self):
        vector = DummyVecEnv([lambda: make_mujoco_env(self.xml, CONFIG)])
        self.addCleanup(vector.close)
        model, _, _ = load_warm_start(PARENT, vector, self.config['training'], 'cpu', 208341)
        model.configure_estimator(self.config['training']['velocity_estimator'])
        actor = model.policy.mlp_extractor.policy_net
        for module in (actor.standing_net, actor.standing_head, actor.velocity_estimator):
            self.assertTrue(all(not p.requires_grad for p in module.parameters()))
        self.assertTrue(all(p.requires_grad for p in actor.rising_net.parameters()))
        self.assertFalse(model.policy.log_std.requires_grad)


if __name__ == '__main__':
    unittest.main()
