"""Stationary balance must track displacement and heading, not just survival."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ('mujoco', 'numpy', 'yaml', 'gymnasium', 'stable_baselines3'))


@unittest.skipUnless(AVAILABLE, 'Training dependencies absent')
class StationaryHoldTests(unittest.TestCase):
    def setUp(self):
        import yaml
        from climber_legged_rl.mujoco_env import ClimberMujocoEnv
        self.temp = tempfile.TemporaryDirectory()
        self.config = yaml.safe_load((ROOT / 'configs/robot/cad_dynamics_round4.yaml').read_text())
        self.model = ROOT / 'outputs/cad_dynamics_round3/robot_experiment.xml'
        self.path = Path(self.temp.name) / 'config.yaml'
        self.path.write_text(yaml.safe_dump(self.config))
        self.env = ClimberMujocoEnv(self.model, self.path)

    def tearDown(self):
        self.env.close()
        self.temp.cleanup()

    def test_legacy_inputs_are_preserved_and_new_feedback_resets_to_zero(self):
        import numpy as np
        import yaml
        from climber_legged_rl.mujoco_env import ClimberMujocoEnv
        config = copy.deepcopy(self.config)
        config['task']['stationary_hold'] = False
        legacy_path = Path(self.temp.name) / 'legacy.yaml'
        legacy_path.write_text(yaml.safe_dump(config))
        legacy = ClimberMujocoEnv(self.model, legacy_path)
        try:
            old, _ = legacy.reset(seed=17)
            new, _ = self.env.reset(seed=17)
            self.assertEqual(old.shape, (24,))
            self.assertEqual(new.shape, (30,))
            np.testing.assert_allclose(new[:24], old, atol=1e-7)
            np.testing.assert_allclose(new[24:28], 0., atol=1e-7)
            self.env.data.qpos[0] += .1
            import mujoco
            mujoco.mj_forward(self.env.model, self.env.data)
            self.assertGreater(np.linalg.norm(self.env._observation()[24:26]), .4)
            reset, _ = self.env.reset(seed=17)
            np.testing.assert_allclose(reset[24:28], 0., atol=1e-7)
        finally:
            legacy.close()

    def test_yaw_is_visible_and_suppresses_heading_reward(self):
        import mujoco
        import numpy as np
        self.env.reset(seed=17)
        self.env.data.qpos[2] += 1.
        self.env.data.qvel[:] = 0.
        mujoco.mj_forward(self.env.model, self.env.data)
        quat = np.empty(4)
        delta = np.array([np.cos(np.pi/4), 0., 0., np.sin(np.pi/4)])
        mujoco.mju_mulQuat(quat, delta, self.env.data.qpos[3:7].copy())
        self.env.data.qpos[3:7] = quat
        mujoco.mj_forward(self.env.model, self.env.data)
        _, heading = self.env._hold_errors()
        self.assertAlmostEqual(heading, np.pi/2, places=6)
        obs = self.env._observation()
        np.testing.assert_allclose(obs[26:28], [1., -1.], atol=1e-6)
        _, _, _, _, info = self.env.step(np.zeros(6))
        self.assertLess(info['reward_terms']['heading_hold'], 1e-10)
        self.assertGreater(info['reward_terms']['position_hold'], 1.9)

    def test_displacement_suppresses_position_reward(self):
        import mujoco
        import numpy as np
        self.env.reset(seed=17)
        self.env.data.qpos[0] += .2
        self.env.data.qpos[2] += 1.
        self.env.data.qvel[:] = 0.
        mujoco.mj_forward(self.env.model, self.env.data)
        _, _, _, _, info = self.env.step(np.zeros(6))
        self.assertGreater(info['planar_displacement_m'], .19)
        self.assertLess(info['reward_terms']['position_hold'], 1e-5)
        self.assertGreater(info['reward_terms']['heading_hold'], 1.4)

    def test_success_rejects_turning_drifting_fast_motion_and_falling(self):
        success = self.env.stationary_success
        self.assertTrue(success(True, .01, .01, .01, .01))
        for arguments in [(False, 0., 0., 0., 0.), (True, .1, 0., 0., 0.),
                          (True, 0., .2, 0., 0.), (True, 0., 0., .1, 0.),
                          (True, 0., 0., 0., .1)]:
            self.assertFalse(success(*arguments))

    def test_warm_start_preserves_old_actions_and_learns_from_new_inputs(self):
        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from climber_legged_rl.mujoco_training import load_warm_start
        from climber_legged_rl.mujoco_env import ClimberMujocoEnv
        import yaml
        old_config = copy.deepcopy(self.config)
        old_config['task']['stationary_hold'] = False
        path = Path(self.temp.name) / 'old.yaml'
        path.write_text(yaml.safe_dump(old_config))
        old_env = ClimberMujocoEnv(self.model, path)
        vector = DummyVecEnv([lambda: ClimberMujocoEnv(self.model, self.path)])
        try:
            old = PPO('MlpPolicy', old_env, n_steps=8, batch_size=8, n_epochs=1,
                      policy_kwargs={'net_arch': {'pi': [8], 'vf': [8]}}, device='cpu', seed=7)
            checkpoint = Path(self.temp.name) / 'policy.zip'
            old.num_timesteps = 1234
            old.save(checkpoint)
            settings = dict(self.config['training'], n_steps=8, batch_size=8, n_epochs=1)
            new, parent, expanded = load_warm_start(checkpoint, vector, settings, 'cpu', 45)
            self.assertTrue(expanded)
            self.assertEqual(parent, 1234)
            observations = np.random.default_rng(7).normal(0., .1, (5, 24)).astype(np.float32)
            extra = np.ones((5, 6), dtype=np.float32)
            before, _ = old.predict(observations, deterministic=True)
            after, _ = new.predict(np.concatenate([observations, extra], axis=1), deterministic=True)
            np.testing.assert_allclose(before, after, atol=1e-6)
            new.learn(total_timesteps=8)
            weights = new.policy.state_dict()['mlp_extractor.policy_net.0.weight']
            self.assertGreater(float(weights[:, 24:].abs().sum()), 0.)
        finally:
            old_env.close()
            vector.close()


if __name__ == '__main__':
    unittest.main()
