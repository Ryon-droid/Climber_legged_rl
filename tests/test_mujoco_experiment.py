"""Optional runtime tests; skip where MuJoCo/Gymnasium are not installed."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"source"))
AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ("mujoco","gymnasium","scipy","yaml"))


@unittest.skipUnless(AVAILABLE,"MuJoCo experiment dependencies not installed")
class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from climber_legged_rl.mujoco_model import load_config,DEFAULT_CONFIG
        cls.config=load_config()
        cls.path=ROOT/cls.config["experiment"]["output_directory"]/"robot_experiment.xml"
        if not cls.path.is_file():
            raise unittest.SkipTest("Prepare the separate model first")
        cls.config_path=DEFAULT_CONFIG

    def setUp(self):
        from climber_legged_rl.mujoco_env import ClimberMujocoEnv
        self.env=ClimberMujocoEnv(self.path,self.config_path)

    def tearDown(self):
        self.env.close()

    def test_model_contract(self):
        import mujoco
        self.assertEqual(self.env.model.nu,6)
        self.assertEqual(self.env.model.jnt_type[0],mujoco.mjtJoint.mjJNT_FREE)
        self.assertAlmostEqual(float(self.env.model.body_mass.sum()),15.0,places=7)
        self.assertTrue(self.env.config["experiment"]["simulation_only"])

    def test_observations_and_seeded_reset(self):
        import numpy as np
        a,_=self.env.reset(seed=123)
        b,_=self.env.reset(seed=123)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(24,))
        self.assertTrue(self.env.observation_space.contains(a))
        self.assertGreater(self.env.data.body("left_link 3").xpos[1],self.env.data.body("right_link 3").xpos[1])

    def test_finite_dynamics_limits_and_reset_closure(self):
        import numpy as np
        from climber_legged_rl.mujoco_model import closure_error
        self.env.reset(seed=42)
        self.assertLess(np.linalg.norm(closure_error(self.env.model,self.env.data),axis=1).max(),0.0001)
        for _ in range(50):
            obs,reward,done,truncated,info=self.env.step(np.ones(6))
            self.assertTrue(np.isfinite(reward))
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertTrue((np.abs(self.env.data.ctrl)<=self.env.torque_limits+1e-10).all())
            if done or truncated:
                self.env.reset(seed=42)

    def test_invalid_action_is_rejected(self):
        import numpy as np
        self.env.reset(seed=42)
        with self.assertRaises(ValueError):
            self.env.step(np.zeros(5))
        with self.assertRaises(ValueError):
            self.env.step(np.full(6,np.nan))

    def test_time_limit_is_truncation(self):
        import numpy as np
        self.env.reset(seed=42)
        self.env.max_steps=1
        _,_,terminated,truncated,_=self.env.step(np.zeros(6))
        self.assertFalse(terminated)
        self.assertTrue(truncated)


if __name__=="__main__":
    unittest.main()
