"""Output torque envelopes and the physical rise-to-policy handoff."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))


class SupportedGetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from climber_legged_rl.cad_dynamics import build_dynamics
        from climber_legged_rl.getup import prepare_pose_bank
        cls.temp = tempfile.TemporaryDirectory()
        cls.config = Path(cls.temp.name)/'config.yaml'
        cfg = yaml.safe_load((ROOT/'configs/robot/cad_getup_stand_round6_support.yaml').read_text())
        cfg['experiment']['output_directory'] = cls.temp.name
        cls.config.write_text(yaml.safe_dump(cfg))
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

    def test_peak_window_returns_to_rated_then_historical_standing_limit(self):
        import numpy as np
        e = self.env
        e.reset(seed=1, options={'reset_mode':'folded'})
        e.steps = 59; e._set_available_torque()
        np.testing.assert_array_equal(e.torque_limits, [40,40,40,40,2,2])
        e.steps = 60; e._set_available_torque()
        np.testing.assert_array_equal(e.torque_limits, [20,20,20,20,2,2])
        e.steps = 102; e._set_available_torque()
        np.testing.assert_array_equal(e.torque_limits, [10,10,10,10,2,2])
        self.assertEqual((e.control['leg_kp'],e.control['leg_kd']), (20.,1.))
        # Changing controller stages never grants acquisition or moves the robot.
        self.assertFalse(e.acquired)

    def test_reference_and_handoff_never_write_physical_state(self):
        import numpy as np
        e = self.env
        e.reset(seed=1, options={'reset_mode':'folded'})
        before = e.data.qpos.copy()
        e._leg_targets(np.ones(6))
        e.steps = 102; e._set_available_torque()
        expected = np.clip(e.nominal_legs+.25, e.leg_low,e.leg_high)
        np.testing.assert_allclose(e._leg_targets(np.ones(6)), expected)
        np.testing.assert_array_equal(e.data.qpos,before)
        np.testing.assert_array_equal(e._leg_reference(),e.nominal_legs)
        self.assertFalse(e.acquired)

    def test_standing_reset_uses_original_actor_contract_and_no_peak(self):
        import numpy as np
        e = self.env
        obs,_ = e.reset(seed=1,options={'reset_mode':'standing'})
        self.assertEqual(obs.shape,(36,))
        self.assertEqual(e._phase(),1.)
        self.assertFalse(e._assisting())
        np.testing.assert_array_equal(e.torque_limits[:4],[10]*4)
        action = np.array([.5,-.5,1.,-1.,0,0])
        np.testing.assert_allclose(e._leg_targets(action),np.clip(e.nominal_legs+.25*action[:4],e.leg_low,e.leg_high))

    def test_peak_measurements_sample_every_physics_step(self):
        import numpy as np
        e = self.env
        e.reset(seed=1,options={'reset_mode':'folded'})
        for _ in range(65):
            e.step(np.zeros(6))
        self.assertTrue(np.any(e.over_rated_seconds>0))
        self.assertTrue(np.all(e.over_rated_seconds<=.6+1e-9))
        self.assertTrue(np.all(e.peak_torque_seen<=np.array([40]*4+[2,2])+1e-8))
        self.assertAlmostEqual(float(e.model.body_mass.sum()),16.463456595727138)
        self.assertEqual(e.model.neq,6)
        self.assertEqual(e.model.nmesh,511)
        self.assertEqual(e.getup['leg_tolerance_m'],.01)
        self.assertEqual(e.config['task']['stationary_max_displacement_m'],.05)


if __name__ == '__main__':
    unittest.main()
