"""Dynamic action freedom must retain actuator limits and strict final standing."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))

class DynamicGetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from climber_legged_rl.cad_dynamics import build_dynamics
        from climber_legged_rl.getup import prepare_pose_bank
        cls.temp=tempfile.TemporaryDirectory()
        cls.config=Path(cls.temp.name)/'config.yaml'
        cfg=yaml.safe_load((ROOT/'configs/robot/cad_getup_stand_round5_dynamic.yaml').read_text())
        cfg['experiment']['output_directory']=cls.temp.name
        cls.config.write_text(yaml.safe_dump(cfg))
        cls.model,_=build_dynamics(cls.config)
        prepare_pose_bank(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        from climber_legged_rl.mujoco_env import make_mujoco_env
        self.env=make_mujoco_env(self.model,self.config)

    def tearDown(self):
        self.env.close()

    def test_incremental_rise_targets_do_not_set_configuration_or_follow_clock(self):
        import numpy as np
        e=self.env
        e.reset(seed=1,options={'reset_mode':'folded'})
        before=e.data.qpos.copy()
        action=np.array([1,-1,.5,-.5,0,0])
        expected=np.clip(before[e.q_indices[:4]]+.6*action[:4],e.leg_low,e.leg_high)
        np.testing.assert_allclose(e._leg_targets(action),expected)
        e.steps=400
        np.testing.assert_allclose(e._leg_targets(action),expected)
        np.testing.assert_array_equal(e.data.qpos,before)
        np.testing.assert_array_equal(e._leg_reference(),e.nominal_legs)

    def test_standing_preserves_historical_action_mapping(self):
        import numpy as np
        e=self.env
        observation,_=e.reset(seed=1,options={'reset_mode':'standing'})
        action=np.array([.5,-.5,1,-1,0,0])
        np.testing.assert_allclose(e._leg_targets(action),np.clip(e.nominal_legs+.25*action[:4],e.leg_low,e.leg_high))
        self.assertEqual(observation.shape,(36,))
        self.assertEqual(e._phase(),1.)
        np.testing.assert_array_equal(e.torque_limits,[10,10,10,10,2,2])

    def test_pitch_and_roll_envelopes_apply_only_before_acquisition(self):
        e=self.env
        e.reset(seed=1,options={'reset_mode':'folded'})
        with patch.object(e,'_attitude',return_value=(1.,0.)):
            self.assertIsNone(e._termination_reason(True,.54,0.))
        with patch.object(e,'_attitude',return_value=(0.,.4)):
            self.assertEqual(e._termination_reason(True,.92,0.),'getup_roll')
        with patch.object(e,'_attitude',return_value=(1.4,0.)):
            self.assertEqual(e._termination_reason(True,.17,0.),'getup_pitch')
        e.acquired=True
        self.assertEqual(e._termination_reason(True,1.,0.),'base_contact')
        self.assertEqual(e._termination_reason(False,.54,0.),'tilt')

    def test_prone_survival_and_extra_lift_reward_do_not_imply_getup_success(self):
        import numpy as np
        e=self.env
        e.reset(seed=1,options={'reset_mode':'folded'})
        e.max_steps=1
        _,reward,_,truncated,info=e.step(np.zeros(6))
        self.assertTrue(truncated)
        self.assertFalse(info['combined_success'])
        self.assertFalse(info['getup_acquired'])
        self.assertFalse(info['getup_metrics']['guided_reference'])
        self.assertAlmostEqual(reward,info['episode_metrics']['reward'])
        self.assertLessEqual(max(abs(e.data.ctrl)-e.torque_limits),1e-8)

    def test_physics_xml_and_final_success_thresholds_match_previous_round(self):
        import yaml
        e=self.env
        previous=yaml.safe_load((ROOT/'configs/robot/cad_getup_stand_round4.yaml').read_text())
        for key in ('mass','physics','control','spring','task','reward','push'):
            self.assertEqual(e.config[key],previous[key])
        self.assertEqual(self.model.read_bytes(),(ROOT/'outputs/cad_getup_stand_round4/robot_experiment.xml').read_bytes())
        for key in ('leg_tolerance_m','height_tolerance_m','tilt_tolerance_rad','settle_seconds'):
            self.assertEqual(e.getup[key],previous['getup'][key])

if __name__=='__main__':
    unittest.main()
