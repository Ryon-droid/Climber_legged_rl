"""Rise-finish resets are feasible and never count as full-prone successes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
ROOT=Path(__file__).resolve().parents[1]

class GetupCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from climber_legged_rl.cad_dynamics import build_dynamics
        from climber_legged_rl.getup import prepare_pose_bank
        from climber_legged_rl.getup_curriculum import prepare_curriculum_bank
        cls.temp=tempfile.TemporaryDirectory()
        folder=Path(cls.temp.name)
        cls.config=folder/'config.yaml'
        cfg=yaml.safe_load((ROOT/'configs/robot/cad_getup_stand_round2.yaml').read_text())
        cfg['experiment']['output_directory']=str(folder)
        cls.config.write_text(yaml.safe_dump(cfg))
        cls.model,_=build_dynamics(cls.config)
        prepare_pose_bank(cls.config)
        prepare_curriculum_bank(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        from climber_legged_rl.mujoco_env import make_mujoco_env
        self.env=make_mujoco_env(self.model,self.config)

    def tearDown(self):
        self.env.close()

    def test_reset_labels_and_paths_do_not_confuse_curriculum_with_prone(self):
        import numpy as np
        from climber_legged_rl.mujoco_model import closure_error
        for mode in ('curriculum','folded','standing'):
            obs,info=self.env.reset(seed=2,options={'reset_mode':mode})
            self.assertEqual(info['reset_mode'],mode)
            self.assertEqual(obs.shape,(36,))
            self.assertEqual(self.env.acquired,mode=='standing')
            self.assertLess(np.linalg.norm(closure_error(self.env.model,self.env.data),axis=1).max(),1e-4)
            if mode=='curriculum':self.assertFalse(info['full_prone'])
            if mode=='folded':self.assertGreater(info['reset_length_m'],.35)
        for pose in self.env.curriculum_bank:
            self.assertTrue(0<pose['curriculum_start_progress']<1)
            self.assertTrue(all(.155<=v<=.22 for v in pose['spring_eye_distances_m'].values()))

    def test_successful_curriculum_is_reported_separately_and_needs_settling(self):
        import numpy as np
        self.env.reset(seed=2,options={'reset_mode':'curriculum'})
        self.env.max_steps=1
        _,_,done,truncated,info=self.env.step(np.zeros(6))
        self.assertEqual(info['getup_metrics']['reset_mode'],'curriculum')
        self.assertFalse(info['combined_success'])
        self.assertFalse(info['getup_acquired'])

    def test_same_seed_produces_same_curriculum_selection(self):
        import numpy as np
        obs,_=self.env.reset(seed=85)
        qpos=self.env.data.qpos.copy()
        again,_=self.env.reset(seed=85)
        np.testing.assert_array_equal(obs,again)
        np.testing.assert_array_equal(qpos,self.env.data.qpos)

    def test_shorter_than_target_reset_cannot_earn_progress_by_retracting_further(self):
        import numpy as np
        from unittest.mock import patch
        from climber_legged_rl.mujoco_env import ClimberMujocoEnv
        self.env.reset(seed=1,options={'reset_mode':'curriculum'})
        info=dict(virtual_leg_lengths_m=[.12,.12],reward_terms={},wheel_contacts=0,hip_height_m=.1,
            tilt_rad=0.,planar_displacement_m=0.,heading_error_rad=0.,planar_speed_m_s=0.,yaw_rate_rad_s=0.)
        self.env.reset_length_m=.134
        with patch.object(ClimberMujocoEnv,'step',return_value=(self.env._observation(),0.,False,False,info)):
            _,_,_,_,result=self.env.step(np.zeros(6))
            self.assertEqual(result['getup_progress'],0.)

    def test_curriculum_duration_changes_only_the_curriculum_reference_clock(self):
        self.env.settings['rise_seconds']=1.
        self.env.reset(seed=1,options={'reset_mode':'curriculum'})
        self.env.steps=50
        self.assertEqual(self.env._phase(),.5)
        self.env.reset(seed=1,options={'reset_mode':'folded'})
        self.env.steps=50
        self.assertEqual(self.env._phase(),.125)
        self.env.reset(seed=1,options={'reset_mode':'standing'})
        self.env.steps=50
        self.assertEqual(self.env._phase(),1.)

    def test_curriculum_duration_must_leave_time_for_settling(self):
        import yaml
        from climber_legged_rl.getup_curriculum import prepare_curriculum_bank
        with tempfile.TemporaryDirectory() as directory:
            cfg=yaml.safe_load(self.config.read_text())
            cfg['getup']['reset_curriculum']['rise_seconds']=10.
            path=Path(directory)/'bad.yaml'
            path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError,'settling time'):
                prepare_curriculum_bank(path)
