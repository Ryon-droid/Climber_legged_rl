"""Combined get-up retains closed loops, ground contact and historical control limits."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))


class CombinedGetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from climber_legged_rl.cad_dynamics import build_dynamics
        from climber_legged_rl.getup import prepare_pose_bank
        cls.temp = tempfile.TemporaryDirectory()
        folder = Path(cls.temp.name)
        cls.config = folder / 'config.yaml'
        cfg = yaml.safe_load((ROOT / 'configs/robot/cad_getup_stand_push.yaml').read_text())
        cfg['experiment']['output_directory'] = str(folder)
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

    def test_folded_reset_is_closed_supported_and_preserves_inertials(self):
        import numpy as np
        import mujoco
        from climber_legged_rl.mujoco_model import closure_error
        old = mujoco.MjModel.from_xml_path(str(ROOT / 'outputs/cad_dynamics_round8_push_recovery/robot_experiment.xml'))
        np.testing.assert_array_equal(old.body_mass, self.env.model.body_mass)
        np.testing.assert_array_equal(old.body_inertia, self.env.model.body_inertia)
        for seed in range(5):
            obs, info = self.env.reset(seed=seed, options={'reset_mode': 'folded'})
            self.assertEqual(obs.shape, (36,))
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertFalse(self.env.acquired)
            self.assertEqual(info['reset_mode'], 'folded')
            self.assertLess(np.linalg.norm(closure_error(self.env.model, self.env.data), axis=1).max(), 1e-4)
            self.assertTrue(np.all(self.env.data.warning.number == 0))
        for pose in self.env.pose_bank:
            self.assertGreaterEqual(min(pose['base_bottom_m'], pose['wheel_bottom_m']), 0.)
        self.assertTrue(any(p['base_bottom_m'] < .001 for p in self.env.pose_bank))

    def test_prone_legs_extend_rearward_with_belly_and_both_tires_near_ground(self):
        import mujoco
        import numpy as np
        from climber_legged_rl.stance_geometry import virtual_leg_geometry
        for pose in self.env.pose_bank:
            self.env.data.qpos[:] = pose['qpos']
            mujoco.mj_forward(self.env.model, self.env.data)
            shape = virtual_leg_geometry(self.env.model, self.env.data, self.env.motor_midpoints,
                                         self.env.wheel_centers, self.env.chassis_axes)
            np.testing.assert_allclose(shape['forward_offset_m'], pose['rearward_offset_m'], atol=1e-5)
            self.assertGreater(pose['rearward_offset_m'], .35)
            self.assertLess(pose['base_bottom_m'], .0015)
            self.assertLess(pose['wheel_bottom_m'], .0015)
            np.testing.assert_allclose(self.env._rotation()[:, 2], [0, 0, 1], atol=1e-8)
            for j in range(1, self.env.model.njnt):
                if self.env.model.jnt_limited[j]:
                    value = self.env.data.qpos[self.env.model.jnt_qposadr[j]]
                    low, high = self.env.model.jnt_range[j]
                    self.assertGreaterEqual(value, low)
                    self.assertLessEqual(value, high)

    def test_joint_overrides_are_opt_in_and_parent_standing_pose_is_preserved(self):
        import mujoco
        import numpy as np
        old = mujoco.MjModel.from_xml_path(str(ROOT / 'outputs/cad_dynamics_round8_push_recovery/robot_experiment.xml'))
        np.testing.assert_array_equal(old.key_qpos[0], self.env.model.key_qpos[0])
        changed = {self.env.model.joint(j).name for j in range(self.env.model.njnt)
                   if not np.array_equal(old.jnt_range[j], self.env.model.jnt_range[j])}
        expected = set(self.env.config['physics']['joint_range_overrides_rad'])
        expected.update(('left_gas_slide_joint', 'right_gas_slide_joint'))
        self.assertEqual(changed, expected)

    def test_preparing_prone_requires_supplied_spring_lengths(self):
        from climber_legged_rl.cad_dynamics import build_dynamics
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            cfg = yaml.safe_load(self.config.read_text())
            cfg['spring'].pop('eye_center_limits_m')
            path = Path(directory)/'config.yaml'
            path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, 'pin-center lengths are required'):
                build_dynamics(path)

    def test_pin_center_bounds_are_translated_to_slider_displacement(self):
        import numpy as np
        from climber_legged_rl.spring_geometry import eye_center_distance
        for pose in self.env.pose_bank:
            self.env.data.qpos[:] = pose['qpos']
            import mujoco
            mujoco.mj_forward(self.env.model, self.env.data)
            for side in ('left', 'right'):
                joint = self.env.model.joint(side+'_gas_slide_joint')
                self.assertTrue(self.env.model.jnt_limited[joint.id])
                zero = np.linalg.norm(self.env.model.body(side+'_gas_rod').pos)
                np.testing.assert_allclose(joint.range+zero, [.155, .220])
                length = eye_center_distance(self.env.model, self.env.data, side)
                self.assertGreaterEqual(length, .155)
                self.assertLessEqual(length, .220)

    def test_unknown_compressed_length_is_kept_explicit(self):
        from climber_legged_rl.spring_geometry import configured_eye_limits
        cfg = copy.deepcopy(self.env.config)
        cfg['spring']['eye_center_limits_m'] = {'left': [None, .22], 'right': [None, .22]}
        limits = configured_eye_limits(cfg, required=True)
        self.assertIsNone(limits['left'][0])
        self.assertEqual(limits['left'][1], .22)

    def test_dynamic_end_stops_limit_travel_under_nominal_spring_force(self):
        import numpy as np
        from climber_legged_rl.spring_geometry import eye_center_distance
        self.env.pose_bank = [self.env.pose_bank[-1]]
        self.env.reset(seed=99102, options={'reset_mode': 'folded'})
        for _ in range(350):
            _, _, done, truncated, _ = self.env.step(np.zeros(6))
            for side in ('left', 'right'):
                length = eye_center_distance(self.env.model, self.env.data, side)
                self.assertGreaterEqual(length, .154)
                self.assertLessEqual(length, .221)
            if done or truncated:
                break

    def test_contact_is_allowed_only_before_standing_and_push_is_gated(self):
        import numpy as np
        self.env.reset(seed=1, options={'reset_mode': 'folded'})
        self.env.steps = 301
        self.env.push.episode_enabled = True
        self.env.push.next_step = 0
        np.testing.assert_array_equal(self.env._push_force(), 0.)
        self.assertIsNone(self.env._termination_reason(True, 1., 0.))
        self.env.acquired = True
        self.assertEqual(self.env._termination_reason(True, 1., 0.), 'base_contact')
        self.env.acquired = False
        self.env.steps = 801
        self.assertEqual(self.env._termination_reason(True, 1., 0.), 'getup_timeout')
        self.env.steps = 0
        self.assertEqual(self.env._termination_reason(False, 1., .1), 'closure')

    def test_reference_moves_smoothly_without_changing_action_scale_or_limits(self):
        import numpy as np
        self.env.reset(seed=2, options={'reset_mode': 'folded'})
        np.testing.assert_allclose(self.env._leg_reference(), self.env.reset_legs)
        self.env.steps = int(self.env.getup['rise_seconds'] / self.env.dt)
        np.testing.assert_allclose(self.env._leg_reference(), self.env.nominal_legs)
        self.assertEqual(self.env.control['leg_action_scale_rad'], .25)
        np.testing.assert_array_equal(self.env.torque_limits, [10, 10, 10, 10, 2, 2])
        self.env.reset(seed=1, options={'reset_mode': 'folded'})
        _, reward, _, _, info = self.env.step(np.zeros(6))
        self.assertTrue(np.isfinite(reward))
        self.assertIn('getup_height', info['reward_terms'])

    def test_resume_rejects_new_getup_settings(self):
        import yaml
        from climber_legged_rl.mujoco_training import validate_policy_transfer
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / 'best').mkdir()
            checkpoint = run / 'best/best_model.zip'
            checkpoint.write_bytes(b'fixture')
            (run / 'robot_experiment.xml').write_bytes(self.model.read_bytes())
            config = yaml.safe_load(self.config.read_text())
            (run / 'config.yaml').write_text(yaml.safe_dump(config))
            config['getup']['rise_seconds'] = 4.
            with self.assertRaisesRegex(ValueError, 'getup'):
                validate_policy_transfer(checkpoint, self.model, config)
            validate_policy_transfer(checkpoint, self.model, config, warm_start=True)

    def test_no_push_success_rejects_heading_drift_after_getup(self):
        import numpy as np
        self.env.reset(seed=1, options={'reset_mode': 'folded'})
        self.env.push.episode_enabled = False
        self.env.acquired = True
        self.env.acquisition_step = 1
        self.env.hold_after_rise_max_heading = .2
        self.env.data.qpos[:] = self.env.nominal_qpos
        import mujoco
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.max_steps = 1
        _, _, _, truncated, info = self.env.step(np.zeros(6))
        self.assertTrue(truncated)
        self.assertFalse(info['combined_success'])

    def test_30_to_36_extension_preserves_old_actions_and_resets_optimizer(self):
        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from climber_legged_rl.mujoco_env import ClimberMujocoEnv
        from climber_legged_rl.mujoco_training import load_warm_start
        old_env = ClimberMujocoEnv(self.model, self.config)
        vector = DummyVecEnv([lambda: self.env])
        try:
            with tempfile.TemporaryDirectory() as temp:
                old = PPO('MlpPolicy', old_env, n_steps=8, batch_size=8, n_epochs=1,
                          policy_kwargs={'net_arch': {'pi': [8], 'vf': [8]}}, device='cpu', seed=7)
                old.num_timesteps = 123
                path = Path(temp) / 'policy.zip'
                old.save(path)
                settings = dict(self.env.config['training'], n_steps=8, batch_size=8, n_epochs=1)
                new, parent, expanded = load_warm_start(path, vector, settings, 'cpu', 50)
                self.assertEqual(parent, 123)
                self.assertTrue(expanded)
                inputs = np.random.default_rng(1).normal(0, .1, (4, 30)).astype(np.float32)
                before, _ = old.predict(inputs, deterministic=True)
                after, _ = new.predict(np.column_stack([inputs, np.ones((4, 6), np.float32)]), deterministic=True)
                np.testing.assert_allclose(before, after, atol=1e-6)
        finally:
            old_env.close()

    def test_closed_chain_waypoints_use_solved_route_and_preserve_reset_and_standing(self):
        import numpy as np
        self.env.getup['reference_mode']='closed_chain_waypoints'
        for pose in self.env.pose_bank:
            self.env.pose_bank=[pose]
            self.env.reset(seed=1,options={'reset_mode':'folded'})
            np.testing.assert_allclose(self.env._leg_reference(),self.env.reset_legs)
            self.assertTrue(np.all(np.diff(self.env.rise_progress)>0))
            self.env.steps=round(self.env.getup['rise_seconds']/self.env.dt)
            np.testing.assert_allclose(self.env._leg_reference(),self.env.nominal_legs,atol=1e-8)
        self.env.reset(seed=1,options={'reset_mode':'standing'})
        self.assertIsNone(self.env.rise_waypoints)
        np.testing.assert_allclose(self.env._leg_reference(),self.env.nominal_legs)

    def test_prone_posture_reward_is_gated_by_progress_but_standing_is_unchanged(self):
        import numpy as np
        self.env.getup['dense_progress_reward']=6.
        self.env.getup['unfinished_penalty']=.1
        self.env.reset(seed=1,options={'reset_mode':'folded'})
        _,reward,_,_,info=self.env.step(np.zeros(6))
        self.assertLess(info['reward_terms']['upright'],1.)
        self.assertEqual(info['reward_terms']['getup_unfinished'],-.1)
        self.assertGreater(reward,0.)
        self.assertTrue(np.isfinite(reward))
        self.env.reset(seed=1,options={'reset_mode':'standing'})
        _,_,_,_,info=self.env.step(np.zeros(6))
        self.assertNotIn('getup_unfinished',info['reward_terms'])
