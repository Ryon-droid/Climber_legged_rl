"""CAD training must preserve source mechanics and actually simulate six motors."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
AVAILABLE=all(importlib.util.find_spec(n) is not None for n in ('mujoco','numpy','scipy','yaml','gymnasium'))

@unittest.skipUnless(AVAILABLE,'CAD dynamics dependencies absent')
class CadDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from climber_legged_rl.cad_dynamics import build_dynamics,DEFAULT_DYNAMICS_CONFIG
        cls.temp=tempfile.TemporaryDirectory()
        cfg=yaml.safe_load(DEFAULT_DYNAMICS_CONFIG.read_text())
        cfg['experiment']['output_directory']=cls.temp.name
        cls.config=Path(cls.temp.name)/'config.yaml'
        cls.config.write_text(yaml.safe_dump(cfg))
        cls.path,cls.report=build_dynamics(cls.config)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        from climber_legged_rl.mujoco_env import ClimberMujocoEnv
        self.env=ClimberMujocoEnv(self.path,self.config)

    def tearDown(self):self.env.close()

    def test_masses_geometry_and_six_closures_are_preserved(self):
        import mujoco
        self.assertEqual(self.env.model.nu,6)
        self.assertEqual(self.env.model.neq,6)
        self.assertEqual(self.env.model.jnt_type[0],mujoco.mjtJoint.mjJNT_FREE)
        self.assertAlmostEqual(self.env.model.body_mass.sum(),16.46345659572714,places=8)
        self.assertEqual(self.env.model.opt.gravity.tolist(),[0.,0.,-9.81])
        source=ET.parse(ROOT/'outputs/cad_repaired/robot_cad_repaired.xml').getroot()
        generated=ET.parse(self.path).getroot()
        self.assertEqual([x.attrib for x in source.findall('.//body/inertial')],
                         [x.attrib for x in generated.findall('.//body/inertial')])
        self.assertEqual(len(source.findall('.//body/geom')),len(generated.findall('.//body/geom')))
        self.assertEqual(ET.tostring(source.find('equality')),ET.tostring(generated.find('equality')))

    def test_reset_has_clearance_closed_chains_and_reproducible_observations(self):
        import numpy as np
        from climber_legged_rl.cad_dynamics import geom_vertices
        from climber_legged_rl.mujoco_model import closure_error
        a,_=self.env.reset(seed=7);b,_=self.env.reset(seed=7)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(24,))
        self.assertLess(np.linalg.norm(closure_error(self.env.model,self.env.data),axis=1).max(),1e-4)
        # Use the exact unperturbed keyframe for mesh-ground clearance.
        import mujoco
        mujoco.mj_resetDataKeyframe(self.env.model,self.env.data,0)
        mujoco.mj_forward(self.env.model,self.env.data)
        lowest=min(geom_vertices(self.env.model,self.env.data,g)[:,2].min()
                   for g in range(self.env.model.ngeom) if self.env.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH)
        self.assertGreater(lowest,0)
        np.testing.assert_allclose(self.report['standing_geometry']['side_view_distance_m'],.135,atol=1e-6)
        np.testing.assert_allclose(self.env._rotation(),np.eye(3),atol=1e-8)
        self.assertGreater(self.env.target_hip_height,.20)

    def test_physical_motor_midpoint_distance_is_not_driven_joint_height(self):
        import mujoco
        import numpy as np
        from climber_legged_rl.stance_geometry import virtual_leg_geometry
        mujoco.mj_resetDataKeyframe(self.env.model,self.env.data,0)
        mujoco.mj_forward(self.env.model,self.env.data)
        geometry=virtual_leg_geometry(self.env.model,self.env.data,self.env.motor_midpoints,self.env.wheel_centers,self.env.chassis_axes)
        np.testing.assert_allclose(geometry['side_view_distance_m'],.135,atol=1e-6)
        np.testing.assert_allclose(geometry['motor_midpoints_world_m'][0,[0,2]],geometry['motor_midpoints_world_m'][1,[0,2]],atol=1e-6)
        self.assertLess(max(abs(geometry['forward_offset_m'])),.01)
        self.assertGreater(min(geometry['vertical_separation_m']),.13)
        self.assertTrue(np.all(geometry['vertical_separation_m']<geometry['side_view_distance_m']))
        for side in ('left','right'):
            self.assertGreater(self.report['motor_center_evidence'][side]['motor_spacing_side_view_m'],.19)
        _,_,_,_,info=self.env.step(np.zeros(6))
        self.assertIn('virtual_leg_length',info['reward_terms'])
        self.assertAlmostEqual(info['reward_terms']['hip_height'],0.)

    def test_physical_chassis_plane_matches_gravity_not_cad_root_z(self):
        import mujoco
        import numpy as np
        from climber_legged_rl.chassis_frame import frame_from_mounting_plate
        frame=frame_from_mounting_plate(self.env.model,'part_151_solid_151')
        np.testing.assert_allclose(frame.T@frame,np.eye(3),atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(frame),1.,places=10)
        # Independent CAD members on the same mounting plane must agree.
        for mesh in ('part_073_solid_073','part_074_solid_074'):
            other=frame_from_mounting_plate(self.env.model,mesh)
            np.testing.assert_allclose(other[:,2],frame[:,2],atol=1e-6)
        mujoco.mj_resetDataKeyframe(self.env.model,self.env.data,0)
        mujoco.mj_forward(self.env.model,self.env.data)
        np.testing.assert_allclose(self.env._rotation(),np.eye(3),atol=1e-8)
        self.assertGreater(np.degrees(np.arccos(abs(self.env.data.xmat[self.env.base_id].reshape(3,3)[2,2]))),60)

    def test_virtual_length_ignores_lateral_offsets_and_rigid_body_rotation(self):
        import mujoco
        import numpy as np
        from climber_legged_rl.stance_geometry import virtual_leg_geometry
        mujoco.mj_resetDataKeyframe(self.env.model,self.env.data,0)
        mujoco.mj_forward(self.env.model,self.env.data)
        original=virtual_leg_geometry(self.env.model,self.env.data,self.env.motor_midpoints,self.env.wheel_centers,self.env.chassis_axes)
        spatial=np.linalg.norm(original['motor_midpoints_world_m']-original['wheel_centers_world_m'],axis=1)
        self.assertTrue(np.all(spatial>.15))
        delta=np.array([np.cos(.4),0,0,np.sin(.4)])
        result=np.empty(4)
        mujoco.mju_mulQuat(result,delta,self.env.nominal_qpos[3:7])
        self.env.data.qpos[3:7]=result
        self.env.data.qpos[:3]+=[1,2,3]
        mujoco.mj_forward(self.env.model,self.env.data)
        rotated=virtual_leg_geometry(self.env.model,self.env.data,self.env.motor_midpoints,self.env.wheel_centers,self.env.chassis_axes)
        np.testing.assert_allclose(rotated['side_view_distance_m'],original['side_view_distance_m'],atol=1e-10)

    def test_tilted_chassis_has_a_strong_orientation_penalty(self):
        import mujoco
        import numpy as np
        mujoco.mj_resetDataKeyframe(self.env.model,self.env.data,0)
        # Move the whole robot up so this test measures posture without contacts.
        self.env.data.qpos[2]+=1
        mujoco.mj_forward(self.env.model,self.env.data)
        _,_,_,_,level=self.env.step(np.zeros(6))
        mujoco.mj_resetDataKeyframe(self.env.model,self.env.data,0)
        delta=np.array([np.cos(np.deg2rad(15)/2),0,np.sin(np.deg2rad(15)/2),0])
        result=np.empty(4)
        mujoco.mju_mulQuat(result,delta,self.env.nominal_qpos[3:7])
        self.env.data.qpos[3:7]=result
        self.env.data.qpos[2]+=1
        self.env.steps=0
        self.env.action[:]=0
        mujoco.mj_forward(self.env.model,self.env.data)
        _,_,_,_,tilted=self.env.step(np.zeros(6))
        self.assertGreater(level['reward_terms']['upright'],.9)
        self.assertLess(tilted['reward_terms']['upright'],.02)

    def test_each_motor_maps_to_requested_joint_and_applies_clipped_torque(self):
        import numpy as np
        names=self.env.control['leg_joints']+self.env.control['wheel_joints']
        for i,name in enumerate(names):
            self.assertEqual(self.env.model.actuator(i).trnid[0],self.env.model.joint(name).id)
            self.env.reset(seed=17)
            action=np.zeros(6);action[i]=.5
            obs,reward,_,_,_=self.env.step(action)
            self.assertTrue(np.isfinite(obs).all() and np.isfinite(reward))
            self.assertGreater(abs(self.env.data.ctrl[i]),1e-5)
            self.assertTrue((abs(self.env.data.ctrl)<=self.env.torque_limits+1e-8).all())
            np.testing.assert_allclose(self.env.data.qfrc_applied[self.env.spring_dofs],450.)

    def test_recovery_reward_terms_are_opt_in_and_fall_still_terminates(self):
        import numpy as np
        self.env.config['reward'].update(alive=.5,angular_velocity=.1,fall_penalty=20.)
        self.env.reset(seed=17)
        _,_,done,_,info=self.env.step(np.zeros(6))
        self.assertFalse(done)
        self.assertEqual(info['reward_terms']['alive'],.5)
        self.assertLessEqual(info['reward_terms']['angular_velocity'],0.)
        for _ in range(1000):
            _,reward,done,truncated,info=self.env.step(np.zeros(6))
            if done or truncated:
                break
        self.assertTrue(done)
        self.assertEqual(info['reward_terms']['alive'],0.)
        self.assertAlmostEqual(reward,sum(info['reward_terms'].values())-20.)

    def test_zero_and_random_actions_make_ground_contact_without_solver_warnings(self):
        import numpy as np
        rng=np.random.default_rng(13)
        for random in (False,True):
            self.env.reset(seed=13);contact=False
            for _ in range(200):
                obs,reward,done,truncated,info=self.env.step(rng.uniform(-1,1,6) if random else np.zeros(6))
                contact |= info['wheel_contacts']>0
                self.assertTrue(np.isfinite(reward) and np.isfinite(obs).all())
                self.assertEqual(int(np.sum(self.env.data.warning.number)),0)
                if done or truncated:self.env.reset(seed=13)
            self.assertTrue(contact)

    def test_training_gate_rejects_missing_or_stale_smoke_evidence(self):
        from climber_legged_rl.cad_dynamics import validate_artifacts
        with self.assertRaises(FileNotFoundError):validate_artifacts(self.config)

if __name__=='__main__':unittest.main()
