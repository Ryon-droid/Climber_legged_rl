"""Curriculum warm starts may change task/reward, never physics or controls."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
AVAILABLE=all(importlib.util.find_spec(n) is not None for n in ('stable_baselines3','yaml','mujoco'))

@unittest.skipUnless(AVAILABLE,'Training dependencies absent')
class PolicyTransferTests(unittest.TestCase):
    def setUp(self):
        import yaml
        self.temp=tempfile.TemporaryDirectory()
        self.folder=Path(self.temp.name)
        previous=self.folder/'previous'
        (previous/'best').mkdir(parents=True)
        self.checkpoint=previous/'best/best_model.zip'
        self.checkpoint.write_bytes(b'transfer validation fixture')
        self.model=self.folder/'robot_experiment.xml'
        self.model.write_text('<mujoco/>')
        (previous/'robot_experiment.xml').write_bytes(self.model.read_bytes())
        self.config=yaml.safe_load((ROOT/'configs/robot/cad_dynamics.yaml').read_text())
        (previous/'config.yaml').write_text(yaml.safe_dump(self.config))
        from climber_legged_rl.mujoco_training import validate_policy_transfer
        self.validate=validate_policy_transfer

    def tearDown(self):self.temp.cleanup()

    def test_resume_accepts_identical_task(self):
        self.assertEqual(self.validate(self.checkpoint,self.model,self.config),self.checkpoint)

    def test_resume_rejects_task_change(self):
        self.config['task']['episode_seconds']=20
        with self.assertRaisesRegex(ValueError,'task'):
            self.validate(self.checkpoint,self.model,self.config)

    def test_warm_start_accepts_new_task_reward_and_learning_rate(self):
        self.config['task']['episode_seconds']=20
        self.config['reward']['upright']=4
        self.config['training']['learning_rate']=1e-4
        self.assertEqual(self.validate(self.checkpoint,self.model,self.config,warm_start=True),self.checkpoint)

    def test_warm_start_rejects_physics_control_and_spring_changes(self):
        for section,key in [('physics','timestep'),('control','leg_kp'),('spring','force_N')]:
            config=copy.deepcopy(self.config)
            config[section][key]*=2
            with self.assertRaisesRegex(ValueError,section):
                self.validate(self.checkpoint,self.model,config,warm_start=True)

    def test_warm_start_rejects_changed_model_bytes(self):
        self.model.write_text('<mujoco model="different"/>')
        with self.assertRaisesRegex(ValueError,'identical'):
            self.validate(self.checkpoint,self.model,self.config,warm_start=True)

    def test_new_dynamic_action_mapping_requires_explicit_warm_start_flag(self):
        self.config['getup']={'enabled':True,'dynamic_control':{'enabled':True}}
        with self.assertRaisesRegex(ValueError,'action map'):
            self.validate(self.checkpoint,self.model,self.config,warm_start=True)
        self.config['training']['allow_dynamic_getup_warm_start']=True
        self.assertEqual(self.validate(self.checkpoint,self.model,self.config,warm_start=True),self.checkpoint)

    def test_explicit_same_dimension_optimizer_reset_preserves_policy_actions(self):
        import gymnasium as gym
        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from climber_legged_rl.mujoco_training import load_warm_start
        vector=DummyVecEnv([lambda:gym.make('CartPole-v1')])
        try:
            previous=PPO('MlpPolicy',vector,n_steps=8,batch_size=8,n_epochs=1,device='cpu')
            previous.learn(8)
            self.assertTrue(previous.policy.optimizer.state)
            path=self.folder/'optimizer_fixture.zip'
            previous.save(path)
            settings=dict(self.config['training'],n_steps=8,batch_size=8,n_epochs=1,reset_optimizer_on_warm_start=True,target_kl=.005)
            settings.pop('vf_coef', None)
            inherited,_,_=load_warm_start(path,vector,settings,'cpu',63)
            self.assertEqual(inherited.vf_coef, previous.vf_coef)
            settings['vf_coef']=.0001
            current,_,expanded=load_warm_start(path,vector,settings,'cpu',63)
            self.assertFalse(expanded)
            self.assertEqual(current.vf_coef,.0001)
            self.assertEqual(current.target_kl,.005)
            self.assertFalse(current.policy.optimizer.state)
            observation=np.zeros((1,4),dtype=np.float32)
            np.testing.assert_array_equal(previous.predict(observation,deterministic=True)[0],current.predict(observation,deterministic=True)[0])
        finally:
            vector.close()

    def supported_getup_fixture(self):
        import yaml
        import xml.etree.ElementTree as ET
        self.config = yaml.safe_load((ROOT/'configs/robot/cad_getup_stand_round7_handoff.yaml').read_text())
        previous = copy.deepcopy(self.config)
        previous['getup'].pop('support_control')
        previous['control'].update(leg_kp=20., leg_kd=1., leg_torque_limit_Nm=10.)
        (self.checkpoint.parent.parent/'config.yaml').write_text(yaml.safe_dump(previous))
        old = ET.fromstring('<mujoco><worldbody><body name="base"/></worldbody><actuator/></mujoco>')
        for index, joint in enumerate(previous['control']['leg_joints']):
            ET.SubElement(old.find('actuator'), 'motor', name=f'motor_{index}', joint=joint,
                          gear='1', ctrllimited='true', ctrlrange='-10.0 10.0')
        (self.checkpoint.parent.parent/'robot_experiment.xml').write_text(ET.tostring(old, encoding='unicode'))
        new = copy.deepcopy(old)
        for motor in new.findall('./actuator/motor'):
            motor.set('ctrlrange', '-40.0 40.0')
        self.model.write_text(ET.tostring(new, encoding='unicode'))
        return new

    def test_same_dimension_warm_start_can_change_worker_count(self):
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from climber_legged_rl.mujoco_training import load_warm_start
        old_vector = DummyVecEnv([lambda:gym.make('CartPole-v1')])
        new_vector = DummyVecEnv([lambda:gym.make('CartPole-v1') for _ in range(2)])
        try:
            policy = PPO('MlpPolicy',old_vector,n_steps=8,batch_size=8,n_epochs=1,device='cpu')
            path = self.folder/'worker_fixture.zip'
            policy.save(path)
            settings = dict(self.config['training'],n_steps=8,batch_size=8,n_epochs=1)
            current,_,expanded = load_warm_start(path,new_vector,settings,'cpu',63)
            self.assertFalse(expanded)
            current.learn(16)
            self.assertEqual(current.n_envs,2)
            self.assertEqual(current.num_timesteps,16)
        finally:
            old_vector.close()
            new_vector.close()

    def test_supported_transfer_requires_explicit_flag_and_preserves_standing_map(self):
        self.supported_getup_fixture()
        self.assertEqual(self.validate(self.checkpoint,self.model,self.config,warm_start=True),self.checkpoint)
        self.config['training']['allow_supported_getup_warm_start'] = False
        with self.assertRaisesRegex(ValueError, 'explicit'):
            self.validate(self.checkpoint,self.model,self.config,warm_start=True)

    def test_full_policy_transfer_requires_explicit_flag_and_preserves_physics(self):
        import yaml
        self.supported_getup_fixture()
        self.config = yaml.safe_load((ROOT/'configs/robot/cad_getup_stand_round8_policy.yaml').read_text())
        self.assertEqual(self.validate(self.checkpoint,self.model,self.config,warm_start=True),self.checkpoint)
        for section, key in [('physics','timestep'), ('spring','force_N'), ('control','wheel_kv')]:
            changed = copy.deepcopy(self.config)
            changed[section][key] = 'unexpected'
            with self.assertRaises(ValueError):
                self.validate(self.checkpoint,self.model,changed,warm_start=True)
        self.config['training']['allow_policy_getup_warm_start'] = False
        with self.assertRaisesRegex(ValueError, 'explicit'):
            self.validate(self.checkpoint,self.model,self.config,warm_start=True)

    def test_full_policy_transfer_rejects_xml_changes_and_longer_peak_window(self):
        import yaml
        import xml.etree.ElementTree as ET
        new = self.supported_getup_fixture()
        self.config = yaml.safe_load((ROOT/'configs/robot/cad_getup_stand_round8_policy.yaml').read_text())
        new.find('.//body').set('pos', '1 0 0')
        self.model.write_text(ET.tostring(new,encoding='unicode'))
        with self.assertRaisesRegex(ValueError, 'XML'):
            self.validate(self.checkpoint,self.model,self.config,warm_start=True)
        self.supported_getup_fixture()
        self.config = yaml.safe_load((ROOT/'configs/robot/cad_getup_stand_round8_policy.yaml').read_text())
        self.config['getup']['policy_control']['peak_window_seconds'] = 5.
        with self.assertRaisesRegex(ValueError, 'envelope'):
            self.validate(self.checkpoint,self.model,self.config,warm_start=True)

    def test_supported_transfer_rejects_unrelated_mechanics_and_control_changes(self):
        self.supported_getup_fixture()
        for section, key in [('mass','method'), ('physics','timestep'), ('spring','force_N'),
                             ('control','wheel_kv')]:
            changed = copy.deepcopy(self.config)
            changed[section][key] = 'unexpected'
            with self.assertRaises(ValueError):
                self.validate(self.checkpoint,self.model,changed,warm_start=True)
        self.config['getup']['support_control']['standing_torque_limit_Nm'] = 20.
        with self.assertRaisesRegex(ValueError, 'standing'):
            self.validate(self.checkpoint,self.model,self.config,warm_start=True)

    def test_supported_transfer_rejects_xml_changes_and_invalid_peak_schedule(self):
        import xml.etree.ElementTree as ET
        new = self.supported_getup_fixture()
        new.find('.//body').set('pos', '1 0 0')
        self.model.write_text(ET.tostring(new,encoding='unicode'))
        with self.assertRaisesRegex(ValueError, 'XML'):
            self.validate(self.checkpoint,self.model,self.config,warm_start=True)
        self.supported_getup_fixture()
        self.config['getup']['support_control']['peak_window_seconds'] = 10.
        with self.assertRaisesRegex(ValueError, 'schedule'):
            self.validate(self.checkpoint,self.model,self.config,warm_start=True)

    def spring_limit_fixture(self):
        import yaml
        import xml.etree.ElementTree as ET
        self.config['spring'].pop('eye_center_limits_m', None)
        (self.checkpoint.parent.parent/'config.yaml').write_text(yaml.safe_dump(self.config))
        old = ET.fromstring('<mujoco><worldbody><body name="base"/></worldbody></mujoco>')
        for side in ('left', 'right'):
            body = ET.SubElement(old.find('.//body'), 'body', name=side+'_gas_rod', pos='0.154 0 0')
            ET.SubElement(body, 'joint', name=side+'_gas_slide_joint', type='slide', limited='false')
        (self.checkpoint.parent.parent/'robot_experiment.xml').write_text(ET.tostring(old,encoding='unicode'))
        new = copy.deepcopy(old)
        for joint in new.findall('.//joint'):
            joint.set('limited','true'); joint.set('range','0.001 0.066')
            joint.set('solreflimit','0.002 1')
            joint.set('solimplimit','0.999 0.9999 0.0001 0.5 2')
        self.model.write_text(ET.tostring(new,encoding='unicode'))
        self.config['spring']['eye_center_limits_m']={'left':[.155,.22],'right':[.155,.22]}
        self.config['training']['allow_spring_limit_warm_start']=True
        return new

    def test_explicit_spring_limit_transfer_accepts_only_two_slider_stops(self):
        self.spring_limit_fixture()
        self.assertEqual(self.validate(self.checkpoint,self.model,self.config,warm_start=True),self.checkpoint)
        with self.assertRaisesRegex(ValueError,'identical'):
            self.validate(self.checkpoint,self.model,self.config,warm_start=False)

    def test_spring_limit_transfer_rejects_wrong_length_or_other_model_changes(self):
        import xml.etree.ElementTree as ET
        new=self.spring_limit_fixture()
        for kind in ('wrong_length','unrelated'):
            changed=copy.deepcopy(new)
            if kind=='wrong_length':changed.find('.//joint').set('range','0.001 0.07')
            else:changed.find('.//body').set('mass','99')
            self.model.write_text(ET.tostring(changed,encoding='unicode'))
            with self.assertRaises(ValueError):
                self.validate(self.checkpoint,self.model,self.config,warm_start=True)

    def getup_guard_fixture(self):
        import yaml
        import xml.etree.ElementTree as ET
        new = self.spring_limit_fixture()
        old_path = self.checkpoint.parent.parent/'robot_experiment.xml'
        old = ET.parse(old_path).getroot()
        cfg = yaml.safe_load((ROOT/'configs/robot/cad_getup_stand.yaml').read_text())
        self.config['physics']['joint_range_overrides_rad'] = cfg['physics']['joint_range_overrides_rad']
        self.config['getup'] = cfg['getup']
        self.config['training'].pop('allow_spring_limit_warm_start')
        self.config['training']['allow_getup_guard_warm_start'] = True
        for name, bounds in cfg['physics']['joint_range_overrides_rad'].items():
            ET.SubElement(old.find('.//body'), 'joint', name=name, type='hinge', limited='true', range='-1.57 1.57')
            ET.SubElement(new.find('.//body'), 'joint', name=name, type='hinge', limited='true', range=' '.join(map(str,bounds)))
        old_path.write_text(ET.tostring(old,encoding='unicode'))
        self.model.write_text(ET.tostring(new,encoding='unicode'))
        return new

    def test_getup_guard_transfer_is_explicit_and_keeps_physics_and_torque_limits(self):
        self.getup_guard_fixture()
        self.assertEqual(self.validate(self.checkpoint,self.model,self.config,warm_start=True),self.checkpoint)
        for section,key,value in [('physics','timestep',.002), ('control','leg_torque_limit_Nm',20), ('spring','force_N',0)]:
            changed=copy.deepcopy(self.config)
            changed[section][key]=value
            with self.assertRaises(ValueError):
                self.validate(self.checkpoint,self.model,changed,warm_start=True)
        changed=copy.deepcopy(self.config)
        changed['training']['allow_getup_guard_warm_start']=False
        with self.assertRaisesRegex(ValueError,'identical'):
            self.validate(self.checkpoint,self.model,changed,warm_start=True)

    def test_getup_guard_transfer_rejects_unlisted_xml_or_config_changes(self):
        import xml.etree.ElementTree as ET
        new=self.getup_guard_fixture()
        for key,value in [('damping','9'),('range','-4 4')]:
            changed=copy.deepcopy(new)
            changed.find('.//joint[@name="left_link2_joint"]').set(key,value)
            self.model.write_text(ET.tostring(changed,encoding='unicode'))
            with self.assertRaises(ValueError):
                self.validate(self.checkpoint,self.model,self.config,warm_start=True)
        self.model.write_text(ET.tostring(new,encoding='unicode'))
        self.config['physics']['joint_range_overrides_rad']['left_link2_joint']=[-4,4]
        with self.assertRaises(ValueError):
            self.validate(self.checkpoint,self.model,self.config,warm_start=True)

if __name__=='__main__':unittest.main()
