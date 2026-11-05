"""Movement must preserve balance inputs, follow commands and release stop anchors."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from climber_legged_rl.mujoco_env import ClimberMujocoEnv, make_mujoco_env
from climber_legged_rl.velocity_env import ClimberVelocityEnv
from climber_legged_rl.velocity_training import command_cases, VelocityEvalCallback


class VelocityTaskTests(unittest.TestCase):
    def setUp(self):
        self.path=ROOT/'configs/robot/cad_velocity_round1.yaml'
        self.config=yaml.safe_load(self.path.read_text())
        self.model=ROOT/self.config['experiment']['output_directory']/'robot_experiment.xml'
        self.env=make_mujoco_env(self.model,self.path)

    def tearDown(self):self.env.close()

    def test_zero_command_matches_standing_observation_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            config=yaml.safe_load(self.path.read_text())
            config['task'].update(velocity_tracking=False,stationary_hold=True,command_forward_m_s=[0.,0.],command_yaw_rad_s=[0.,0.])
            path=Path(directory)/'stationary.yaml';path.write_text(yaml.safe_dump(config))
            standing=ClimberMujocoEnv(self.model,path)
            try:
                before,_=standing.reset(seed=53)
                after,_=self.env.reset(seed=53,options={'velocity_command':[0.,0.]})
                self.assertIsInstance(self.env,ClimberVelocityEnv)
                self.assertEqual(after.shape,(30,))
                np.testing.assert_array_equal(before,after)
            finally:standing.close()

    def test_forward_and_reverse_commands_ramp_without_position_anchor(self):
        for command in ([.15,0.],[-.15,0.],[0.,.3],[0.,-.3]):
            self.env.reset(seed=1,options={'velocity_command':command})
            self.env.steps=100;self.env._update_commands()
            self.env.steps=150;self.env._update_commands()
            np.testing.assert_allclose(self.env.commands,np.array(command)*.5)
            self.assertFalse(self.env.idle_hold)
            self.env.data.qpos[0]+=.3
            mujoco.mj_forward(self.env.model,self.env.data)
            np.testing.assert_allclose(self.env._observation()[24:28],0.,atol=1e-8)

    def test_stop_reanchors_at_new_position_instead_of_returning_to_start(self):
        self.env.reset(seed=1,options={'velocity_command':[.15,0.],'stop_after_seconds':5.})
        for steps in (100,200):self.env.steps=steps;self.env._update_commands()
        self.env.data.qpos[0]+=.5;mujoco.mj_forward(self.env.model,self.env.data)
        self.env.steps=500;self.env._update_commands()
        self.env.steps=600;self.env._update_commands()
        self.assertTrue(self.env.idle_hold)
        np.testing.assert_allclose(self.env._hold_errors()[0],0.,atol=1e-10)

    def test_rewards_use_command_error_not_zero_speed_and_totals_match(self):
        self.env.reset(seed=53,options={'velocity_command':[.15,0.]})
        self.env._new_command([.15,0.]);self.env.commands[:]=[.15,0.];self.env.idle_hold=False
        _,reward,_,_,info=self.env.step(np.zeros(6))
        v=info['forward_velocity_m_s'];side=info['lateral_velocity_m_s']
        expected=np.exp(-((v-.15)**2+side**2)/.08**2)*4.
        self.assertAlmostEqual(info['reward_terms']['linear_tracking'],expected)
        self.assertNotIn('position_hold',info['reward_terms'])
        self.assertAlmostEqual(reward,self.env._total_reward)

    def test_evaluation_cases_cover_both_directions_and_zero(self):
        cases=dict(command_cases(self.config))
        self.assertEqual(len(cases),7)
        self.assertEqual(cases['idle'],[0.,0.])
        self.assertEqual(cases['forward'],[.15,0.])
        self.assertEqual(cases['backward'],[-.15,0.])
        # A policy that stands still cannot pass either commanded translation.
        for key in ('forward','backward'):
            self.assertGreater(abs(cases[key][0]),self.config['task']['velocity_evaluation']['forward_rmse_m_s'])

    def test_invalid_command_rejected(self):
        with self.assertRaisesRegex(ValueError,'exceeds'):
            self.env.reset(options={'velocity_command':[5.,0.]})

    def test_single_axis_sampling_keeps_forward_and_turn_cases_separate(self):
        self.env.schedule=dict(self.env.schedule,axis_probability=1.,idle_probability=0.)
        self.env.reset(seed=57)
        targets=[]
        for _ in range(16):
            self.env.steps=round(self.env.next_command_seconds/self.env.dt)
            self.env._update_commands()
            targets.append(self.env.command_target.copy())
        self.assertTrue(all(np.count_nonzero(target)==1 for target in targets))
        self.assertTrue(any(target[0]!=0 for target in targets))
        self.assertTrue(any(target[1]!=0 for target in targets))

    def test_checkpoint_selection_preserves_parent_when_new_policy_loses_idle(self):
        with tempfile.TemporaryDirectory() as directory:
            saved=[]
            def save(path):
                Path(path).write_bytes(b'policy');saved.append(Path(path).name)
            callback=VelocityEvalCallback(directory,self.path,self.model,frequency=1)
            callback.model=SimpleNamespace(save=save,logger=SimpleNamespace(record=lambda *args:None))
            callback.num_timesteps=0
            def report(rates,reward):
                return dict(per_command_success=rates,velocity_success_fraction=sum(rates.values())/len(rates),
                    stop_success_fraction=1.,timeout_fraction=1.,mean_reward=reward,mean_episode_seconds=20.)
            first=report({'idle':1.,'forward':1.},10.)
            failed=report({'idle':0.,'forward':1.},1000.)
            with patch('climber_legged_rl.velocity_training.evaluate_velocity',side_effect=[first,failed]) as assess:
                callback._on_training_start()
                callback.num_timesteps=1;callback._on_step()
                self.assertEqual(assess.call_args.kwargs['episodes'],21)
            self.assertEqual(saved.count('best_model.zip'),1)
            self.assertEqual(callback.history[0]['timesteps'],0)
            self.assertFalse((Path(directory)/'velocity_candidate.zip').exists())
