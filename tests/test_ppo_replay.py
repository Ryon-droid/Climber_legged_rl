"""Rehearsal changes only actor parameters and saves ordinary playback policies."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from climber_legged_rl.ppo_replay import DemonstrationReplayPPO


class ReplayTests(unittest.TestCase):
    def test_backtracking_accepts_only_a_smaller_valid_actor_and_resets_moments(self):
        self.model._replay_settings['prone_gate_backtracks'] = 3
        previous = [p.detach().clone() for p in self.model._replay_parameters]
        with torch.no_grad():
            for p in self.model._replay_parameters:
                p.add_(1.)
                self.model.policy.optimizer.state[p] = {'step': 1}
                self.model._replay_optimizer.state[p] = {'step': 1}
        with patch.object(self.model, '_passes_prone_gate', side_effect=[False, False, True]) as gate:
            fraction = self.model._gate_actor_update(previous)
        self.assertEqual(fraction, .25)
        self.assertEqual(gate.call_count, 3)
        for parameter, old in zip(self.model._replay_parameters, previous):
            torch.testing.assert_close(parameter, old + .25)
            self.assertNotIn(parameter, self.model.policy.optimizer.state)
            self.assertNotIn(parameter, self.model._replay_optimizer.state)

    def test_backtracking_exhaustion_restores_exact_original_actor(self):
        self.model._replay_settings['prone_gate_backtracks'] = 2
        previous = [p.detach().clone() for p in self.model._replay_parameters]
        with torch.no_grad():
            for parameter in self.model._replay_parameters:
                parameter.add_(1.)
        with patch.object(self.model, '_passes_prone_gate', return_value=False) as gate:
            self.assertEqual(self.model._gate_actor_update(previous), 0.)
        self.assertEqual(gate.call_count, 3)
        for parameter, old in zip(self.model._replay_parameters, previous):
            torch.testing.assert_close(parameter, old, rtol=0, atol=0)

    def test_invalid_backtracking_settings_fail_early(self):
        for value in (-1, 5, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.model.configure_replay((self.x, self.y, np.ones(32)/32),
                    dict(self.settings, prone_gate_backtracks=value), 9)
        with self.assertRaisesRegex(ValueError, 'requires prone protection'):
            self.model.configure_replay((self.x, self.y, np.ones(32)/32),
                dict(self.settings, prone_gate_backtracks=1), 9)

    def test_prone_protection_gate_runs_without_random_pushes(self):
        self.model._gate_model='model.xml';self.model._gate_config='config.yaml'
        push=SimpleNamespace(enabled=True)
        def step(action):
            self.assertFalse(push.enabled)
            return np.zeros(3,dtype=np.float32),0.,False,True,dict(combined_success=True)
        closed=[]
        env=SimpleNamespace(push=push,pose_bank=[{}],
            reset=lambda **kwargs:(np.zeros(3,dtype=np.float32),{}),step=step,close=lambda:closed.append(True))
        with patch('climber_legged_rl.mujoco_env.make_mujoco_env',return_value=env):
            self.assertTrue(self.model._passes_prone_gate())
        self.assertEqual(closed,[True])

    def setUp(self):
        torch.set_num_threads(1)
        self.env = gym.make('Pendulum-v1')
        self.model = DemonstrationReplayPPO('MlpPolicy', self.env, n_steps=8, batch_size=8,
            n_epochs=1, device='cpu', seed=9)
        self.x = np.zeros((32, 3), dtype=np.float32)
        self.y = np.full((32, 1), .25, dtype=np.float32)
        self.settings = dict(updates_per_rollout=4, batch_size=16, learning_rate=1e-3)
        self.model.configure_replay((self.x, self.y, np.ones(32)/32), self.settings, 9)

    def tearDown(self):
        self.env.close()

    def test_actor_replay_does_not_modify_value_or_exploration_parameters(self):
        before = {k:v.clone() for k,v in self.model.policy.state_dict().items()}
        loss = self.model.replay_update()
        after = self.model.policy.state_dict()
        self.assertTrue(np.isfinite(loss))
        for key in before:
            if 'value_net' in key or key == 'log_std':
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(before[k], v) for k,v in after.items() if 'action_net' in k))

    def test_combined_update_runs_and_checkpoint_loads_without_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.set_logger(configure(directory, []))
            self.model.learn(total_timesteps=16)
            self.assertIn('train/demonstration_mse', self.model.logger.name_to_value)
            path=Path(directory)/'policy.zip'
            self.model.save(path)
            loaded=PPO.load(path, device='cpu')
            self.assertFalse(hasattr(loaded, '_replay_x'))
            expected,_=self.model.predict(self.x[:1], deterministic=True)
            actual,_=loaded.predict(self.x[:1], deterministic=True)
            np.testing.assert_array_equal(actual, expected)

    def test_rejects_observation_mapping_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            self.model.configure_replay((np.zeros((32,4),dtype=np.float32),self.y,np.ones(32)/32),self.settings,9)

    def test_failed_prone_gate_restores_actor_but_retains_critic_learning(self):
        self.model._success_gate = True
        before = {k:v.clone() for k,v in self.model.policy.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            self.model.set_logger(configure(directory, []))
            with patch.object(self.model, '_passes_prone_gate', return_value=False):
                self.model.learn(total_timesteps=16)
            after = self.model.policy.state_dict()
            actor_keys = [k for k in before if 'policy_net' in k or 'action_net' in k]
            for key in actor_keys:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(before[k], v) for k,v in after.items() if 'value_net' in k))
            self.assertEqual(self.model.logger.name_to_value['train/prone_actor_update_accepted'], 0.)
            for parameter in self.model._replay_parameters:
                self.assertNotIn(parameter, self.model.policy.optimizer.state)
                self.assertNotIn(parameter, self.model._replay_optimizer.state)
            path = Path(directory) / 'gated_policy.zip'
            self.model.save(path)
            loaded = PPO.load(path, device='cpu')
            self.assertFalse(hasattr(loaded, '_success_gate'))
            self.assertFalse(hasattr(loaded, '_gate_model'))
            self.assertFalse(hasattr(loaded, '_replay_x'))

    def test_successful_prone_gate_accepts_actor_update(self):
        self.model._success_gate = True
        before = self.model.policy.action_net.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            self.model.set_logger(configure(directory, []))
            with patch.object(self.model, '_passes_prone_gate', return_value=True):
                self.model.learn(total_timesteps=8)
            self.assertFalse(torch.equal(before, self.model.policy.action_net.weight))
            self.assertEqual(self.model.logger.name_to_value['train/prone_actor_update_accepted'], 1.)
