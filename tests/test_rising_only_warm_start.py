"""Warm starts must honor the protected standing actor during rising adaptation."""
from pathlib import Path
import sys
import tempfile
import unittest
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
from climber_legged_rl.mujoco_training import load_warm_start, load_config
from climber_legged_rl.velocity_estimation import VelocityEstimatedPolicy, VelocityEstimationPPO


class FixtureEnv(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, (183,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (6,), np.float32)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(183, np.float32), {}
    def step(self, action):
        return np.zeros(183, np.float32), 0., False, False, {}


class RisingOnlyWarmStartTests(unittest.TestCase):
    def test_loading_and_optimization_preserve_standing_and_update_rising(self):
        torch.set_num_threads(1)
        vector = DummyVecEnv([FixtureEnv])
        self.addCleanup(vector.close)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'parent.zip'
            parent = VelocityEstimationPPO(VelocityEstimatedPolicy, vector, n_steps=8, batch_size=8,
                policy_kwargs=dict(net_arch=dict(pi=[16, 8], vf=[16, 8])), device='cpu')
            parent.save(path)
            settings = dict(load_config(ROOT/'configs/robot/cad_balance_spring208.yaml')['training'],
                n_steps=8, batch_size=8, n_epochs=1, learning_rate=1e-3,
                freeze_rising_actor=False, freeze_standing_actor=True)
            model, _, _ = load_warm_start(path, vector, settings, 'cpu', 208)
            model.configure_estimator(dict(trainable=False, epochs=1, batch_size=8, learning_rate=1e-4))
            actor = model.policy.mlp_extractor.policy_net
            for module in (actor.standing_net, actor.standing_head, actor.velocity_estimator):
                self.assertTrue(all(not p.requires_grad for p in module.parameters()))
            self.assertTrue(all(p.requires_grad for p in actor.rising_net.parameters()))
            original = {k:v.detach().clone() for k,v in actor.state_dict().items()}
            observation = torch.randn(8, 183)
            observation[:, 179] = 0.
            loss = actor(observation).square().mean()
            model.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.policy.optimizer.step()
            changed = []
            for name, value in actor.state_dict().items():
                if name.startswith(('rising_net.', 'rising_head.')):
                    changed.append(not torch.equal(value, original[name]))
                else:
                    self.assertTrue(torch.equal(value, original[name]), name)
            self.assertTrue(any(changed))
            saved = Path(directory)/'student.zip'
            model.save(saved)
            reloaded = VelocityEstimationPPO.load(saved, device='cpu')
            self.assertFalse(reloaded.policy.freeze_rising_actor)
            np.testing.assert_array_equal(model.predict(observation.numpy(), deterministic=True)[0],
                                          reloaded.predict(observation.numpy(), deterministic=True)[0])


if __name__ == '__main__':
    unittest.main()
