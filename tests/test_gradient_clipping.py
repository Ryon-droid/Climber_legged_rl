"""Critic scale must not suppress the independent actor's clipped gradient."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from test_velocity_estimation import StudentEnv, model
from climber_legged_rl.velocity_estimation import VelocityEstimationPPO


class GradientClippingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_large_critic_does_not_suppress_actor_and_both_groups_are_bounded(self):
        learner = model()
        learner.configure_gradient_clipping(True)
        learner.max_grad_norm = 1.
        learner.set_logger(configure(None, []))
        actor = learner.policy.log_std
        critic = learner.policy.value_net.bias
        captured = []

        def capture(optimizer, args, kwargs):
            captured.append((actor.grad.norm().item(), critic.grad.norm().item()))

        def fake_ppo_train(instance):
            instance.policy.optimizer.zero_grad(set_to_none=True)
            actor.grad = torch.full_like(actor, 2.)
            critic.grad = torch.full_like(critic, 1e6)
            # Exercise the same global clip then Adam path as stock PPO.
            torch.nn.utils.clip_grad_norm_(instance.policy.parameters(), instance.max_grad_norm)
            handle = instance.policy.optimizer.register_step_pre_hook(capture)
            try:
                instance.policy.optimizer.step()
            finally:
                handle.remove()

        with patch.object(PPO, 'train', fake_ppo_train):
            learner._train_ppo()
        np.testing.assert_allclose(captured, [[1., 1.]], atol=1e-5)
        self.assertEqual(learner.max_grad_norm, 1.)
        self.assertFalse(learner.policy.optimizer._optimizer_step_pre_hooks)
        self.assertGreater(learner.logger.name_to_value['train/critic_grad_norm'], 999999.)
        self.assertGreater(learner.logger.name_to_value['train/actor_grad_norm'], 4.)

    def test_hook_and_bound_restored_even_when_training_raises(self):
        learner = model()
        learner.configure_gradient_clipping(True)
        original = learner.max_grad_norm
        with patch.object(PPO, 'train', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                learner._train_ppo()
        self.assertEqual(learner.max_grad_norm, original)
        self.assertFalse(learner.policy.optimizer._optimizer_step_pre_hooks)

    def test_legacy_path_keeps_stock_clip_and_has_no_hook(self):
        learner = model()
        bound = learner.max_grad_norm

        def stock(instance):
            self.assertEqual(instance.max_grad_norm, bound)
            self.assertFalse(instance.policy.optimizer._optimizer_step_pre_hooks)

        with patch.object(PPO, 'train', stock):
            learner._train_ppo()

    def test_invalid_configuration_rejected(self):
        learner = model()
        for enabled in ('true', 1, None):
            with self.assertRaises(ValueError):
                learner.configure_gradient_clipping(enabled)
        for bound in (0., -1., np.nan, np.inf):
            learner.max_grad_norm = bound
            with self.assertRaises(ValueError):
                learner.configure_gradient_clipping(True)

    def test_real_update_freezes_rising_and_estimator_and_reloads_for_playback(self):
        learner = model()
        learner.configure_estimator({'trainable': False})
        learner.configure_gradient_clipping(True)
        frozen = {k: v.clone() for k, v in learner.policy.state_dict().items()
                  if 'rising_' in k or 'velocity_estimator.' in k}
        learner.learn(64)
        for key, value in frozen.items():
            torch.testing.assert_close(value, learner.policy.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(learner.policy.optimizer._optimizer_step_pre_hooks)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'policy.zip'
            learner.save(path)
            ordinary = PPO.load(path, device='cpu')
            resumed = VelocityEstimationPPO.load(path, env=StudentEnv(), device='cpu')
            self.assertTrue(resumed.separate_actor_critic_gradient_clipping)
            self.assertFalse(resumed.policy.optimizer._optimizer_step_pre_hooks)
            obs, _ = StudentEnv().reset(seed=11)
            np.testing.assert_array_equal(learner.predict(obs, deterministic=True)[0],
                                          ordinary.predict(obs, deterministic=True)[0])
            resumed.learn(32, reset_num_timesteps=False)


if __name__ == '__main__':
    unittest.main()
