"""Successful history rehearsal changes standing only and survives training resume."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from stable_baselines3 import PPO

from test_velocity_estimation import model, StudentEnv
from climber_legged_rl.standing_rehearsal import StandingRehearsal, load_standing_rehearsal
from climber_legged_rl.velocity_estimation import VelocityEstimationPPO


def samples(learner):
    x = np.random.default_rng(4).normal(size=(128, 180)).astype(np.float32)
    x.reshape(128, 5, 36)[:, :, 28:30] = 0.
    x[:, -1] = 1.
    with torch.no_grad():
        y = learner.policy.mlp_extractor.policy_net(torch.from_numpy(x)).clamp(-1, 1).numpy()
    return x, y, np.full(128, 1/128)


def settings():
    return dict(enabled=True, updates_per_rollout=4, batch_size=32, learning_rate=.001,
                max_grad_norm=1., action_error_scales=[.05]*4+[.2]*2)


def fixture(folder, learner):
    folder = Path(folder)
    x, y, weights = samples(learner)
    dataset = folder/'standing.npz'
    np.savez_compressed(dataset, observations=x, actions=y, weights=weights)
    (folder/'config.yaml').write_text('experiment:\n  simulation_only: true\n  model_route: cad_dynamics\n')
    xml = folder/'model.xml'; xml.write_text('test model')
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    dataset.with_suffix('.json').write_text(json.dumps(dict(dataset_sha256=digest,
        successful_data_only=True, runtime_reference_assistance=False,
        model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest())))
    return dict(settings(), dataset=str(dataset), dataset_sha256=digest), xml


class StandingRehearsalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rehearsal_learns_targets_and_only_changes_standing(self):
        learner = model(); learner.configure_estimator({'trainable': False})
        arrays = samples(learner)
        actor = learner.policy.mlp_extractor.policy_net
        with torch.no_grad():actor.standing_head.bias.add_(.03)
        before = {k: v.clone() for k, v in learner.policy.state_dict().items()}
        rehearsal = StandingRehearsal(learner.policy, arrays, dict(settings(), learning_rate=.0001), 3)
        initial = rehearsal.update()
        for _ in range(80):final = rehearsal.update()
        self.assertLess(final, initial/5)
        changed = []
        for key, old in before.items():
            if not torch.equal(old, learner.policy.state_dict()[key]):changed.append(key)
        self.assertTrue(changed)
        self.assertTrue(all('standing_net.' in k or 'standing_head.' in k for k in changed))
        self.assertTrue(all(p.grad is None for p in actor.velocity_estimator.parameters()))

    def test_loader_rejects_changed_dataset_model_and_configuration(self):
        learner = model()
        with tempfile.TemporaryDirectory() as folder:
            cfg, xml = fixture(folder, learner)
            arrays, provenance = load_standing_rehearsal(cfg, {}, xml)
            self.assertEqual(arrays[0].shape, (128, 180))
            self.assertEqual(provenance['samples'], 128)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                load_standing_rehearsal(cfg, {'task': {'changed': True}}, xml)
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_standing_rehearsal(dict(cfg, dataset_sha256='wrong'), {}, xml)
            xml.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'model'):
                load_standing_rehearsal(cfg, {}, xml)

    def test_loader_rejects_velocity_leak_rising_frames_and_bad_weights(self):
        learner = model()
        for kind in ('velocity', 'phase', 'weight', 'action'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                cfg, xml = fixture(folder, learner)
                x, y, weights = samples(learner)
                if kind == 'velocity':x[0, 28] = 1.
                if kind == 'phase':x[0, -1] = .5
                if kind == 'weight':weights[0] = -1.
                if kind == 'action':y[0, 0] = 2.
                p = Path(cfg['dataset'])
                np.savez_compressed(p, observations=x, actions=y, weights=weights)
                cfg['dataset_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
                r = json.loads(p.with_suffix('.json').read_text());r['dataset_sha256']=cfg['dataset_sha256']
                p.with_suffix('.json').write_text(json.dumps(r))
                with self.assertRaisesRegex(ValueError, 'arrays'):
                    load_standing_rehearsal(cfg, {}, xml)

    def test_settings_validation(self):
        learner = model(); arrays = samples(learner)
        for update in ({'updates_per_rollout': 0}, {'batch_size': True}, {'learning_rate': -1.},
                       {'max_grad_norm': float('nan')}, {'action_error_scales': [.1]*5},
                       {'reference_transitions': 0}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                StandingRehearsal(learner.policy, arrays, dict(settings(), **update), 2)

    def test_update_budget_scales_with_rollout_size(self):
        learner = model()
        rehearsal = StandingRehearsal(learner.policy, samples(learner),
                                     dict(settings(), updates_per_rollout=8), 2)
        rehearsal.update(2048)
        self.assertEqual(rehearsal.last_updates, 2)
        rehearsal.update(8192)
        self.assertEqual(rehearsal.last_updates, 8)

    def test_checkpoint_inference_has_no_dataset_dependency_and_resume_restores_optimizer(self):
        learner = model(); learner.configure_estimator({'trainable': False})
        with tempfile.TemporaryDirectory() as folder:
            cfg, xml = fixture(folder, learner)
            learner.configure_standing_rehearsal(cfg, {}, xml, 2)
            learner.learn(32)
            state = learner._standing_rehearsal.state_dict()
            checkpoint = Path(folder)/'policy.zip';learner.save(checkpoint)
            resumed = VelocityEstimationPPO.load(checkpoint, env=StudentEnv(), device='cpu')
            self.assertIsNone(getattr(resumed, '_standing_rehearsal', None))
            resumed.configure_estimator({'trainable': False})
            resumed.configure_standing_rehearsal(cfg, {}, xml, 9, restore_optimizer=True)
            self.assertEqual(resumed._standing_rehearsal.state_dict()['rng'], state['rng'])
            self.assertEqual(len(resumed._standing_rehearsal.optimizer.state), len(state['optimizer']['state']))
            resumed.learn(32)
            Path(cfg['dataset']).unlink()
            ordinary = PPO.load(checkpoint, device='cpu')
            obs, _ = StudentEnv().reset(seed=8)
            np.testing.assert_array_equal(learner.predict(obs, deterministic=True)[0],
                                          ordinary.predict(obs, deterministic=True)[0])


if __name__ == '__main__':unittest.main()
