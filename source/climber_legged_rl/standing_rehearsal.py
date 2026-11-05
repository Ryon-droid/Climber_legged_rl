"""Training-only rehearsal of successful history-policy standing trajectories."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .mujoco_model import ROOT, load_config


def load_standing_rehearsal(settings, config, model_path):
    dataset = (ROOT / settings['dataset']).resolve()
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if digest != settings['dataset_sha256']:
        raise ValueError('Standing rehearsal dataset hash mismatch')
    report_path = dataset.with_suffix('.json')
    report = json.loads(report_path.read_text())
    if (report.get('dataset_sha256') != digest or not report.get('successful_data_only')
            or report.get('runtime_reference_assistance') is not False):
        raise ValueError('Standing rehearsal requires successful pure-policy trajectories')
    if report['model_sha256'] != hashlib.sha256(Path(model_path).read_bytes()).hexdigest():
        raise ValueError('Standing rehearsal physical model differs')
    source = load_config(dataset.parent / 'config.yaml')
    for key in ('control', 'physics', 'spring', 'task', 'getup', 'observations',
                'reward', 'push', 'balance_recovery'):
        if source.get(key) != config.get(key):
            raise ValueError('Standing rehearsal configuration differs: ' + key)
    with np.load(dataset, allow_pickle=False) as arrays:
        x = np.asarray(arrays['observations'], dtype=np.float32)
        y = np.asarray(arrays['actions'], dtype=np.float32)
        weights = np.asarray(arrays['weights'], dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] <= 36 or x.shape[1] % 36
            or y.shape != (len(x), 6) or weights.shape != (len(x),) or not len(x)
            or not all(np.isfinite(a).all() for a in (x, y, weights))
            or (weights <= 0).any() or (np.abs(y) > 1.000001).any()
            or (x[:, -1] <= .95).any()
            or np.any(x.reshape(len(x), -1, 36)[:, :, 28:30] != 0)):
        raise ValueError('Invalid standing rehearsal arrays or information boundary')
    return (x, y, weights / weights.sum()), dict(dataset=str(dataset),
        dataset_sha256=digest, report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        samples=len(x), runtime_reference_assistance=False)


class StandingRehearsal:
    def __init__(self, policy, arrays, settings, seed):
        x, y, probabilities = arrays
        for key in ('updates_per_rollout', 'batch_size'):
            if type(settings.get(key)) is not int or settings[key] <= 0:
                raise ValueError('Invalid standing rehearsal ' + key)
        reference = settings.get('reference_transitions', 8192)
        if type(reference) is not int or reference <= 0:
            raise ValueError('Invalid standing rehearsal reference_transitions')
        for key in ('learning_rate', 'max_grad_norm'):
            if not np.isfinite(settings.get(key, 0)) or settings[key] <= 0:
                raise ValueError('Invalid standing rehearsal ' + key)
        scales = np.asarray(settings['action_error_scales'], dtype=np.float32)
        if scales.shape != (6,) or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('Standing rehearsal requires six positive action error scales')
        if x.shape[1] != policy.observation_space.shape[0] - 3:
            raise ValueError('Standing rehearsal history dimensions differ')
        self.actor = policy.mlp_extractor.policy_net
        self.parameters = [p for name, p in self.actor.named_parameters()
                           if name.startswith(('standing_net.', 'standing_head.')) and p.requires_grad]
        if not self.parameters:
            raise ValueError('Standing rehearsal requires trainable standing parameters')
        self.x = torch.as_tensor(x, device=policy.device)
        self.y = torch.as_tensor(y, device=policy.device)
        self.scales = torch.as_tensor(scales, device=policy.device)
        self.probabilities = probabilities
        self.rng = np.random.default_rng(seed)
        self.settings = dict(settings)
        self.optimizer = torch.optim.Adam(self.parameters, lr=settings['learning_rate'])

    def update(self, transitions=None):
        losses = []
        updates = self.settings['updates_per_rollout']
        if transitions is not None:
            updates = max(1, round(updates * transitions / self.settings.get('reference_transitions', 8192)))
        self.last_updates = updates
        for _ in range(updates):
            indices = self.rng.choice(len(self.x), self.settings['batch_size'],
                                      p=self.probabilities)
            indices = torch.as_tensor(indices, device=self.x.device)
            predicted = self.actor(self.x[indices]).clamp(-1., 1.)
            loss = ((predicted - self.y[indices]) / self.scales).square().mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, self.settings['max_grad_norm'],
                                           error_if_nonfinite=True)
            self.optimizer.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    def state_dict(self):
        return dict(optimizer=self.optimizer.state_dict(), rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        self.rng.bit_generator.state = state['rng']
