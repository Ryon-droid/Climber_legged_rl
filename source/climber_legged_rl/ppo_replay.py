"""Optional supervised rehearsal after PPO updates; never used during playback."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from stable_baselines3 import PPO

from .mujoco_model import ROOT, load_config


def load_demonstration_replay(settings, config, model_path):
    dataset = (ROOT / settings['dataset']).resolve()
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if digest != settings['dataset_sha256']:
        raise ValueError('Demonstration dataset hash mismatch')
    report_path = dataset.parent / 'imitation_report.json'
    report = json.loads(report_path.read_text())
    source = load_config(dataset.parent / 'config.yaml')
    if report['model_sha256'] != hashlib.sha256(Path(model_path).read_bytes()).hexdigest():
        raise ValueError('Demonstration physical model differs')
    for key in ('control', 'physics', 'spring'):
        if source[key] != config[key]:
            raise ValueError('Demonstration configuration differs: ' + key)
    for key in ('policy_control', 'dynamic_control', 'rearward_offsets_m'):
        if source['getup'][key] != config['getup'][key]:
            raise ValueError('Demonstration action or initial-state mapping differs: ' + key)
    if not settings.get('successful_expert_only'):
        raise ValueError('Replay requires the successful expert prefix')
    outcomes = [r for r in report['expert_outcomes'] if r['mode'] == 'folded']
    if not outcomes or not all(r['success'] for r in outcomes):
        raise ValueError('Expert prone trajectories must all succeed')
    count = int(report['stages'][0]['samples'])
    with np.load(dataset, allow_pickle=False) as data:
        x = np.asarray(data['observations'][:count], dtype=np.float32)
        y = np.asarray(data['actions'][:count], dtype=np.float32)
        w = np.asarray(data['weights'][:count], dtype=np.float64)
    if (count < 1 or x.shape != (count, 36) or y.shape != (count, 6)
            or w.shape != (count,) or not all(np.isfinite(a).all() for a in (x, y, w))
            or (w <= 0).any() or (np.abs(y) > 1.000001).any()):
        raise ValueError('Invalid expert replay arrays')
    for key in ('updates_per_rollout', 'batch_size'):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError('Invalid replay setting: ' + key)
    if not np.isfinite(settings['learning_rate']) or settings['learning_rate'] <= 0:
        raise ValueError('Invalid replay learning rate')
    provenance = dict(dataset=str(dataset.relative_to(ROOT)), dataset_sha256=digest,
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        expert_samples=count, successful_expert_only=True, runtime_reference_assistance=False)
    return (x, y, w / w.sum()), provenance


class DemonstrationReplayPPO(PPO):
    """Standard PPO followed by a few actor-only supervised replay updates."""
    def configure_replay(self, arrays, settings, seed, config_path=None, model_path=None):
        x, y, probabilities = arrays
        backtracks = settings.get('prone_gate_backtracks', 0)
        if type(backtracks) is not int or not 0 <= backtracks <= 4:
            raise ValueError('prone_gate_backtracks must be an integer in [0,4]')
        if backtracks and not settings.get('preserve_successful_prone', False):
            raise ValueError('Prone gate backtracking requires prone protection')
        if x.shape[1:] != self.observation_space.shape or y.shape[1:] != self.action_space.shape:
            raise ValueError('Replay observation/action dimensions differ')
        self._replay_x = torch.as_tensor(x, device=self.device)
        self._replay_y = torch.as_tensor(y, device=self.device)
        self._replay_probabilities = probabilities
        self._replay_rng = np.random.default_rng(seed)
        self._replay_settings = dict(settings)
        # The MLP actor and critic are separate. No value/std parameters enter this optimizer.
        actor = list(self.policy.mlp_extractor.policy_net.parameters()) + list(self.policy.action_net.parameters())
        self._replay_parameters = actor
        self._replay_optimizer = torch.optim.Adam(actor, lr=settings['learning_rate'])
        self._success_gate = bool(settings.get('preserve_successful_prone', False))
        if self._success_gate:
            if config_path is None or model_path is None:
                raise ValueError('Prone success gate requires the training configuration and physical model')
            self._gate_config = str(config_path)
            self._gate_model = str(model_path)
            if not self._passes_prone_gate():
                raise ValueError('Prone success gate requires a successful initial policy')

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            '_replay_x', '_replay_y', '_replay_probabilities', '_replay_rng',
            '_replay_settings', '_replay_parameters', '_replay_optimizer',
            '_success_gate', '_gate_config', '_gate_model']

    def _passes_prone_gate(self):
        # Training only. Runtime policies have no gate environment or controller.
        from .mujoco_env import make_mujoco_env
        env = make_mujoco_env(self._gate_model, self._gate_config)
        # Protect the original get-up skill independently of randomized push trials.
        # Push recovery is measured separately by the checkpoint-selection callback.
        env.push.enabled = False
        try:
            for pose in list(env.pose_bank):
                env.pose_bank = [pose]
                obs, _ = env.reset(seed=41000, options={'reset_mode': 'folded'})
                while True:
                    action, _ = self.predict(obs, deterministic=True)
                    obs, _, done, truncated, info = env.step(action)
                    if done or truncated:
                        break
                if not info['combined_success']:
                    return False
            return True
        finally:
            env.close()

    def _restore_actor(self, previous):
        with torch.no_grad():
            for parameter, value in zip(self._replay_parameters, previous):
                parameter.copy_(value)
                # Discard moments from rejected actor steps, retain critic learning.
                self.policy.optimizer.state.pop(parameter, None)
                self._replay_optimizer.state.pop(parameter, None)

    def _gate_actor_update(self, previous):
        if self._passes_prone_gate():
            return 1.0
        candidate = [p.detach().clone() for p in self._replay_parameters]
        for attempt in range(1, self._replay_settings.get('prone_gate_backtracks', 0) + 1):
            fraction = .5 ** attempt
            with torch.no_grad():
                for parameter, old, new in zip(self._replay_parameters, previous, candidate):
                    parameter.copy_(old + fraction * (new - old))
            if self._passes_prone_gate():
                # Moments describe the full rejected step, not the smaller accepted one.
                for parameter in self._replay_parameters:
                    self.policy.optimizer.state.pop(parameter, None)
                    self._replay_optimizer.state.pop(parameter, None)
                return fraction
        self._restore_actor(previous)
        return 0.0

    def replay_update(self):
        settings = self._replay_settings
        losses = []
        for _ in range(settings['updates_per_rollout']):
            indices = self._replay_rng.choice(len(self._replay_x), settings['batch_size'],
                                               p=self._replay_probabilities)
            mean = self.policy.get_distribution(self._replay_x[indices]).distribution.mean
            loss = (mean - self._replay_y[indices]).square().mean()
            self._replay_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._replay_parameters, self.max_grad_norm)
            self._replay_optimizer.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    def train(self):
        if not hasattr(self, '_replay_settings'):
            raise RuntimeError('Configure training-only demonstration replay before learning')
        previous = ([p.detach().clone() for p in self._replay_parameters]
                    if self._success_gate else None)
        super().train()
        self.logger.record('train/demonstration_mse', self.replay_update())
        self.logger.record('train/demonstration_updates', self._replay_settings['updates_per_rollout'])
        if self._success_gate:
            fraction = self._gate_actor_update(previous)
            accepted = fraction > 0
            self.logger.record('train/prone_actor_update_accepted', float(accepted))
            self.logger.record('train/prone_actor_update_fraction', fraction)
            print('PRONE_ACTOR_UPDATE ' + ('accepted fraction='+str(fraction) if accepted else 'rejected; restored previous successful actor'), flush=True)
