"""History velocity supervision with actor/critic information separated explicitly."""
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from .history_policy import HistoryInputNetwork, HistoryPhaseActor, HistoryStandingPolicy
from .phase_split_policy import PhaseSplitStandingPolicy


class VelocitySupervision(gym.Wrapper):
    """Mask velocity in every actor frame, append a critic/training-only 3D label."""
    def __init__(self, env):
        super().__init__(env)
        self.actor_dimensions = env.observation_space.shape[0]
        if self.actor_dimensions <= 36 or self.actor_dimensions % 36:
            raise ValueError('Velocity estimation requires history of 36-input frames')
        self.observation_space = gym.spaces.Box(
            np.r_[env.observation_space.low, np.full(3, -np.inf, np.float32)],
            np.r_[env.observation_space.high, np.full(3, np.inf, np.float32)], dtype=np.float32)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def max_steps(self):
        return self.env.max_steps

    @max_steps.setter
    def max_steps(self, value):
        self.env.max_steps = value

    @property
    def pose_bank(self):
        return self.env.pose_bank

    @pose_bank.setter
    def pose_bank(self, value):
        self.env.pose_bank = value

    def _augment(self, observation):
        frames = observation.reshape(-1, 36).copy()
        frames[:, 28:30] = 0.
        _, velocity, _ = self.env.unwrapped._kinematics()
        return np.r_[frames.ravel(), velocity].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._augment(obs), reward, done, truncated, info


class EstimatingHistoryActor(HistoryPhaseActor):
    def __init__(self, network, latent_dim, actions, dimensions):
        super().__init__(network, latent_dim, actions)
        self.actor_dimensions = dimensions
        # IMU angular velocity/gravity, four positions, six velocities, six
        # preceding commands. No linear velocity, position, height or labels.
        indices = [i * 36 + j for i in range(dimensions // 36)
                   for j in [*range(6), *range(8, 24)]]
        self.register_buffer('sensor_indices', torch.tensor(indices, dtype=torch.long))
        self.velocity_estimator = nn.Sequential(nn.Linear(len(indices), 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 3))

    def estimate_velocity(self, observations):
        return self.velocity_estimator(observations[:, :self.actor_dimensions].index_select(1, self.sensor_indices))

    def forward(self, observations):
        features = observations[:, :self.actor_dimensions].clone()
        # Also mask inside the network: even an incorrectly constructed runtime
        # input cannot pass simulator velocity through historical frame slots.
        frames = features.reshape(-1, self.actor_dimensions // 36, 36)
        frames[:, :, 28:30] = 0.
        frames[:, -1, 28:30] = 2. * self.estimate_velocity(observations).detach()[:, :2]
        return super().forward(features)


class PrivilegedHistoryCritic(nn.Module):
    def __init__(self, network, dimensions):
        super().__init__()
        self.history_net = HistoryInputNetwork(network)
        self.actor_dimensions = dimensions

    def forward(self, observations):
        features = observations[:, :self.actor_dimensions].clone()
        features[:, -8:-6] = 2. * observations[:, -3:-1]
        return self.history_net(features)


class VelocityEstimatedPolicy(HistoryStandingPolicy):
    def __init__(self, *args, estimator_learning_rate=1e-3, **kwargs):
        self.estimator_learning_rate = estimator_learning_rate
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        dimensions = self.observation_space.shape[0] - 3
        if dimensions <= 36 or dimensions % 36 or self.action_space.shape != (6,):
            raise ValueError('Velocity policy requires actor history plus three privileged labels')
        ActorCriticPolicy._build_mlp_extractor(self)
        for network in (self.mlp_extractor.policy_net, self.mlp_extractor.value_net):
            first = network[0]
            network[0] = nn.Linear(dimensions, first.out_features)
        self.mlp_extractor.policy_net = EstimatingHistoryActor(self.mlp_extractor.policy_net,
            self.mlp_extractor.latent_dim_pi, 6, dimensions)
        self.mlp_extractor.policy_net.rising_net.requires_grad_(not self.freeze_rising_actor)
        self.mlp_extractor.policy_net.rising_head.requires_grad_(not self.freeze_rising_actor)
        self.mlp_extractor.value_net = PrivilegedHistoryCritic(self.mlp_extractor.value_net, dimensions)
        self.mlp_extractor.latent_dim_pi = 6

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.estimator_optimizer = torch.optim.Adam(
            self.mlp_extractor.policy_net.velocity_estimator.parameters(), lr=self.estimator_learning_rate)


def copy_estimation_parent(source, target):
    """Transfer learned control, not the untrained velocity estimator's accuracy."""
    actor = target.mlp_extractor.policy_net
    critic = target.mlp_extractor.value_net.history_net
    if isinstance(source, VelocityEstimatedPolicy):
        raise ValueError('Use normal checkpoint loading for an existing estimator policy')
    if isinstance(source, HistoryStandingPolicy):
        if source.observation_space.shape[0] != actor.actor_dimensions:
            raise ValueError('Estimator migration must retain the parent history length')
        parent = source.mlp_extractor.policy_net
        for name in ('rising_net', 'rising_head', 'standing_net', 'standing_head'):
            getattr(actor, name).load_state_dict(getattr(parent, name).state_dict())
        critic.load_state_dict(source.mlp_extractor.value_net.state_dict())
    elif source.observation_space.shape == (36,):
        if isinstance(source, PhaseSplitStandingPolicy):
            parent = source.mlp_extractor.policy_net
            for name in ('rising_net', 'rising_head', 'standing_head'):
                getattr(actor, name).load_state_dict(getattr(parent, name).state_dict())
            actor.standing_net.current_net.load_state_dict(parent.standing_net.state_dict())
        else:
            actor.rising_net.load_state_dict(source.mlp_extractor.policy_net.state_dict())
            actor.rising_head.load_state_dict(source.action_net.state_dict())
            actor.standing_net.current_net.load_state_dict(source.mlp_extractor.policy_net.state_dict())
            actor.standing_head.load_state_dict(source.action_net.state_dict())
        critic.current_net.load_state_dict(source.mlp_extractor.value_net.state_dict())
        with torch.no_grad():
            actor.standing_net.past_projection.weight.zero_()
            critic.past_projection.weight.zero_()
    else:
        raise ValueError('Estimator migration requires a 36-input or matching history parent')
    target.value_net.load_state_dict(source.value_net.state_dict())
    with torch.no_grad():
        target.log_std.copy_(source.log_std)


def estimator_update(policy, observations, epochs=4, batch_size=512):
    actor = policy.mlp_extractor.policy_net
    observations = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
    observations = observations.reshape(-1, policy.observation_space.shape[0])
    losses = []
    for _ in range(epochs):
        for indices in torch.randperm(len(observations), device=policy.device).split(batch_size):
            batch = observations[indices]
            error = actor.estimate_velocity(batch) - batch[:, -3:].detach()
            loss = error.square().mean()
            policy.estimator_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.velocity_estimator.parameters(), 1.)
            policy.estimator_optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.sqrt(np.mean(losses)))


class VelocityEstimationPPO(PPO):
    def __init__(self, *args, separate_actor_critic_gradient_clipping=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_gradient_clipping(separate_actor_critic_gradient_clipping)

    def configure_gradient_clipping(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('Separate actor/critic gradient clipping must be a boolean')
        if enabled and (not np.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError('Separate clipping requires a finite positive max_grad_norm')
        self.separate_actor_critic_gradient_clipping = enabled

    def _train_ppo(self):
        if not self.separate_actor_critic_gradient_clipping:
            return super().train()
        self.configure_gradient_clipping(True)
        if not isinstance(self.policy, VelocityEstimatedPolicy):
            raise ValueError('Separate clipping requires VelocityEstimatedPolicy')
        # This policy has parameter-free feature extraction and disjoint actor
        # and critic networks. Reject shared learned extractors rather than
        # silently assigning their gradients to the wrong group.
        if any(p.requires_grad for p in self.policy.features_extractor.parameters()):
            raise ValueError('Separate clipping does not support learned shared features')
        groups = {'actor': [], 'critic': []}
        for name, parameter in self.policy.named_parameters():
            if parameter.requires_grad:
                groups['critic' if 'value_net' in name else 'actor'].append(parameter)
        bound = self.max_grad_norm
        norms = {name: [] for name in groups}

        def clip_before_step(optimizer, args, kwargs):
            for name, parameters in groups.items():
                norm = nn.utils.clip_grad_norm_(parameters, bound, error_if_nonfinite=True)
                norms[name].append(float(norm))

        # Keep SB3's PPO loss, minibatching and KL early stopping unchanged.
        # Its global clip becomes a no-op; clip each network immediately before
        # Adam instead. The temporary hook never becomes checkpoint state.
        hook = self.policy.optimizer.register_step_pre_hook(clip_before_step)
        self.max_grad_norm = float('inf')
        try:
            super().train()
        finally:
            hook.remove()
            self.max_grad_norm = bound
        for name, values in norms.items():
            if values:
                self.logger.record(f'train/{name}_grad_norm', float(np.mean(values)))

    def configure_estimator(self, settings):
        if not isinstance(self.policy, VelocityEstimatedPolicy):
            raise ValueError('VelocityEstimationPPO requires VelocityEstimatedPolicy')
        self.estimator_trainable = settings.get('trainable', True)
        if type(self.estimator_trainable) is not bool:
            raise ValueError('Estimator trainable must be a boolean')
        self.policy.mlp_extractor.policy_net.velocity_estimator.requires_grad_(self.estimator_trainable)
        self.estimator_epochs = settings.get('epochs', 4)
        self.estimator_batch_size = settings.get('batch_size', 512)
        if any(type(n) is not int or n <= 0 for n in (self.estimator_epochs, self.estimator_batch_size)):
            raise ValueError('Estimator epochs and batch size must be positive integers')
        rate = settings.get('learning_rate', 1e-3)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('Estimator learning rate must be positive')
        for group in self.policy.estimator_optimizer.param_groups:
            group['lr'] = rate

    def train(self):
        if (getattr(self, 'standing_rehearsal_settings', {}).get('enabled')
                and getattr(self, '_standing_rehearsal', None) is None):
            raise RuntimeError('Configure standing rehearsal before resuming training')
        self._train_ppo()
        if getattr(self, '_standing_rehearsal', None) is not None:
            transitions = self.rollout_buffer.buffer_size * self.n_envs
            self.logger.record('train/standing_rehearsal_loss', self._standing_rehearsal.update(transitions))
            self.logger.record('train/standing_rehearsal_updates', self._standing_rehearsal.last_updates)
        if getattr(self, 'estimator_trainable', True):
            error = estimator_update(self.policy, self.rollout_buffer.observations,
                self.estimator_epochs, self.estimator_batch_size)
        else:
            # Keep the same learned sensor model in the actor. Freezing never
            # switches to privileged velocity, including during get-up.
            with torch.no_grad():
                observations = torch.as_tensor(self.rollout_buffer.observations,
                    dtype=torch.float32, device=self.device).reshape(-1, self.observation_space.shape[0])
                prediction = self.policy.mlp_extractor.policy_net.estimate_velocity(observations)
                error = float((prediction-observations[:, -3:]).square().mean().sqrt())
        self.logger.record('train/velocity_estimator_rmse_m_s', error)
        self.logger.record('train/velocity_estimator_trainable', int(getattr(self, 'estimator_trainable', True)))

    def configure_standing_rehearsal(self, settings, config, model_path, seed, restore_optimizer=False):
        enabled = settings.get('enabled', False)
        if type(enabled) is not bool:
            raise ValueError('Standing rehearsal enabled must be a boolean')
        self._standing_rehearsal = None
        self.standing_rehearsal_settings = dict(settings)
        if not enabled:
            return None
        from .standing_rehearsal import load_standing_rehearsal, StandingRehearsal
        arrays, provenance = load_standing_rehearsal(settings, config, model_path)
        self._standing_rehearsal = StandingRehearsal(self.policy, arrays, settings, seed)
        if restore_optimizer and hasattr(self, 'standing_rehearsal_state'):
            if self.standing_rehearsal_state['dataset_sha256'] != provenance['dataset_sha256']:
                raise ValueError('Resumed standing rehearsal dataset differs')
            self._standing_rehearsal.load_state_dict(self.standing_rehearsal_state['state'])
        return provenance

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['_standing_rehearsal']

    def save(self, path, exclude=None, include=None):
        # Keep standard PPO tensor keys so ordinary PPO.load remains a playback
        # interface. Store the second optimizer as CPU-only serialized metadata.
        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu(v) for v in value]
            if isinstance(value, tuple):
                return tuple(cpu(v) for v in value)
            return value
        self.velocity_optimizer_state = cpu(self.policy.estimator_optimizer.state_dict())
        if getattr(self, '_standing_rehearsal', None) is not None:
            self.standing_rehearsal_state = dict(
                dataset_sha256=self.standing_rehearsal_settings['dataset_sha256'],
                state=cpu(self._standing_rehearsal.state_dict()))
        return super().save(path, exclude=exclude, include=include)

    @classmethod
    def load(cls, *args, **kwargs):
        model = super().load(*args, **kwargs)
        if isinstance(model.policy, VelocityEstimatedPolicy) and hasattr(model, 'velocity_optimizer_state'):
            model.policy.estimator_optimizer.load_state_dict(model.velocity_optimizer_state)
        if isinstance(model.policy, VelocityEstimatedPolicy):
            model.policy.mlp_extractor.policy_net.velocity_estimator.requires_grad_(
                getattr(model, 'estimator_trainable', True))
        return model
