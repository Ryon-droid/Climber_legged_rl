"""Frozen current-frame rising actor and trainable history standing actor."""
from copy import deepcopy

import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor

from .phase_split_policy import PhaseSplitStandingPolicy


class HistoryInputNetwork(nn.Module):
    """Keep the parent's current-frame GEMM and add a zero-initialized past term.

    A single widened GEMM can change reduction rounding even with zero columns.
    This decomposition preserves the parent's calculation at initialization.
    """
    def __init__(self, network):
        super().__init__()
        first = network[0]
        self.past_projection = nn.Linear(first.in_features - 36, first.out_features, bias=False)
        nn.init.zeros_(self.past_projection.weight)
        network[0] = nn.Linear(36, first.out_features)
        self.current_net = network

    def __getitem__(self, index):
        return self.current_net[index]

    def forward(self, features):
        latent = self.current_net[0](features[:, -36:]) + self.past_projection(features[:, :-36])
        for layer in list(self.current_net)[1:]:
            latent = layer(latent)
        return latent


class HistoryPhaseActor(nn.Module):
    def __init__(self, actor_net, latent_dim, actions):
        super().__init__()
        self.standing_net = HistoryInputNetwork(actor_net)
        self.standing_head = nn.Linear(latent_dim, actions)
        self.rising_net = deepcopy(self.standing_net.current_net)
        self.rising_head = nn.Linear(latent_dim, actions)
        self.rising_net.requires_grad_(False)
        self.rising_head.requires_grad_(False)

    def forward(self, features):
        current = features[:, -36:]
        rising = self.rising_head(self.rising_net(current))
        standing = self.standing_head(self.standing_net(features))
        return torch.where(current[:, -1:] > .95, standing, rising)


class HistoryStandingPolicy(ActorCriticPolicy):
    def __init__(self, *args, freeze_rising_actor=True, **kwargs):
        self.freeze_rising_actor = freeze_rising_actor
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        dimensions = self.observation_space.shape
        if (len(dimensions) != 1 or dimensions[0] <= 36 or dimensions[0] % 36
                or self.action_space.shape != (6,) or not isinstance(self.features_extractor, FlattenExtractor)):
            raise ValueError('History policy requires stacked 36-input frames and six actions')
        super()._build_mlp_extractor()
        self.mlp_extractor.policy_net = HistoryPhaseActor(
            self.mlp_extractor.policy_net, self.mlp_extractor.latent_dim_pi, 6)
        self.mlp_extractor.policy_net.rising_net.requires_grad_(not self.freeze_rising_actor)
        self.mlp_extractor.policy_net.rising_head.requires_grad_(not self.freeze_rising_actor)
        self.mlp_extractor.value_net = HistoryInputNetwork(self.mlp_extractor.value_net)
        self.mlp_extractor.latent_dim_pi = 6

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def _copy_latest(source, target):
    target.current_net.load_state_dict(source.state_dict())
    with torch.no_grad():
        target.past_projection.weight.zero_()


def copy_history_parent(source, target):
    """Preserve both deterministic actions and critic; ignore older frames initially."""
    if (source.observation_space.shape != (36,) or not isinstance(target, HistoryStandingPolicy)
            or source.action_space != target.action_space
            or not isinstance(source.features_extractor, FlattenExtractor)):
        raise ValueError('History migration requires a 36-input parent with matching actions')
    actor = target.mlp_extractor.policy_net
    if isinstance(source, PhaseSplitStandingPolicy):
        parent = source.mlp_extractor.policy_net
        actor.rising_net.load_state_dict(parent.rising_net.state_dict())
        actor.rising_head.load_state_dict(parent.rising_head.state_dict())
        _copy_latest(parent.standing_net, actor.standing_net)
        actor.standing_head.load_state_dict(parent.standing_head.state_dict())
    else:
        actor.rising_net.load_state_dict(source.mlp_extractor.policy_net.state_dict())
        actor.rising_head.load_state_dict(source.action_net.state_dict())
        _copy_latest(source.mlp_extractor.policy_net, actor.standing_net)
        actor.standing_head.load_state_dict(source.action_net.state_dict())
    _copy_latest(source.mlp_extractor.value_net, target.mlp_extractor.value_net)
    target.value_net.load_state_dict(source.value_net.state_dict())
    with torch.no_grad():
        target.log_std.copy_(source.log_std)
