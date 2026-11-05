"""Two learned actor branches in one policy; freeze rising while refining balance."""
from copy import deepcopy

import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor


class PhaseSplitActor(nn.Module):
    """The existing observation ends with <=0.9 while rising and 1 after acquisition."""
    def __init__(self, actor_net, latent_dim, actions):
        super().__init__()
        self.rising_net = deepcopy(actor_net)
        self.rising_head = nn.Linear(latent_dim, actions)
        self.standing_net = deepcopy(actor_net)
        self.standing_head = nn.Linear(latent_dim, actions)
        self.rising_net.requires_grad_(False)
        self.rising_head.requires_grad_(False)

    def forward(self, features):
        rising = self.rising_head(self.rising_net(features))
        standing = self.standing_head(self.standing_net(features))
        return torch.where(features[:, -1:] > .95, standing, rising)


class PhaseSplitStandingPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        if (self.observation_space.shape != (36,) or self.action_space.shape != (6,)
                or not isinstance(self.features_extractor, FlattenExtractor)):
            raise ValueError('Phase split requires the unmodified 36-input, six-action get-up observation')
        super()._build_mlp_extractor()
        self.mlp_extractor.policy_net = PhaseSplitActor(
            self.mlp_extractor.policy_net, self.mlp_extractor.latent_dim_pi, 6)
        self.mlp_extractor.latent_dim_pi = 6

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def copy_parent_policy(source, target):
    """Both branches initially reproduce the old actor; the critic is unchanged."""
    if isinstance(source, PhaseSplitStandingPolicy) or not isinstance(target, PhaseSplitStandingPolicy):
        raise ValueError('Phase split conversion expects an ordinary parent actor')
    if (source.observation_space != target.observation_space or source.action_space != target.action_space
            or not isinstance(source.features_extractor, FlattenExtractor)):
        raise ValueError('Phase split conversion cannot change observation/action mappings')
    actor = target.mlp_extractor.policy_net
    for network in (actor.rising_net, actor.standing_net):
        network.load_state_dict(source.mlp_extractor.policy_net.state_dict())
    for head in (actor.rising_head, actor.standing_head):
        head.load_state_dict(source.action_net.state_dict())
    target.mlp_extractor.value_net.load_state_dict(source.mlp_extractor.value_net.state_dict())
    target.value_net.load_state_dict(source.value_net.state_dict())
    with torch.no_grad():
        target.log_std.copy_(source.log_std)
