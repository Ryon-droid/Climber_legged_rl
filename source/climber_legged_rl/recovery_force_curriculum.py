"""Training-only force caps, including seeded replay, fixed per episode.

This wrapper belongs outside ExactPushResetReplay. Evaluation never uses it.
Caps advance by collected transitions, not by a claim of task mastery.
"""
import copy

import gymnasium as gym
import numpy as np

from .push_reset_replay import ExactPushResetReplay


FORCE_SCHEDULE = ((0, 12.), (12288, 15.), (36864, 18.))


class RecoveryForceCurriculum(gym.Wrapper):
    def __init__(self, env, schedule=FORCE_SCHEDULE):
        super().__init__(env)
        if not isinstance(env, ExactPushResetReplay):
            raise ValueError('Force curriculum must wrap exact replay directly')
        self.schedule = tuple(schedule)
        steps = [s for s, _ in self.schedule]
        forces = [f for _, f in self.schedule]
        maximum = env.original['force_N'][1]
        if (not steps or steps[0] != 0 or any(type(s) is not int or s < 0 for s in steps)
                or any(b <= a for a, b in zip(steps, steps[1:]))
                or not np.isfinite(forces).all() or any(f <= 0 or f > maximum for f in forces)
                or any(b < a for a, b in zip(forces, forces[1:]))):
            raise ValueError('Invalid force curriculum schedule')
        self.original = copy.deepcopy(env.original)
        self.cases = copy.deepcopy(env.cases)
        self.transitions = 0
        self.episode_cap = forces[0]

    def reset(self, **kwargs):
        self.episode_cap = next(f for s, f in reversed(self.schedule) if self.transitions >= s)
        # Set both sources BEFORE reset: the first pulse is sampled during reset.
        self.env.original = copy.deepcopy(self.original)
        self.env.cases = copy.deepcopy(self.cases)
        for settings in [self.env.original, *self.env.cases]:
            settings['force_N'] = [min(v, self.episode_cap) for v in settings['force_N']]
        for band in self.env.original.get('force_bands', []):
            band['force_N'] = [min(v, self.episode_cap) for v in band['force_N']]
        obs, info = self.env.reset(**kwargs)
        scaled = bool(info['training_exact_force_replay']
                      and self.episode_cap < self.original['force_N'][1])
        return obs, dict(info, training_force_cap_N=self.episode_cap,
                         training_force_scaled_replay=scaled,
                         training_exact_force_replay=info['training_exact_force_replay'] and not scaled)

    def step(self, action):
        result = self.env.step(action)
        self.transitions += 1
        obs, reward, done, truncated, info = result
        return obs, reward, done, truncated, dict(info, training_force_cap_N=self.episode_cap)
