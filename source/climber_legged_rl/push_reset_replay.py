"""Training-only replay of seeded standing episodes with actual force pulses."""
import gymnasium as gym
import numpy as np


class PushResetReplay(gym.Wrapper):
    def __init__(self, env, settings, seed):
        super().__init__(env)
        self.probability = settings['probability']
        self.seeds = list(settings['seeds'])
        if not np.isfinite(self.probability) or not 0 < self.probability < 1:
            raise ValueError('Hard push reset probability must be in (0,1)')
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(type(s) is not int or s < 0 for s in self.seeds)):
            raise ValueError('Hard push resets require distinct nonnegative integer seeds')
        if not env.unwrapped.push.enabled:
            raise ValueError('Hard push resets require enabled pushes')
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        replay = seed is None and options is None and self.rng.random() < self.probability
        if replay:
            seed = int(self.rng.choice(self.seeds))
            options = {'reset_mode': 'standing'}
            push = self.env.unwrapped.push
            original = push.config
            # Only choose this episode's enabled flag. Restore the original
            # distribution immediately; the pulse force law and RNG are intact.
            push.config = dict(original, episode_probability=1.)
            try:
                observation, info = self.env.reset(seed=seed, options=options)
            finally:
                push.config = original
        else:
            observation, info = self.env.reset(seed=seed, options=options)
        return observation, {**info, 'training_hard_push_reset': replay,
            'training_hard_push_seed': seed if replay else None}
