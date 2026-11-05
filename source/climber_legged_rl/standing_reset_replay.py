"""Training-only sampling of standing reset seeds that previously failed."""
import gymnasium as gym
import numpy as np


class StandingResetReplay(gym.Wrapper):
    def __init__(self, env, settings, seed):
        super().__init__(env)
        self.probability = float(settings['probability'])
        self.seeds = list(settings['seeds'])
        if not np.isfinite(self.probability) or not 0 < self.probability < 1:
            raise ValueError('Standing reset replay probability must be in (0,1)')
        if not self.seeds or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError('Standing reset replay requires nonnegative integer seeds')
        probabilities = settings.get('seed_probabilities')
        self.seed_probabilities = None
        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=float)
            if (probabilities.shape != (len(self.seeds),)
                    or not np.isfinite(probabilities).all()
                    or (probabilities <= 0).any()
                    or not np.isclose(probabilities.sum(), 1., rtol=0., atol=1e-10)):
                raise ValueError('Standing seed probabilities must be positive, sum to one, and match seeds')
            self.seed_probabilities = probabilities / probabilities.sum()
        self.state_scale_range = settings.get('initial_state_scale_range')
        self.full_state_probability = settings.get('full_state_probability', .25)
        if self.state_scale_range is not None:
            bounds = np.asarray(self.state_scale_range, dtype=float)
            if (bounds.shape != (2,) or not np.isfinite(bounds).all()
                    or not 0 < bounds[0] <= bounds[1] <= 1
                    or not np.isfinite(self.full_state_probability)
                    or not 0 <= self.full_state_probability <= 1):
                raise ValueError('Standing state scale must be within (0,1] and full-state probability within [0,1]')
            self.state_scale_range = bounds
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        # Explicit resets remain reproducible and authoritative. Only automatic
        # training resets draw from the hard set; evaluation never uses this wrapper.
        replay = seed is None and options is None and self.rng.random() < self.probability
        state_scale = 1.
        if replay:
            # Retain the original RNG sequence when weights are not configured.
            seed = int(self.rng.choice(self.seeds) if self.seed_probabilities is None
                       else self.rng.choice(self.seeds, p=self.seed_probabilities))
            options = {'reset_mode': 'standing'}
            if (self.state_scale_range is not None
                    and self.rng.random() >= self.full_state_probability):
                state_scale = float(self.rng.uniform(*self.state_scale_range))
        if state_scale != 1.:
            # Scale only initial tilt/velocity. Restore the task dictionary even
            # on reset failure; dynamics, rewards and success gates stay intact.
            task = self.env.unwrapped.config['task']
            names = ('reset_tilt_std_rad', 'reset_velocity_std')
            original = {name: task[name] for name in names}
            try:
                task.update({name: value * state_scale for name, value in original.items()})
                observation, info = self.env.reset(seed=seed, options=options)
            finally:
                task.update(original)
        else:
            observation, info = self.env.reset(seed=seed, options=options)
        return observation, {**info, 'training_hard_standing_reset': replay,
                             'training_hard_standing_seed': seed if replay else None,
                             'training_hard_standing_state_scale': state_scale}
