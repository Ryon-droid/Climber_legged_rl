"""Training-only replay of seeded standing episodes with actual force pulses."""
import gymnasium as gym
import numpy as np
import copy


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


class ExactPushResetReplay(gym.Wrapper):
    """Keep the discovered force law for the entire replayed episode.

    Matching seed, reset mode and force law reproduces the original pulse RNG.
    Recovery actions still come exclusively from the currently trained policy.
    """
    def __init__(self,env,cases,probability,seed):
        super().__init__(env)
        if not cases or not np.isfinite(probability) or not 0<probability<1:
            raise ValueError('Invalid exact push replay settings')
        self.cases=copy.deepcopy(cases);self.probability=probability
        self.original=copy.deepcopy(env.unwrapped.push.config)
        for case in self.cases:
            force=case.get('force_N',[])
            if (type(case.get('seed')) is not int or case['seed']<0 or len(force)!=2
                    or not np.isfinite(force).all() or not 0<force[0]<=force[1]
                    or force[1]>self.original['force_N'][1]):
                raise ValueError('Invalid exact replay case or stage force exceeded')
        self.rng=np.random.default_rng(seed)

    def reset(self,*,seed=None,options=None):
        push=self.env.unwrapped.push
        push.config=copy.deepcopy(self.original)
        replay=seed is None and options is None and self.rng.random()<self.probability
        if replay:
            case=self.cases[int(self.rng.integers(len(self.cases)))]
            seed=case['seed'];options={'reset_mode':'standing'}
            push.config.update(episode_probability=float(case.get('push',True)),force_N=list(case['force_N']))
            push.config.pop('force_bands',None)
        try:
            observation,info=self.env.reset(seed=seed,options=options)
        except BaseException:
            push.config=copy.deepcopy(self.original)
            raise
        return observation,{**info,'training_hard_push_reset':replay,
            'training_hard_push_seed':seed if replay else None,
            'training_exact_force_replay':replay}
