"""Training-only anticipation of falls; no changes to actions or physics."""
import math

import gymnasium as gym
import numpy as np


EARLY_RECOVERY_SETTINGS = dict(prediction_seconds=.25, guard_tilt_deg=4.,
    risk_weight=2., maximum_normalized_risk=4., maximum_outward_rate_rad_s=2.,
    transient_leg_length_fraction=.25, transient_linear_tracking_fraction=.25)

BALANCED_RECOVERY_SETTINGS = dict(EARLY_RECOVERY_SETTINGS, prediction_seconds=.20,
    guard_tilt_deg=6., risk_weight=.25, transient_leg_length_fraction=.5,
    transient_linear_tracking_fraction=.5)


class EarlyRecoveryReward(gym.Wrapper):
    def __init__(self, env, settings=None):
        super().__init__(env)
        self.settings = dict(EARLY_RECOVERY_SETTINGS if settings is None else settings)
        self.previous_tilt = None

    def reset(self, **kwargs):
        self.previous_tilt = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        base = self.unwrapped
        tilt = float(info['tilt_rad'])
        if not np.isfinite(tilt):
            raise ValueError('Nonfinite tilt in recovery reward')
        outward = 0. if self.previous_tilt is None else max((tilt-self.previous_tilt)/base.dt, 0.)
        self.previous_tilt = tilt
        acquired = bool(info.get('getup_acquired', True))
        settings = self.settings
        predicted = tilt + settings['prediction_seconds'] * min(outward, settings['maximum_outward_rate_rad_s'])
        guard = math.radians(settings['guard_tilt_deg'])
        risk = min(max((predicted-guard)/guard, 0.), settings['maximum_normalized_risk'])
        penalty = settings['risk_weight'] * risk**2 if acquired else 0.
        events = base.push.events
        recovering = bool(acquired and events and
            (base.steps <= events[-1]['end_step'] or info.get('balance_settled_seconds', 0.) < .5))
        delta = -penalty
        if recovering:
            for key, fraction in (
                    ('virtual_leg_length', settings['transient_leg_length_fraction']),
                    ('linear_tracking', settings['transient_linear_tracking_fraction'])):
                original = info['reward_terms'].get(key, 0.)
                info['reward_terms'][key] = original * fraction
                delta += original * (fraction-1.)
        info['reward_terms']['early_recovery_risk'] = -penalty
        info.update(early_recovery_active=recovering, predicted_tilt_rad=predicted,
                    tilt_outward_rate_rad_s=outward)
        reward += delta
        base._total_reward += delta
        if done or truncated:
            info['episode_metrics']['reward'] = base._total_reward
        return obs, float(reward), done, truncated, info
