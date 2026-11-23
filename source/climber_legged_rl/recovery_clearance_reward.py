"""Training-only clearance anticipation and bounded action-curvature cost.

Actions, observations, mechanics and termination stay exactly as supplied by the
base environment. Evaluation does not use this reward wrapper.
"""
import gymnasium as gym
import mujoco
import numpy as np

SETTINGS = dict(clearance_guard_m=.025, prediction_seconds=.15,
                maximum_closing_speed_m_s=.3, clearance_weight=2.,
                curvature_scales=[.04]*4+[.10]*2, curvature_weight=.10,
                maximum_curvature_cost=25., transient_leg_length_fraction=.5,
                transient_linear_tracking_fraction=.5)

def minimum_base_clearance(base):
    return min(float(mujoco.mj_geomDistance(base.model, base.data, base.ground_id,
                                          geom, 10., None))
               for geom in base.base_geom_ids)

class RecoveryClearanceReward(gym.Wrapper):
    def __init__(self, env, settings=None):
        super().__init__(env)
        self.settings = dict(SETTINGS if settings is None else settings)
        self.previous_action = self.previous_delta = self.previous_gap = None

    def reset(self, **kwargs):
        self.previous_action = self.previous_delta = self.previous_gap = None
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1., 1.)
        obs, reward, done, truncated, info = self.env.step(action)
        base = self.unwrapped
        gap = minimum_base_clearance(base)
        settings = self.settings
        speed = 0. if self.previous_gap is None else (gap-self.previous_gap)/base.dt
        closing = np.clip(speed, -settings['maximum_closing_speed_m_s'], 0.)
        predicted = gap + settings['prediction_seconds']*closing
        risk = np.clip((settings['clearance_guard_m']-predicted)/settings['clearance_guard_m'], 0., 3.)
        delta_action = None if self.previous_action is None else action-self.previous_action
        curvature = (0. if delta_action is None or self.previous_delta is None else
                     float(np.mean(((delta_action-self.previous_delta)/np.asarray(settings['curvature_scales']))**2)))
        acquired = bool(info.get('getup_acquired', True))
        terms = dict(recovery_clearance=-settings['clearance_weight']*float(risk**2) if acquired else 0.,
                     recovery_action_curvature=-settings['curvature_weight']*min(curvature, settings['maximum_curvature_cost']) if acquired else 0.)
        events = base.push.events
        recovering = bool(acquired and events and
            (base.steps <= events[-1]['end_step'] or info.get('balance_settled_seconds', 0.) < .5))
        delta = sum(terms.values())
        if recovering:
            for key, fraction in (('virtual_leg_length', settings['transient_leg_length_fraction']),
                                  ('linear_tracking', settings['transient_linear_tracking_fraction']),
                                  ('recovery_velocity', settings.get('transient_velocity_fraction', 1.))):
                original = info['reward_terms'].get(key, 0.)
                info['reward_terms'][key] = original*fraction
                delta += original*(fraction-1.)
        info['reward_terms'].update(terms)
        info.update(base_clearance_m=gap, predicted_base_clearance_m=float(predicted),
                    recovery_clearance_active=recovering)
        reward += delta
        base._total_reward += delta
        if done or truncated:
            info['episode_metrics']['reward'] = base._total_reward
        self.previous_gap = gap
        self.previous_action = action.copy()
        self.previous_delta = None if delta_action is None else delta_action.copy()
        return obs, float(reward), done, truncated, info
