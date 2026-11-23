"""One actor controls all six motors from prone reset through standing.

The time envelope only limits motor authority and selects low-level PD gains.
It never supplies joint trajectories, balancing feedback or replacement actions.
"""
from __future__ import annotations

import numpy as np

from .dynamic_getup import ClimberDynamicGetupEnv


class ClimberPolicyGetupEnv(ClimberDynamicGetupEnv):
    def __init__(self, model_path, config_path, render_mode=None):
        super().__init__(model_path, config_path, render_mode)
        self.envelope = self.getup['policy_control']
        if self.getup.get('support_control', {}).get('enabled'):
            raise ValueError('Full policy control cannot enable reference assistance')
        self.control = dict(self.control)
        self.rise_kp, self.rise_kd = self.control['leg_kp'], self.control['leg_kd']
        self.rise_wheel_kv = self.control['wheel_kv']
        positive = ('rated_torque_Nm', 'peak_window_seconds',
                    'standing_leg_kp', 'standing_leg_kd', 'standing_torque_limit_Nm')
        if any(not np.isfinite(self.envelope.get(k, 0)) or self.envelope[k] <= 0 for k in positive):
            raise ValueError('Invalid policy motor envelope')
        if not (self.envelope['standing_torque_limit_Nm'] <= self.envelope['rated_torque_Nm']
                <= self.control['leg_torque_limit_Nm']):
            raise ValueError('Standing, rated and peak torque must be ordered')
        if self.envelope['peak_window_seconds'] >= self.getup['deadline_seconds']:
            raise ValueError('Policy peak envelope must end before the get-up deadline')
        gain_seconds = self.envelope.get('rise_gain_seconds')
        if gain_seconds is not None and (not np.isfinite(gain_seconds) or not
                self.envelope['peak_window_seconds'] <= gain_seconds < self.getup['deadline_seconds']):
            raise ValueError('Invalid policy PD gain envelope')
        if 'phase_seconds' in self.envelope and (not np.isfinite(self.envelope['phase_seconds']) or self.envelope['phase_seconds']<=0):
            raise ValueError('Invalid policy phase observation scale')
        self._set_motor_envelope()

    def _phase(self):
        envelope = getattr(self, 'envelope', self.getup.get('policy_control', {}))
        if not getattr(self, 'acquired', True) and 'phase_seconds' in envelope:
            return min(.9, .9*self.steps*self.dt/envelope['phase_seconds'])
        return super()._phase()

    def _set_motor_envelope(self):
        rising = not self.acquired
        seconds = self.steps * self.dt
        high_gain = rising and seconds < self.envelope.get('rise_gain_seconds', float('inf'))
        self.control['leg_kp'] = self.rise_kp if high_gain else self.envelope['standing_leg_kp']
        self.control['leg_kd'] = self.rise_kd if high_gain else self.envelope['standing_leg_kd']
        self.control['wheel_kv'] = (self.rise_wheel_kv if rising else
                                   self.envelope.get('standing_wheel_kv', self.rise_wheel_kv))
        if rising and seconds < self.envelope['peak_window_seconds']:
            limit = self.config['control']['leg_torque_limit_Nm']
        else:
            limit = self.envelope['rated_torque_Nm'] if rising else self.envelope['standing_torque_limit_Nm']
        self.torque_limits[:4] = limit

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._set_motor_envelope()
        self.over_rated_seconds = np.zeros(4)
        self.peak_torque_seen = np.zeros(6)
        return observation, info

    def _observe_motor_torque(self, torque):
        self.over_rated_seconds += (np.abs(torque[:4]) > self.envelope['rated_torque_Nm'] + 1e-8) * self.model.opt.timestep
        self.peak_torque_seen = np.maximum(self.peak_torque_seen, np.abs(torque))

    def step(self, action):
        self._set_motor_envelope()
        observation, reward, done, truncated, info = super().step(action)
        info.update(control_mode='single_actor_getup_and_standing', policy_controls_all_motors=True,
                    controller_phase='policy_standing' if self.acquired else 'policy_getup',
                    over_rated_seconds=self.over_rated_seconds.tolist(),
                    peak_torque_Nm=self.peak_torque_seen.tolist())
        if done or truncated:
            info['getup_metrics'].update(guided_reference=False, control_mode=info['control_mode'],
                                         over_rated_seconds=info['over_rated_seconds'],
                                         peak_torque_Nm=info['peak_torque_Nm'])
        return observation, reward, done, truncated, info
