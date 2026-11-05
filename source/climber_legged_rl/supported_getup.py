"""Closed-chain rise reference, balance feedback and bounded learned residuals.

All motion is produced by the six motors in the floating-base simulation.
The peak window is a simulation setting; the motor's thermal duty is unmeasured.
"""
from __future__ import annotations

import numpy as np
from .dynamic_getup import ClimberDynamicGetupEnv


class ClimberSupportedGetupEnv(ClimberDynamicGetupEnv):
    def __init__(self, model_path, config_path, render_mode=None):
        super().__init__(model_path, config_path, render_mode)
        self.support = self.getup['support_control']
        self.control = dict(self.control)
        self.assist_kp = self.control['leg_kp']
        self.assist_kd = self.control['leg_kd']
        for name in ('reference_seconds', 'peak_window_seconds', 'rated_torque_Nm',
                     'leg_residual_rad', 'wheel_residual_rad_s'):
            if not np.isfinite(self.support[name]) or self.support[name] <= 0:
                raise ValueError('Invalid support control setting: ' + name)
        if self.support['rated_torque_Nm'] > self.control['leg_torque_limit_Nm']:
            raise ValueError('Rated torque must not exceed the peak XML limit')
        if self.support['peak_window_seconds'] >= self.getup['deadline_seconds']:
            raise ValueError('Peak assistance must end before the get-up deadline')
        if not 0 <= self.support['learned_assistance_scale'] <= 1:
            raise ValueError('Assistance residual scale must be in [0,1]')
        if not self.support['reference_seconds'] < self.support['handoff_seconds'] < self.getup['deadline_seconds']:
            raise ValueError('Handoff must follow the reference and precede the deadline')
        self._set_available_torque()

    def _assisting(self):
        return self.reset_mode == 'folded' and self.steps*self.dt < self.support['handoff_seconds']

    def _phase(self):
        if getattr(self, 'reset_mode', 'standing') == 'standing':
            return 1.
        settings = getattr(self, 'support', self.getup.get('support_control', {}))
        return min(1., self.steps*self.dt/settings.get('reference_seconds', .4))

    def _blend(self):
        if self.reset_mode == 'standing':
            return 1.
        return float(np.clip((self.steps*self.dt-self.support['feedback_start_seconds']) /
                             self.support['feedback_blend_seconds'], 0., 1.))

    def _leg_reference(self):
        if not hasattr(self, 'support'):
            return self.nominal_legs
        if not self._assisting():
            return self.nominal_legs
        phase = self._phase()
        smooth = phase*phase*(3.-2.*phase)
        if self.reset_mode == 'folded' and self.rise_waypoints is not None:
            reference = np.array([np.interp(smooth, self.rise_progress, self.rise_waypoints[:, j])
                                  for j in range(4)])
        else:
            reference = self.nominal_legs.copy()
        return reference + np.tile(self.support['leg_bias_rad'], 2)*self._blend()

    def _leg_targets(self, action):
        if not self._assisting():
            return np.clip(self.nominal_legs+action[:4]*self.support['standing_leg_action_scale_rad'],
                           self.leg_low,self.leg_high)
        return np.clip(self._leg_reference()+action[:4]*self.support['leg_residual_rad'],
                       self.leg_low, self.leg_high)

    def _set_available_torque(self):
        peak = (self.reset_mode == 'folded' and
                self.steps*self.dt < self.support['peak_window_seconds'])
        assisting = self._assisting()
        self.torque_limits[:4] = (self.control['leg_torque_limit_Nm'] if peak else
                                  self.support['rated_torque_Nm'] if assisting else
                                  self.support['standing_torque_limit_Nm'])
        self.control['leg_kp'] = self.assist_kp if assisting else self.support['standing_leg_kp']
        self.control['leg_kd'] = self.assist_kd if assisting else self.support['standing_leg_kd']

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._set_available_torque()
        self.over_rated_seconds = np.zeros(4)
        self.peak_torque_seen = np.zeros(6)
        return observation, info

    def _observe_motor_torque(self, torque):
        self.over_rated_seconds += (np.abs(torque[:4]) > self.support['rated_torque_Nm']+1e-8)*self.model.opt.timestep
        self.peak_torque_seen = np.maximum(self.peak_torque_seen, np.abs(torque))

    def assisted_action(self, action):
        """Return the assisted motor action without advancing physical state."""
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError('Expected six finite policy actions')
        action = np.clip(action, -1., 1.)
        assisting = self._assisting()
        angular, linear, _ = self._kinematics()
        displacement, heading = self._hold_errors()
        local = self._rotation().T @ np.r_[displacement, 0.]
        pitch, _ = self._attitude()
        gains = self.support['wheel_gains']
        late = float(np.clip((self.steps*self.dt-self.support['wheel_bias_start_seconds']) /
                             self.support['wheel_bias_blend_seconds'], 0., 1.))
        forward = (gains['pitch']*pitch+gains['pitch_rate']*angular[1]+
                   gains['position']*local[0]+gains['velocity']*linear[0]+
                   self.support['wheel_bias_rad_s']*late)
        forward = float(np.clip(forward, -10., 10.))*self._blend()
        yaw = np.clip(gains['heading']*heading+gains['yaw_rate']*angular[2], -3., 3.)*self._blend()
        executed = action.copy()
        if assisting:
            executed *= self.support['learned_assistance_scale']
            executed[4:] = (np.array([forward, -forward])+yaw+
                            executed[4:]*self.support['wheel_residual_rad_s'])/self.control['wheel_action_scale_rad_s']
        return executed

    def step(self, action):
        assisting = self._assisting()
        executed = self.assisted_action(action)
        self._set_available_torque()
        observation, reward, done, truncated, info = super().step(executed)
        info['control_mode'] = 'reference_assisted_rise_then_learned_standing'
        info['controller_phase'] = 'rise_assistance' if assisting else 'standing_policy'
        info['over_rated_seconds'] = self.over_rated_seconds.tolist()
        info['peak_torque_Nm'] = self.peak_torque_seen.tolist()
        if done or truncated:
            info['getup_metrics'].update(guided_reference=True, control_mode=info['control_mode'],
                                         over_rated_seconds=info['over_rated_seconds'],
                                         peak_torque_Nm=info['peak_torque_Nm'])
        return observation, reward, done, truncated, info
