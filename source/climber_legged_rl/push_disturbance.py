"""Seeded horizontal force pulses and post-push recovery accounting."""
from __future__ import annotations
import math
import numpy as np


class PushDisturbance:
    def __init__(self, config, dt):
        self.config = config
        self.dt = dt
        self.enabled = config.get('enabled', False)
        if self.enabled:
            for key in ('force_N', 'duration_seconds', 'interval_seconds'):
                values = config.get(key, [])
                if len(values) != 2 or not np.isfinite(values).all() or not 0 < values[0] <= values[1]:
                    raise ValueError('Invalid push range: ' + key)
            for key in ('first_push_seconds', 'recovery_seconds', 'settle_seconds',
                        'max_transient_displacement_m', 'max_transient_heading_rad', 'recovery_tilt_rad'):
                if not np.isfinite(config.get(key, 0)) or config.get(key, 0) <= 0:
                    raise ValueError('Invalid push parameter: ' + key)
            if not 0 <= config.get('episode_probability', -1) <= 1:
                raise ValueError('Push episode_probability must be in [0,1]')
            if config['interval_seconds'][0] <= config['duration_seconds'][1] + config['recovery_seconds']:
                raise ValueError('Push intervals must leave time for recovery')
            if config['settle_seconds'] > config['recovery_seconds']:
                raise ValueError('Push settle time exceeds recovery deadline')
        self.reset(np.random.default_rng(0))

    def reset(self, rng):
        self.rng = rng
        self.events = []
        self.active = None
        self.episode_enabled = self.enabled and rng.random() < self.config['episode_probability']
        self.next_step = math.ceil(self.config['first_push_seconds']/self.dt) if self.episode_enabled else None

    def force(self, step, max_steps):
        force = np.zeros(3)
        if not self.episode_enabled:
            return force
        if self.next_step is not None and step >= self.next_step:
            duration_steps = math.ceil(self.rng.uniform(*self.config['duration_seconds'])/self.dt)
            recovery_steps = math.ceil(self.config['recovery_seconds']/self.dt)
            # Never start a pulse whose recovery window lies outside the episode.
            if step + duration_steps + recovery_steps > max_steps:
                self.next_step = None
                return force
            magnitude = float(self.rng.uniform(*self.config['force_N']))
            angle = self.rng.uniform(-np.pi, np.pi)
            force[:2] = magnitude*np.array([np.cos(angle), np.sin(angle)])
            event = dict(start_step=step, end_step=step+duration_steps,
                         deadline_step=step+duration_steps+recovery_steps, force_world_N=force.tolist(),
                         duration_seconds=duration_steps*self.dt, impulse_Ns=magnitude*duration_steps*self.dt,
                         recovered=False, recovery_seconds=None, settled_steps=0)
            self.events.append(event)
            self.active = event
            self.next_step = step + math.ceil(self.rng.uniform(*self.config['interval_seconds'])/self.dt)
        if self.active is not None and step < self.active['end_step']:
            force[:] = self.active['force_world_N']
        return force

    def observe(self, step, displacement, heading, speed, yaw_rate, tilt, task):
        event = self.active
        if event is None or event['recovered'] or step <= event['end_step'] or step > event['deadline_step']:
            return
        good = (displacement <= task['stationary_max_displacement_m']
                and abs(heading) <= task['stationary_max_heading_rad']
                and speed <= task['linear_tracking_sigma_m_s']
                and abs(yaw_rate) <= task['yaw_tracking_sigma_rad_s']
                and tilt <= self.config['recovery_tilt_rad'])
        event['settled_steps'] = event['settled_steps']+1 if good else 0
        if event['settled_steps'] >= math.ceil(self.config['settle_seconds']/self.dt):
            event['recovered'] = True
            event['recovery_seconds'] = (step-event['end_step'])*self.dt

    def metrics(self):
        times = [event['recovery_seconds'] for event in self.events if event['recovered']]
        return dict(push_count=len(self.events), recovered_push_count=len(times),
                    mean_recovery_seconds=float(np.mean(times)) if times else None,
                    max_recovery_seconds=max(times) if times else None,
                    push_impulse_Ns=sum(event['impulse_Ns'] for event in self.events),
                    push_events=[dict(event) for event in self.events])

    def success(self, reached_horizon, stationary_success, max_displacement, max_heading,
                displacement, heading, speed, yaw_rate, tilt, task):
        if not self.events:
            return stationary_success
        return bool(reached_horizon and all(event['recovered'] for event in self.events)
                    and max_displacement <= self.config['max_transient_displacement_m']
                    and max_heading <= self.config['max_transient_heading_rad']
                    and displacement <= task['stationary_max_displacement_m']
                    and abs(heading) <= task['stationary_max_heading_rad']
                    and speed <= task['linear_tracking_sigma_m_s']
                    and abs(yaw_rate) <= task['yaw_tracking_sigma_rad_s']
                    and tilt <= self.config['recovery_tilt_rad'])
