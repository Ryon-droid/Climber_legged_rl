"""Commanded low-speed motion and stop-hold, starting from a standing reset."""
import numpy as np

from .mujoco_env import ClimberMujocoEnv
from .mujoco_model import load_config


class ClimberVelocityEnv(ClimberMujocoEnv):
    # Preserve the standing policy's 30-input layout, including body velocity.
    extra_observation_dimensions = 6

    def __init__(self, model_path, config_path, render_mode=None):
        layout = load_config(config_path).get('task', {}).get('velocity_observation_layout')
        if layout not in (None, 'getup_standing'):
            raise ValueError('Unknown velocity observation layout: ' + str(layout))
        self.getup_standing_layout = layout == 'getup_standing'
        if self.getup_standing_layout:
            # Keep the 36-input single-actor layout used by the get-up policy:
            # stationary feedback plus the standing branch's get-up context.
            self.extra_observation_dimensions = 12
        super().__init__(model_path, config_path, render_mode)
        if self.stationary_hold or self.config.get('getup', {}).get('enabled'):
            raise ValueError('Velocity task requires standing resets with stationary_hold=false')
        self.schedule = self.config['task']['velocity_commands']
        for key in ('resample_seconds', 'initial_stand_seconds', 'ramp_seconds'):
            if not np.isfinite(self.schedule[key]) or self.schedule[key] <= 0:
                raise ValueError('Invalid velocity command duration: ' + key)
        if self.schedule['ramp_seconds'] >= self.schedule['resample_seconds']:
            raise ValueError('Command ramp must be shorter than resampling interval')
        if not 0 <= self.schedule['idle_probability'] <= 1:
            raise ValueError('Invalid idle probability')
        if not 0 <= self.schedule.get('axis_probability',0.) <= 1:
            raise ValueError('Invalid single-axis command probability')
        self.idle_hold = True
        self.training_curriculum = False
        self.curriculum_steps = 0

    def enable_training_curriculum(self):
        """Enable training-only motion bootstrap; evaluation never calls this."""
        self.training_curriculum = True

    def _bootstrap_scale(self):
        decay = float(self.config['task'].get('velocity_bootstrap', {}).get('decay_env_steps', 0))
        if decay <= 0:
            return 1.
        return max(0., 1. - self.curriculum_steps / decay)

    def _bootstrap_action(self, action):
        bootstrap = self.config['task'].get('velocity_bootstrap', {})
        if not self.training_curriculum or not bootstrap.get('enabled', False) or self.idle_hold:
            return action
        scale = self._bootstrap_scale()
        seconds = float(bootstrap.get('seconds', 1.2))
        age = self.steps * self.dt - self.command_changed_seconds
        if scale <= 0 or age < 0 or age > seconds:
            return action
        fade = min(1., age / 0.15) * min(1., max(0., (seconds - age) / 0.3))
        gain = float(bootstrap.get('action_per_m_s', 2.5)) * fade * scale
        applied = action.copy()
        applied[4:] -= gain * self.command_target[0]
        return np.clip(applied, -1., 1.)

    def _extra_observation(self):
        _, linear, _ = self._kinematics()
        if getattr(self, 'idle_hold', True):
            displacement, heading = self._hold_errors()
            local = self._rotation().T @ np.r_[displacement, 0.]
            feedback = [* (local[:2] * 5.), np.sin(heading), np.cos(heading)-1.]
        else:
            feedback = [0., 0., 0., 0.]
        extra = np.r_[feedback, linear[:2] * 2.]
        if self.getup_standing_layout:
            # Standing values of the get-up branch: zero joint reference offsets,
            # hip-height error and a phase of one.
            extra = np.r_[extra, np.zeros(4),
                          (self.data.xpos[self.hip_ids, 2].mean() - self.target_hip_height) * 5., 1.]
        return extra

    def _new_command(self, command):
        self.ramp_start = self.commands.copy()
        self.command_target = np.asarray(command, dtype=float).copy()
        self.command_changed_seconds = self.steps * self.dt

    def _update_commands(self):
        seconds = self.steps * self.dt
        if seconds + 1e-9 >= self.next_command_seconds:
            if self.fixed_command is not None:
                target = self.fixed_command if not self.fixed_started else np.zeros(2)
                self.fixed_started = True
                self.next_command_seconds = self.stop_after_seconds
                if seconds + 1e-9 >= self.stop_after_seconds:
                    target = np.zeros(2)
                    self.next_command_seconds = float('inf')
            else:
                task = self.config['task']
                target = np.zeros(2) if self.np_random.random() < self.schedule['idle_probability'] else np.array([
                    self.np_random.uniform(*task['command_forward_m_s']),
                    self.np_random.uniform(*task['command_yaw_rad_s'])])
                axis_probability=self.schedule.get('axis_probability',0.)
                if np.any(target) and axis_probability and self.np_random.random()<axis_probability:
                    target[int(self.np_random.integers(2))]=0.
                self.next_command_seconds = seconds + self.schedule['resample_seconds']
            self._new_command(target)
        fraction = np.clip((seconds-self.command_changed_seconds)/self.schedule['ramp_seconds'], 0., 1.)
        self.commands[:] = self.ramp_start + fraction * (self.command_target-self.ramp_start)
        idle = bool(np.max(np.abs(self.commands)) < 1e-8 and np.max(np.abs(self.command_target)) < 1e-8)
        if idle and not self.idle_hold:
            self._set_hold_reference()
        self.idle_hold = idle

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        options = options or {}
        command = options.get('velocity_command')
        self.fixed_command = None if command is None else np.asarray(command, dtype=float)
        if self.fixed_command is not None:
            if self.fixed_command.shape != (2,) or not np.isfinite(self.fixed_command).all():
                raise ValueError('Expected finite forward/yaw command pair')
            for value, key in zip(self.fixed_command, ('command_forward_m_s', 'command_yaw_rad_s')):
                if not self.config['task'][key][0] <= value <= self.config['task'][key][1]:
                    raise ValueError('Evaluation command exceeds training range')
        self.stop_after_seconds = float(options.get('stop_after_seconds', float('inf')))
        if self.stop_after_seconds <= self.schedule['initial_stand_seconds']:
            raise ValueError('Stop time must follow initial stand phase')
        self.fixed_started = False
        self.commands[:] = 0.
        self.ramp_start = self.commands.copy()
        self.command_target = self.commands.copy()
        self.command_changed_seconds = 0.
        self.next_command_seconds = self.schedule['initial_stand_seconds']
        self.idle_hold = True
        self._set_hold_reference()
        return self._observation(), dict(simulation_only=True, reset_mode='standing', task='velocity_tracking')

    def step(self, action):
        executed_command = self.commands.copy()
        executed_idle = self.idle_hold
        changed = self.command_changed_seconds
        _, reward, done, truncated, info = super().step(self._bootstrap_action(action))
        angular, linear, _ = self._kinematics()
        task, weights = self.config['task'], self.config['reward']
        terms = info['reward_terms']
        previous_sum = sum(terms.values())
        terms['linear_tracking'] = float(np.exp(-((linear[0]-executed_command[0])**2+linear[1]**2)
            / task['linear_tracking_sigma_m_s']**2)) * weights['linear_tracking']
        terms['yaw_tracking'] = float(np.exp(-((angular[2]-executed_command[1])
            / task['yaw_tracking_sigma_rad_s'])**2)) * weights['yaw_tracking']
        if executed_idle:
            displacement, heading = self._hold_errors()
            terms['position_hold'] = float(np.exp(-np.sum(displacement**2)/task['position_sigma_m']**2)) * weights['position_hold']
            terms['heading_hold'] = float(np.exp(-(heading/task['heading_sigma_rad'])**2)) * weights['heading_hold']
        adjustment = sum(terms.values()) - previous_sum
        reward += adjustment
        self._total_reward += adjustment
        info.update(command_forward_m_s=float(executed_command[0]), command_yaw_rad_s=float(executed_command[1]),
            lateral_velocity_m_s=float(linear[1]), idle_hold=executed_idle,
            command_age_seconds=self.steps*self.dt-changed, task='velocity_tracking')
        if done or truncated:
            info['episode_metrics']['reward'] = self._total_reward
        self._update_commands()
        self.curriculum_steps += 1
        return self._observation(), reward, done, truncated, info
