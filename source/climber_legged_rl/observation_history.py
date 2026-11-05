"""Policy-rate history, sensor noise and episode-constant control delay.

Wrap the final task so intermediate observations emitted by inherited step/reset
methods never become extra history frames. Physics and success checks stay inside
the task and always use the uncorrupted simulation state.
"""
from collections import deque

import gymnasium as gym
import numpy as np


def history_frames(config):
    frames = config.get('observations', {}).get('history_frames', 1)
    if type(frames) is not int or not 1 <= frames <= 20:
        raise ValueError('history_frames must be an integer in [1,20]')
    return frames


class ObservationHistory(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.frames = history_frames(config)
        self.frame_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            np.tile(env.observation_space.low, self.frames),
            np.tile(env.observation_space.high, self.frames), dtype=np.float32)
        settings = config.get('robustness', {})
        self.noise = np.zeros(self.frame_dim, dtype=np.float32)
        blocks = {'angular_velocity': slice(0, 3), 'gravity': slice(3, 6),
                  'leg_position': slice(8, 12), 'joint_velocity': slice(12, 18)}
        # Values are physical-unit uniform half-widths before observation scaling.
        scales = {'angular_velocity': .25, 'gravity': 1.,
                  'leg_position': 1., 'joint_velocity': .05}
        noise = settings.get('observation_noise', {})
        if set(noise) - set(blocks):
            raise ValueError('Unknown observation_noise component')
        for name, width in noise.items():
            if not np.isfinite(width) or width < 0:
                raise ValueError('Observation noise must be finite and nonnegative')
            self.noise[blocks[name]] = width * scales[name]
        delay = settings.get('action_delay_steps', [0, 0])
        if (not isinstance(delay, list) or len(delay) != 2
                or any(type(n) is not int for n in delay) or not 0 <= delay[0] <= delay[1] <= 10):
            raise ValueError('action_delay_steps must be an inclusive integer range in [0,10]')
        self.delay_range = delay
        self.domain_ranges = {}
        for name in ('wheel_friction_multiplier', 'motor_strength'):
            limits = settings.get(name, [1., 1.])
            if (not isinstance(limits, list) or len(limits) != 2 or not np.isfinite(limits).all()
                    or not 0 < limits[0] <= limits[1]):
                raise ValueError(name + ' must be a finite positive range')
            self.domain_ranges[name] = limits
        # CAD wheels have multiple contacting meshes, not just the named proxy.
        self._wheel_geoms = sorted(env.wheel_geom_ids)
        self._wheel_group = np.array([env.wheel_ids.index(int(env.model.geom_bodyid[g]))
                                      for g in self._wheel_geoms])
        self._wheel_count = len(env.wheel_ids)
        self._nominal_friction = env.model.geom_friction[self._wheel_geoms].copy()
        self._nominal_ground_friction = env.model.geom_friction[env.ground_id].copy()
        self._history = deque(maxlen=self.frames)
        self._pending = deque()
        self.delay = 0
        self._last_command = np.zeros(env.action_space.shape, dtype=np.float32)
        self._rng = np.random.default_rng(0)

    def __getattr__(self, name):
        # Existing evaluation/rendering code accesses task geometry directly.
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def max_steps(self):
        return self.env.max_steps

    @max_steps.setter
    def max_steps(self, value):
        self.env.max_steps = value

    @property
    def pose_bank(self):
        return self.env.pose_bank

    @pose_bank.setter
    def pose_bank(self, value):
        self.env.pose_bank = value

    def _frame(self, observation):
        frame = observation.copy()
        if np.any(self.noise):
            frame += self._rng.uniform(-1., 1., self.frame_dim).astype(np.float32) * self.noise
        # The policy remembers the command it issued, including with actuator delay.
        frame[18:24] = self._last_command
        return np.clip(frame, -10., 10.).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        # Independent stream: adding sensing randomization cannot change reset/push
        # sampling inside the physical task. Explicit reset seeds remain repeatable.
        if seed is not None:
            self._rng = np.random.default_rng(np.random.SeedSequence([seed, 87431]))
        factors = self._rng.uniform(*self.domain_ranges['wheel_friction_multiplier'], size=self._wheel_count)
        # Contact coefficients are read during collision handling; mass/inertia and
        # closed-chain geometry are untouched. Restore nominal values every reset.
        self.env.model.geom_friction[self._wheel_geoms] = self._nominal_friction * factors[self._wheel_group, None]
        # MuJoCo takes the maximum at equal geom priority. Vary the ground too so
        # its nominal friction cannot mask a sampled reduction in wheel friction.
        self.env.model.geom_friction[self.env.ground_id] = self._nominal_ground_friction * factors.min()
        self.env.unwrapped.motor_strength = self._rng.uniform(*self.domain_ranges['motor_strength'], size=6)
        observation, info = self.env.reset(seed=seed, options=options)
        self.delay = int(self._rng.integers(self.delay_range[0], self.delay_range[1] + 1))
        self._last_command[:] = 0.
        self._pending.clear()
        self._pending.extend(np.zeros_like(self._last_command) for _ in range(self.delay))
        frame = self._frame(observation)
        self._history.clear()
        self._history.extend(frame.copy() for _ in range(self.frames))
        return np.concatenate(self._history), {**info, 'action_delay_steps': self.delay,
            'motor_strength': self.env.unwrapped.motor_strength.tolist(),
            'wheel_friction_multiplier': factors.tolist(),
            'ground_friction_multiplier': float(factors.min())}

    def step(self, action):
        command = np.asarray(action, dtype=np.float32)
        if command.shape != self.action_space.shape or not np.isfinite(command).all():
            raise ValueError('Expected six finite policy actions')
        self._last_command = np.clip(command, -1., 1.).copy()
        self._pending.append(self._last_command.copy())
        observation, reward, done, truncated, info = self.env.step(self._pending.popleft())
        self._history.append(self._frame(observation))
        return np.concatenate(self._history), reward, done, truncated, info


def wrap_observations(env, config):
    if config.get('training', {}).get('velocity_estimator', {}).get('enabled') and history_frames(config) <= 1:
        raise ValueError('Velocity estimation requires multiple 36-input history frames')
    if history_frames(config) == 1 and not config.get('robustness'):
        return env
    if env.observation_space.shape != (36,):
        env.close()
        raise ValueError('History/noise currently requires the 36-input combined task')
    env = ObservationHistory(env, config)
    if config.get('training', {}).get('velocity_estimator', {}).get('enabled'):
        from .velocity_estimation import VelocitySupervision
        env = VelocitySupervision(env)
    return env
