"""Balance-first scoring beside, rather than replacing, native strict scoring.

This wrapper never changes the plant, reset, force generator or termination.
Native recovery events remain available for an apples-to-apples strict report.
"""
from dataclasses import asdict, dataclass
import copy
import math

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class BalanceStage:
    name: str
    position_m: float | None
    heading_rad: float | None
    transient_position_m: float | None
    transient_heading_rad: float | None
    speed_m_s: float
    yaw_rate_rad_s: float
    tilt_rad: float
    tilt_rate_rad_s: float
    hold_reward_fraction: float

    def stable(self, info):
        values = [info[k] for k in ('planar_displacement_m', 'heading_error_rad',
                  'planar_speed_m_s', 'yaw_rate_rad_s', 'tilt_rad', 'tilt_rate_rad_s')]
        return bool(np.isfinite(values).all()
            and info['wheel_contacts'] == 2
            and info['termination_reason'] is None
            and (self.position_m is None or info['planar_displacement_m'] <= self.position_m)
            and (self.heading_rad is None or abs(info['heading_error_rad']) <= self.heading_rad)
            and info['planar_speed_m_s'] <= self.speed_m_s
            and abs(info['yaw_rate_rad_s']) <= self.yaw_rate_rad_s
            and info['tilt_rad'] <= self.tilt_rad
            and info['tilt_rate_rad_s'] <= self.tilt_rate_rad_s)

    def within_transient(self, position, heading):
        return ((self.transient_position_m is None or position <= self.transient_position_m)
                and (self.transient_heading_rad is None or heading <= self.transient_heading_rad))


STAGES = {
    'balance': BalanceStage('balance', None, None, None, None, .08, .15,
                            math.radians(8), .25, 0.),
    'return': BalanceStage('return', .15, math.radians(15), .30, math.radians(45),
                           .06, .10, math.radians(6), .20, .25),
    # This stage delegates success/recovery/reward to the native strict task.
    'strict': BalanceStage('strict', .05, math.radians(5), .20, math.radians(20),
                           .05, .05, math.radians(5), .20, 1.),
}


class RecoveryAccounting:
    """Independent counters; never modify native strict push events."""
    def __init__(self, stage, dt, settle_seconds):
        self.stage = stage
        self.required = math.ceil(settle_seconds / dt)
        self.dt = dt
        self.events = []
        self.settled_steps = 0

    def observe(self, step, events, info, acquired=True):
        for event in events[len(self.events):]:
            current = copy.deepcopy(event)
            current.update(recovered=False, recovery_seconds=None, settled_steps=0)
            self.events.append(current)
        good = acquired and self.stage.stable(info)
        self.settled_steps = self.settled_steps + 1 if good else 0
        for event in self.events:
            if event['recovered'] or not event['end_step'] < step <= event['deadline_step']:
                continue
            event['settled_steps'] = event['settled_steps'] + 1 if good else 0
            if event['settled_steps'] >= self.required:
                event['recovered'] = True
                event['recovery_seconds'] = (step - event['end_step']) * self.dt

    def success(self, reached, max_position, max_heading):
        return bool(reached and self.settled_steps >= self.required
                    and all(e['recovered'] for e in self.events)
                    and self.stage.within_transient(max_position, max_heading))

    def metrics(self):
        times = [e['recovery_seconds'] for e in self.events if e['recovered']]
        return dict(push_count=len(self.events), recovered_push_count=len(times),
                    mean_recovery_seconds=float(np.mean(times)) if times else None,
                    max_recovery_seconds=max(times) if times else None,
                    push_impulse_Ns=sum(e['impulse_Ns'] for e in self.events),
                    push_events=copy.deepcopy(self.events))


class PushBalanceCurriculum(gym.Wrapper):
    def __init__(self, env, stage='balance', sustained_reward=0.):
        super().__init__(env)
        self.stage = STAGES[stage]
        if not np.isfinite(sustained_reward) or sustained_reward < 0:
            raise ValueError('Sustained stability reward must be finite and nonnegative')
        self.sustained_reward = float(sustained_reward)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = self.unwrapped
        self.accounting = RecoveryAccounting(self.stage, base.dt, base.push.config['settle_seconds'])
        self.max_position = self.max_heading = 0.
        return obs, dict(info, balance_curriculum=asdict(self.stage))

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        base = self.unwrapped
        info['tilt_rate_rad_s'] = float(np.linalg.norm(base._kinematics()[0][:2]))
        acquired = bool(info.get('getup_acquired', True))
        if acquired:
            self.max_position = max(self.max_position, info['planar_displacement_m'])
            self.max_heading = max(self.max_heading, abs(info['heading_error_rad']))
        self.accounting.observe(base.steps, base.push.events, info, acquired)
        if acquired and self.stage.name != 'strict':
            # Remove the original-position/original-heading incentive in stage 1.
            # Keep uprightness, leg support, low speed, effort and fall penalties.
            delta = 0.
            for key in ('position_hold', 'heading_hold', 'recovery_position', 'recovery_heading'):
                old = info['reward_terms'].get(key, 0.)
                new = old * self.stage.hold_reward_fraction
                info['reward_terms'][key] = new
                delta += new - old
            reward += delta
            base._total_reward += delta
        # Training-only incentive continues after the first recovered event.
        # Any new motion, support loss or termination resets the timer to zero.
        if acquired and self.sustained_reward:
            bonus = self.sustained_reward * min(self.accounting.settled_steps * base.dt / 2., 1.)
            info['reward_terms']['sustained_balance'] = bonus
            reward += bonus
            base._total_reward += bonus
        info['balance_curriculum_stage'] = self.stage.name
        info['balance_settled_seconds'] = self.accounting.settled_steps * base.dt
        if done or truncated:
            info['strict_success'] = bool(info['combined_success'])
            info['strict_push_metrics'] = copy.deepcopy(info['push_metrics'])
            if self.stage.name != 'strict':
                success = self.accounting.success(truncated and acquired and not done,
                                                  self.max_position, self.max_heading)
                info['combined_success'] = info['is_success'] = info['push_recovery_success'] = success
                info['push_metrics'] = self.accounting.metrics()
                info['getup_metrics'] = dict(info['getup_metrics'], combined_success=success)
            info['episode_metrics'].update(reward=base._total_reward,
                balance_curriculum_stage=self.stage.name, strict_success=info['strict_success'])
            info['episode_metrics']['getup_metrics'] = info['getup_metrics']
        return obs, float(reward), done, truncated, info
