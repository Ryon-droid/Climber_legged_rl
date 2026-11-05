"""Opt-in learned get-up targets with transient pitch and unchanged final hold."""
from __future__ import annotations
import numpy as np
from .getup import ClimberGetupEnv


class ClimberDynamicGetupEnv(ClimberGetupEnv):
    """One actor learns rising; standing retains the historical residual PD map."""

    def __init__(self,model_path,config_path,render_mode=None):
        super().__init__(model_path,config_path,render_mode)
        self.dynamic=self.getup['dynamic_control']
        if self.getup.get('reset_curriculum',{}).get('enabled'):
            raise ValueError('Dynamic get-up does not use near-standing reset curriculum')
        for key in ('incremental_target_scale_rad','max_pitch_rad','max_roll_rad','height_progress_reward'):
            if not np.isfinite(self.dynamic.get(key,0)) or self.dynamic[key]<=0:
                raise ValueError('Invalid dynamic get-up parameter: '+key)
        if not self.getup['tilt_tolerance_rad'] < self.dynamic['max_pitch_rad'] < np.pi/2:
            raise ValueError('Dynamic pitch envelope must precede overturning')
        if not self.getup['tilt_tolerance_rad'] < self.dynamic['max_roll_rad'] < np.pi/2:
            raise ValueError('Dynamic roll envelope must precede overturning')

    def _phase(self):
        if getattr(self,'acquired',True):
            return 1.
        return min(.99,self.steps*self.dt/self.getup['deadline_seconds'])

    def _leg_reference(self):
        # No hidden time-driven joint path in the observation or controller.
        return self.nominal_legs

    def _leg_targets(self,action):
        if getattr(self,'acquired',True):
            return super()._leg_targets(action)
        return np.clip(self.data.qpos[self.q_indices[:4]]+
                       action[:4]*self.dynamic['incremental_target_scale_rad'],
                       self.leg_low,self.leg_high)

    def _attitude(self):
        rotation=self._rotation()
        pitch=float(np.arctan2(-rotation[2,0],np.hypot(rotation[0,0],rotation[1,0])))
        roll=float(np.arctan2(rotation[2,1],rotation[2,2]))
        return pitch,roll

    def _termination_reason(self,base_contact,upright,max_closure):
        if getattr(self,'acquired',True):
            return super()._termination_reason(base_contact,upright,max_closure)
        if max_closure>self.config['task']['terminate_max_closure_error_m']:
            return 'closure'
        pitch,roll=self._attitude()
        if abs(roll)>self.dynamic['max_roll_rad']:
            return 'getup_roll'
        if abs(pitch)>self.dynamic['max_pitch_rad']:
            return 'getup_pitch'
        if self.steps*self.dt>=self.getup['deadline_seconds']:
            return 'getup_timeout'
        return None

    def reset(self,*,seed=None,options=None):
        observation,info=super().reset(seed=seed,options=options)
        self.reset_hip_height=float(np.mean(self.data.xpos[self.hip_ids,2]))
        return observation,info

    def step(self,action):
        was_rising=not self.acquired
        observation,reward,done,truncated,info=super().step(action)
        if was_rising:
            # Supply learning signal before leg compression; no fake state reset.
            denominator=max(self.target_hip_height-self.reset_hip_height,.01)
            height_progress=float(np.clip((info['hip_height_m']-self.reset_hip_height)/denominator,0.,1.))
            pitch,roll=self._attitude()
            # Raising the hip by tipping the chassis alone must not earn full lift reward.
            extra=height_progress*float(np.exp(-(pitch/.7)**2-(roll/.25)**2))*self.dynamic['height_progress_reward']
            reward+=extra
            self._total_reward+=extra
            info['reward_terms']['dynamic_lift_progress']=extra
            info.update(getup_pitch_rad=pitch,getup_roll_rad=roll,dynamic_height_progress=height_progress)
        if done or truncated:
            info['episode_metrics']['reward']=self._total_reward
            info['getup_metrics'].update(guided_reference=False,control_mode='learned_incremental_pd_then_standing_residual')
        return observation,float(reward),done,truncated,info
