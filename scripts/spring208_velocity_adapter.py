"""Command task around the current get-up plant, preserving all 36 sensor slots.

This is preparation for the next curriculum. It never starts training or
changes the accepted push policy. Wrap before history/velocity supervision.
"""
import gymnasium as gym
import numpy as np


class VelocityCommandTask(gym.Wrapper):
    def __init__(self,env,forward=.10,yaw=.30):
        super().__init__(env)
        if env.observation_space.shape!=(36,):
            raise ValueError('Wrap the raw 36-slot get-up plant before history')
        self.forward_limit=forward;self.yaw_limit=yaw
        self.ramp_seconds=1.;self.resample_seconds=5.
        self.command=np.zeros(2);self.target=np.zeros(2);self.start=np.zeros(2)
        self.idle=True

    def __getattr__(self,name):
        if name.startswith('__'):raise AttributeError(name)
        return getattr(self.env,name)

    def _observation(self):
        observation=self.env._observation().copy()
        if not self.idle:observation[24:28]=0.
        return observation

    def _set_target(self,target):
        self.start=self.command.copy();self.target=np.asarray(target,dtype=float)
        self.changed_seconds=self.env.steps*self.env.dt

    def _commands(self):
        seconds=self.env.steps*self.env.dt
        if seconds>=self.next_change:
            if self.fixed is not None:
                target=self.fixed if seconds<self.stop_seconds else np.zeros(2)
                self.next_change=self.stop_seconds if seconds<self.stop_seconds else float('inf')
            else:
                limits=np.array([self.forward_limit,self.yaw_limit])
                if self.env.np_random.random()<.25:target=np.zeros(2)
                else:
                    target=self.env.np_random.uniform(-limits,limits)
                    target[int(self.env.np_random.integers(2))]=0.
                self.next_change=seconds+self.resample_seconds
            self._set_target(target)
        fraction=np.clip((seconds-self.changed_seconds)/self.ramp_seconds,0.,1.)
        self.command=self.start+fraction*(self.target-self.start)
        was_idle=self.idle
        self.idle=bool(max(abs(self.command))<1e-8 and max(abs(self.target))<1e-8)
        if self.idle and not was_idle:self.env._set_hold_reference()
        self.env.commands[:]=self.command

    def reset(self,*,seed=None,options=None):
        options=options or {}
        if options.get('reset_mode','standing')!='standing':
            raise ValueError('Velocity curriculum starts from standing')
        _,info=self.env.reset(seed=seed,options={'reset_mode':'standing'})
        self.fixed=None if options.get('velocity_command') is None else np.asarray(options['velocity_command'],dtype=float)
        if self.fixed is not None:
            if self.fixed.shape!=(2,) or not np.isfinite(self.fixed).all():raise ValueError('Invalid command')
            if np.any(abs(self.fixed)>np.array([self.forward_limit,self.yaw_limit])+1e-9):
                raise ValueError('Command outside the curriculum')
        self.stop_seconds=float(options.get('stop_after_seconds',12.))
        self.command=np.zeros(2);self.target=np.zeros(2);self.start=np.zeros(2)
        self.idle=True;self.next_change=1.;self.changed_seconds=0.
        self.env.commands[:]=0.;self.env._set_hold_reference()
        self.tracking=[];self.stop=[];self.max_tilt=0.;self.leg_errors=[]
        return self._observation(),dict(info,task='spring208_velocity',runtime_reference_assistance=False)

    def step(self,action):
        self._commands()
        obs,reward,done,truncated,info=self.env.step(action)
        executed=self.command.copy()
        angular,linear,_=self.env._kinematics()
        self.max_tilt=max(self.max_tilt,info['tilt_rad'])
        self.leg_errors.append(np.mean((np.asarray(info['virtual_leg_lengths_m'])-self.env.target_virtual_leg_length)**2))
        if not self.idle:
            terms=info['reward_terms'];removed=0.
            for key in ('position_hold','heading_hold','linear_tracking','yaw_tracking',
                        'recovery_position','recovery_velocity','recovery_heading','recovery_yaw_rate'):
                removed+=terms.get(key,0.);terms[key]=0.
            terms['command_linear_tracking']=8.*float(np.exp(-((linear[0]-executed[0])**2+linear[1]**2)/.08**2))
            terms['command_yaw_tracking']=5.*float(np.exp(-((angular[2]-executed[1])/.3)**2))
            adjustment=terms['command_linear_tracking']+terms['command_yaw_tracking']-removed
            reward+=adjustment;self.env._total_reward+=adjustment
            if self.env.steps*self.env.dt-self.changed_seconds>=self.ramp_seconds+1.:
                self.tracking.append([linear[0]-executed[0],linear[1],angular[2]-executed[1]])
        if self.idle and self.env.steps*self.env.dt>=self.stop_seconds+self.ramp_seconds+2.:
            self.stop.append([info['planar_displacement_m'],abs(info['heading_error_rad']),
                              info['planar_speed_m_s'],abs(info['yaw_rate_rad_s'])])
        info.update(command_forward_m_s=float(executed[0]),command_yaw_rad_s=float(executed[1]),
                    idle_hold=self.idle,task='spring208_velocity',runtime_reference_assistance=False)
        if done or truncated:
            rmse=np.sqrt(np.mean(np.square(self.tracking),axis=0)) if self.tracking else None
            final=np.max(self.stop,axis=0) if self.stop else None
            moving=self.fixed is None or bool(np.any(self.fixed))
            motion_ok=bool(rmse is not None and np.all(rmse<=[.04,.04,.10])) if moving else True
            stop_ok=bool(final is not None and np.all(final<=[.05,np.radians(5.),.05,.05]))
            leg_rmse=float(np.sqrt(np.mean(self.leg_errors)))
            info['velocity_metrics']=dict(tracking_rmse=None if rmse is None else rmse.tolist(),
                stop_max_errors=None if final is None else final.tolist(),
                leg_rmse_m=leg_rmse,max_tilt_deg=float(np.degrees(self.max_tilt)),
                tracking_success=motion_ok,stop_success=stop_ok,
                success=bool(truncated and not done and motion_ok and stop_ok
                             and self.max_tilt<=np.radians(10.) and leg_rmse<=.01))
            info['episode_metrics']['reward']=self.env._total_reward
        return self._observation(),float(reward),done,truncated,info
