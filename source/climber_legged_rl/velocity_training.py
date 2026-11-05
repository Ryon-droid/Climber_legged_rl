"""Balanced motion and stop tests; motion cannot pass by standing still."""
from pathlib import Path
import hashlib
import json

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .mujoco_model import load_config
from .mujoco_env import make_mujoco_env


def command_cases(config):
    task=config['task']
    forward=min(abs(v) for v in task['command_forward_m_s'])
    yaw=min(abs(v) for v in task['command_yaw_rad_s'])
    return [('idle', [0.,0.]), ('forward', [forward,0.]), ('backward', [-forward,0.]),
            ('left', [0.,yaw]), ('right', [0.,-yaw]),
            ('forward_left', [forward,yaw]), ('backward_right', [-forward,-yaw])]


def evaluate_velocity(checkpoint, config_path, model_path, episodes=21, seed=42000,
                      horizon_seconds=None, random_actions=False):
    config=load_config(config_path)
    criteria=config['task']['velocity_evaluation']
    horizon=horizon_seconds or config['task']['episode_seconds']
    if episodes<1 or not 10<=horizon<=120:
        raise ValueError('Velocity evaluation needs positive episodes and horizon 10..120 seconds')
    policy=PPO.load(checkpoint,device='cpu') if checkpoint else None
    env=make_mujoco_env(model_path,config_path)
    rng=np.random.default_rng(seed)
    cases=command_cases(config)
    records=[]
    try:
        for index in range(episodes):
            case,command=cases[index%len(cases)]
            # Same initial-state seed for every command in each replicate.
            episode_seed=seed+index//len(cases)
            stop_seconds=horizon*.6
            obs,_=env.reset(seed=episode_seed,options=dict(velocity_command=command,stop_after_seconds=stop_seconds))
            env.max_steps=round(horizon/env.dt)
            tracking=[]; idle=[]; tilts=[]; lengths=[]; contacts=[]
            peak_displacement=0.; total_reward=0.; max_closure=0.
            while True:
                action=policy.predict(obs,deterministic=True)[0] if policy else rng.uniform(-1.,1.,6) if random_actions else np.zeros(6)
                obs,reward,done,truncated,info=env.step(action)
                total_reward+=reward
                tilts.append(info['tilt_rad'])
                lengths.append(float(np.mean((np.asarray(info['virtual_leg_lengths_m'])-env.target_virtual_leg_length)**2)))
                peak_displacement=max(peak_displacement,info['planar_displacement_m'])
                max_closure=max(max_closure,info['closure_error_m'])
                settled=info['command_age_seconds']>=env.schedule['ramp_seconds']+criteria['settle_seconds']
                if settled and not info['idle_hold']:
                    tracking.append([info['forward_velocity_m_s']-info['command_forward_m_s'],
                                     info['lateral_velocity_m_s'],info['yaw_rate_rad_s']-info['command_yaw_rad_s']])
                    contacts.append(info['wheel_contacts']==2)
                # Only the final stop window is scored as stop-hold.
                if settled and info['idle_hold'] and env.steps*env.dt>=stop_seconds+env.schedule['ramp_seconds']+criteria['settle_seconds']:
                    idle.append([info['planar_displacement_m'],abs(info['heading_error_rad']),
                                 info['planar_speed_m_s'],abs(info['yaw_rate_rad_s'])])
                if done or truncated:break
            moving=bool(np.any(command))
            rmse=np.sqrt(np.mean(np.square(tracking),axis=0)) if tracking else None
            idle_max=np.max(idle,axis=0) if idle else None
            idle_speed=np.sqrt(np.mean(np.square(np.asarray(idle)[:,2:]),axis=0)) if idle else None
            limits=[criteria['forward_rmse_m_s'],criteria['lateral_rmse_m_s'],criteria['yaw_rmse_rad_s']]
            motion_ok=bool(rmse is not None and np.all(rmse<=limits)) if moving else True
            stop_ok=bool(idle and idle_max[0]<=criteria['idle_max_displacement_m']
                and idle_max[1]<=criteria['idle_max_heading_rad']
                and idle_speed[0]<=criteria['lateral_rmse_m_s'] and idle_speed[1]<=criteria['yaw_rmse_rad_s'])
            leg_rmse=float(np.sqrt(np.mean(lengths)))
            success=bool(truncated and not done and motion_ok and stop_ok
                and max(tilts)<=criteria['max_tilt_rad'] and leg_rmse<=criteria['leg_rmse_m'])
            records.append(dict(case=case,seed=episode_seed,command=command,success=success,
                tracking_success=motion_ok,stop_success=stop_ok,seconds=env.steps*env.dt,reward=total_reward,
                reason=info['termination_reason'] or 'timeout',tracking_samples=len(tracking),stop_samples=len(idle),
                tracking_rmse=None if rmse is None else rmse.tolist(),
                stop_max_displacement_m=None if idle_max is None else float(idle_max[0]),
                stop_max_heading_rad=None if idle_max is None else float(idle_max[1]),
                stop_speed_yaw_rmse=None if idle_speed is None else idle_speed.tolist(),
                max_tilt_deg=float(np.degrees(max(tilts))),leg_rmse_m=leg_rmse,
                max_displacement_from_latest_stop_anchor_m=peak_displacement,max_closure_m=max_closure,
                both_wheels_contact_fraction=float(np.mean(contacts)) if contacts else None,
                warning_counts=env.data.warning.number.tolist()))
    finally:env.close()
    groups={name:[r for r in records if r['case']==name] for name,_ in cases}
    rates={name:float(np.mean([r['success'] for r in group])) for name,group in groups.items() if group}
    idle_group=groups['idle']
    return dict(simulation_only=True,task='velocity_tracking',episodes=records,seed=seed,
        evaluation_horizon_seconds=horizon,velocity_success_fraction=float(np.mean([r['success'] for r in records])),
        per_command_success=rates,all_commands_covered=len(rates)==len(cases),
        stationary_success_fraction=float(np.mean([r['success'] for r in idle_group])) if idle_group else None,
        stop_success_fraction=float(np.mean([r['stop_success'] for r in records])),
        timeout_fraction=float(np.mean([r['reason']=='timeout' for r in records])),
        mean_episode_seconds=float(np.mean([r['seconds'] for r in records])),
        mean_reward=float(np.mean([r['reward'] for r in records])),criteria=criteria,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
        config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        model_route=config['experiment']['model_route'])


class VelocityEvalCallback(BaseCallback):
    def __init__(self,folder,config_path,model_path,frequency=25000,evaluate_initial=True):
        super().__init__()
        self.folder=Path(folder);self.config_path=config_path;self.model_path=model_path
        self.frequency=frequency;self.evaluate_initial=evaluate_initial
        self.last_evaluation=0;self.best_score=None;self.history=[]

    def _on_training_start(self):
        if self.evaluate_initial:
            self.last_evaluation=self.num_timesteps-self.frequency
            self._on_step()

    def _on_step(self):
        if self.num_timesteps-self.last_evaluation<self.frequency:return True
        self.last_evaluation=self.num_timesteps
        candidate=self.folder/'velocity_candidate.zip'
        self.model.save(candidate)
        report=evaluate_velocity(candidate,self.config_path,self.model_path,episodes=21,seed=42000)
        score=(min(report['per_command_success'].values()),report['velocity_success_fraction'],
               report['stop_success_fraction'],report['timeout_fraction'],report['mean_reward'])
        record=dict(timesteps=self.num_timesteps,score=score,report=report)
        self.history.append(record)
        (self.folder/'velocity_evaluations.json').write_text(json.dumps(self.history,indent=2))
        for key in ('velocity_success_fraction','stop_success_fraction','mean_episode_seconds'):
            self.logger.record('eval/'+key,report[key])
        if self.best_score is None or score>self.best_score:
            self.best_score=score
            (self.folder/'best').mkdir(exist_ok=True)
            self.model.save(self.folder/'best/best_model.zip')
        candidate.unlink()
        print('VELOCITY_EVALUATION '+json.dumps(dict(timesteps=self.num_timesteps,score=score,
            per_command_success=report['per_command_success'])),flush=True)
        return True
