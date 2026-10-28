"""Measure posture, movement and effort under the nominal or next-round task."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))

import numpy as np
from stable_baselines3 import PPO
from climber_legged_rl.mujoco_env import ClimberMujocoEnv


def audit(checkpoint,config_path=None,episodes=5,horizon_seconds=30,seed=81000):
    checkpoint=Path(checkpoint).resolve()
    folder=checkpoint.parent.parent if checkpoint.parent.name in ('best','checkpoints') else checkpoint.parent
    config_path=Path(config_path or folder/'config.yaml').resolve()
    policy=PPO.load(checkpoint,device='cpu')
    env=ClimberMujocoEnv(folder/'robot_experiment.xml',config_path)
    env.max_steps=int(np.ceil(horizon_seconds/env.dt))
    records=[]
    try:
        for episode in range(episodes):
            obs,_=env.reset(seed=seed+episode)
            start=env.data.xpos[env.base_id,:2].copy()
            previous=start.copy()
            tilts=[];height_errors=[];action_rates=[];saturation=[];distances=[]
            length_errors=[];headings=[];planar_speeds=[]
            contact_fraction=[];angles=[];yaw=[];closure=[];path=0.;nonwheel_steps=0
            last_action=np.zeros(6)
            while True:
                action,_=policy.predict(obs,deterministic=True)
                obs,_,done,truncated,info=env.step(action)
                position=env.data.xpos[env.base_id,:2].copy()
                path+=float(np.linalg.norm(position-previous));previous=position
                distances.append(float(np.linalg.norm(position-start)))
                tilts.append(float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))))
                height_errors.append(info['hip_height_m']-env.target_hip_height)
                if 'virtual_leg_lengths_m' in info:
                    length_errors.extend(np.asarray(info['virtual_leg_lengths_m'])-env.target_virtual_leg_length)
                action_rates.append(float(np.mean((action-last_action)**2)));last_action=action.copy()
                saturation.append(float(np.mean(np.abs(env.data.ctrl)>=.99*env.torque_limits)))
                contact_fraction.append(info['wheel_contacts']==2)
                yaw.append(info['yaw_rate_rad_s']);closure.append(info['closure_error_m'])
                headings.append(info['heading_error_rad']);planar_speeds.append(info['planar_speed_m_s'])
                angles.append(env.data.qpos[env.q_indices[:4]].copy())
                nonwheel=False
                for contact in env.data.contact:
                    pair={contact.geom1,contact.geom2}
                    if env.ground_id in pair:
                        other=next(g for g in pair if g!=env.ground_id)
                        if other not in env.wheel_geom_ids:nonwheel=True
                nonwheel_steps+=int(nonwheel)
                if done or truncated:
                    records.append({'seed':seed+episode,'seconds':env.steps*env.dt,
                        'reason':info['termination_reason'] or 'timeout',
                        'tilt_mean_deg':float(np.mean(tilts)),'tilt_max_deg':float(max(tilts)),
                        'height_error_mean_m':float(np.mean(height_errors)),
                        'height_error_rmse_m':float(np.sqrt(np.mean(np.square(height_errors)))),
                        'final_planar_displacement_m':distances[-1],
                        'max_planar_displacement_m':float(max(distances)),'path_length_m':path,
                        'both_wheels_contact_fraction':float(np.mean(contact_fraction)),
                        'nonwheel_ground_contact_fraction':nonwheel_steps/env.steps,
                        'torque_saturation_fraction':float(np.mean(saturation)),
                        'action_rate_mse':float(np.mean(action_rates)),
                        'yaw_rate_rmse_rad_s':float(np.sqrt(np.mean(np.square(yaw)))),
                        'max_closure_error_m':float(max(closure)),
                        'warnings':env.data.warning.number.tolist(),
                        'mean_leg_positions_rad':np.mean(angles,axis=0).tolist()})
                    if length_errors:
                        records[-1]['virtual_leg_length_error_mean_m']=float(np.mean(length_errors))
                        records[-1]['virtual_leg_length_error_rmse_m']=float(np.sqrt(np.mean(np.square(length_errors))))
                    unwrapped=np.unwrap(headings)
                    records[-1]['max_heading_error_deg']=float(np.degrees(max(abs(unwrapped))))
                    records[-1]['final_heading_error_deg']=float(np.degrees(unwrapped[-1]))
                    records[-1]['planar_speed_rmse_m_s']=float(np.sqrt(np.mean(np.square(planar_speeds))))
                    records[-1]['stationary_success']=env.stationary_success(
                        truncated,max(distances),float(max(abs(unwrapped))),
                        records[-1]['planar_speed_rmse_m_s'],records[-1]['yaw_rate_rmse_rad_s'])
                    if env.push.enabled:
                        records[-1]['push_metrics']=info['push_metrics']
                        records[-1]['push_recovery_success']=info['push_recovery_success']
                    break
    finally:env.close()
    return {'checkpoint':str(checkpoint),'checkpoint_timesteps':policy.num_timesteps,
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'config':str(config_path),'horizon_seconds':horizon_seconds,'seed':seed,
            'target_hip_height_m':env.target_hip_height,
            'target_virtual_leg_length_m':getattr(env,'target_virtual_leg_length',None),
            'timeout_fraction':float(np.mean([r['reason']=='timeout' for r in records])),
            'stationary_success_fraction':float(np.mean([r['stationary_success'] for r in records])),
            **({'push_recovery_success_fraction':float(np.mean([r['push_recovery_success'] for r in records])),
                'push_episode_count':sum(r['push_metrics']['push_count']>0 for r in records),
                'pushed_episode_success_fraction':float(np.mean([r['push_recovery_success'] for r in records if r['push_metrics']['push_count']>0]))
                    if any(r['push_metrics']['push_count']>0 for r in records) else None} if env.push.enabled else {}),
            'summary':{key:float(np.mean([r[key] for r in records])) for key in records[0]
                       if key not in ('seed','reason','warnings','mean_leg_positions_rad','push_metrics')},
            'episodes':records,'simulation_only':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--episodes',type=int,default=5)
    parser.add_argument('--horizon-seconds',type=float,default=30)
    parser.add_argument('--seed',type=int,default=81000)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.episodes<1 or not 0<args.horizon_seconds<=120:parser.error('Require positive episodes and horizon in (0,120]')
    report=audit(args.checkpoint,args.config,args.episodes,args.horizon_seconds,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'},ensure_ascii=False,indent=2))
