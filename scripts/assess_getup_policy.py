"""Evaluate get-up and standing separately with physical travel and motion metrics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import numpy as np
import torch
from stable_baselines3 import PPO
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import ROOT, load_config
from climber_legged_rl.spring_geometry import eye_center_distance


def assess(config, checkpoint, episodes, seed, horizon=None, modes=('standing','folded'), rearward_offset=None):
    cfg = load_config(config)
    model_path = ROOT / cfg['experiment']['output_directory'] / 'robot_experiment.xml'
    env = make_mujoco_env(model_path, config)
    if rearward_offset is not None:
        env.pose_bank = [p for p in env.pose_bank if abs(p['rearward_offset_m']-rearward_offset)<1e-8]
        if not env.pose_bank:
            env.close()
            raise ValueError('Rearward offset must match the prepared pose bank')
    policy = PPO.load(checkpoint, device='cpu')
    results = {}
    try:
        for mode in modes:
            records = []
            for episode in range(episodes):
                obs, reset = env.reset(seed=seed+episode, options={'reset_mode': mode})
                if horizon:
                    env.max_steps = round(horizon/env.dt)
                maxima = dict(tilt_rad=0., displacement_m=0., heading_rad=0., closure_m=0.)
                sum_speed = sum_yaw = 0.
                minimum_length = 1.
                spring_min, spring_max = 1., 0.
                saturation = np.zeros(6)
                trace = []
                while True:
                    action, _ = policy.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    for name, key in [('tilt_rad','tilt_rad'),('displacement_m','planar_displacement_m'),
                                      ('heading_rad','heading_error_rad'),('closure_m','closure_error_m')]:
                        maxima[name] = max(maxima[name], abs(info[key]))
                    sum_speed += info['planar_speed_m_s']**2
                    sum_yaw += info['yaw_rate_rad_s']**2
                    minimum_length = min(minimum_length, float(np.mean(info['virtual_leg_lengths_m'])))
                    lengths = [eye_center_distance(env.model, env.data, side) for side in ('left','right')]
                    spring_min, spring_max = min(spring_min,min(lengths)), max(spring_max,max(lengths))
                    saturation += np.asarray(info['motor_saturation_fraction'])
                    if episode < 3 and env.steps % 50 == 0:
                        trace.append(dict(seconds=env.steps*env.dt, virtual_leg_m=info['virtual_leg_lengths_m'],
                            tilt_rad=info['tilt_rad'],position_m=info['planar_displacement_m'],
                            leg_q=env.data.qpos[env.q_indices[:4]].tolist(),torque=env.data.ctrl.tolist(),
                            spring_m=lengths, action=action.tolist()))
                    if done or truncated:
                        records.append(dict(seed=seed+episode,reset=reset,seconds=env.steps*env.dt,
                            reason=info['termination_reason'] or 'timeout',success=info['combined_success'],
                            acquired=info['getup_acquired'],getup_seconds=info['getup_seconds'],maxima=maxima,
                            planar_speed_rmse_m_s=float(np.sqrt(sum_speed/env.steps)),
                            yaw_rate_rmse_rad_s=float(np.sqrt(sum_yaw/env.steps)),
                            minimum_virtual_leg_m=minimum_length,spring_min_m=spring_min,spring_max_m=spring_max,
                            saturation_fraction=(saturation/env.steps).tolist(),warnings=env.data.warning.number.tolist(),trace=trace,
                            over_rated_seconds=info.get('over_rated_seconds'),peak_torque_Nm=info.get('peak_torque_Nm'),
                            final_state={k:info[k] for k in ('hip_height_m','virtual_leg_lengths_m','tilt_rad',
                                'planar_displacement_m','heading_error_rad','planar_speed_m_s','yaw_rate_rad_s','wheel_contacts')}))
                        break
            results[mode] = dict(episodes=records, success_count=sum(r['success'] for r in records),
                acquired_count=sum(r['acquired'] for r in records), count=len(records),
                reasons=dict(Counter(r['reason'] for r in records)),
                max_tilt_deg=float(np.degrees(max(r['maxima']['tilt_rad'] for r in records))),
                max_displacement_m=max(r['maxima']['displacement_m'] for r in records),
                max_heading_deg=float(np.degrees(max(r['maxima']['heading_rad'] for r in records))),
                max_planar_speed_rmse_m_s=max(r['planar_speed_rmse_m_s'] for r in records),
                max_yaw_rate_rmse_rad_s=max(r['yaw_rate_rmse_rad_s'] for r in records),
                min_virtual_leg_m=min(r['minimum_virtual_leg_m'] for r in records),
                max_spring_m=max(r['spring_max_m'] for r in records),
                warning_count=sum(sum(r['warnings']) for r in records))
            print(mode,json.dumps({k:v for k,v in results[mode].items() if k!='episodes'}),flush=True)
    finally:
        env.close()
    return dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        config=str(config),config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),seed=seed,horizon=horizon,groups=results,
        rearward_offset_m=rearward_offset,simulation_only=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--seed',type=int,default=72000)
    parser.add_argument('--horizon',type=float)
    parser.add_argument('--rearward-offset-m',type=float)
    parser.add_argument('--reset-modes',nargs='+',choices=['standing','folded','curriculum'],default=['standing','folded'])
    args=parser.parse_args()
    if args.episodes<1 or (args.horizon is not None and not 0<args.horizon<=120):parser.error('Invalid episodes or horizon')
    torch.set_num_threads(1)
    report=assess(args.config,args.checkpoint,args.episodes,args.seed,args.horizon,args.reset_modes,args.rearward_offset_m)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
