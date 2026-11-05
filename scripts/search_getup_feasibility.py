"""Search bounded action sequences; validate actual full-prone get-up separately."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import numpy as np
from scipy.optimize import differential_evolution
from stable_baselines3 import PPO
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import ROOT,load_config
from climber_legged_rl.spring_geometry import eye_center_distance

_ENV=None
_ARGS=None


def rollout(parameters,*,offset=None,validate=False,policy=None):
    global _ENV
    args=_ARGS
    if _ENV is None:
        cfg=load_config(args.config)
        _ENV=make_mujoco_env(ROOT/cfg['experiment']['output_directory']/'robot_experiment.xml',args.config)
    env=_ENV
    bank=env.pose_bank
    if offset is not None:
        env.pose_bank=[p for p in bank if abs(p['rearward_offset_m']-offset)<1e-8]
        if not env.pose_bank:
            env.pose_bank=bank
            raise ValueError('Rearward offset is not present in the actual pose bank')
    else:
        env.pose_bank=[bank[len(bank)//2]]
    try:
        observation,reset=env.reset(seed=args.seed,options={'reset_mode':'folded'})
    finally:
        env.pose_bank=bank
    sequence=np.asarray(parameters,dtype=float).reshape(args.segments,3)
    best=math.inf
    best_at=0.
    minimum_leg=math.inf
    minimum_spring=math.inf
    maximum_tilt=maximum_displacement=0.
    trace=[]
    last=None
    acquired=False
    try:
        limit=env.max_steps if validate else min(env.max_steps,math.ceil(args.horizon_seconds/env.dt))
        for step in range(limit):
            if env.acquired and policy is not None:
                action,_=policy.predict(observation,deterministic=True)
            else:
                index=min(args.segments-1,int(step*env.dt/args.segment_seconds))
                front,rear,wheel=sequence[index]
                action=np.array([front,rear,front,rear,wheel,-wheel])
            observation,_,done,truncated,info=env.step(action)
            leg=float(np.mean(info['virtual_leg_lengths_m']))
            springs=[eye_center_distance(env.model,env.data,side) for side in ('left','right')]
            minimum_leg=min(minimum_leg,leg)
            minimum_spring=min(minimum_spring,min(springs))
            maximum_tilt=max(maximum_tilt,info['tilt_rad'])
            maximum_displacement=max(maximum_displacement,info['planar_displacement_m'])
            cost=(3*abs(leg-env.target_virtual_leg_length)+.15*info['tilt_rad']+
                  .3*abs(info['hip_height_m']-env.target_hip_height)+.03*info['planar_displacement_m'])
            if cost<best:
                best,best_at=cost,env.steps*env.dt
            acquired|=bool(info['getup_acquired'])
            last=info
            if validate and (step%10==0 or done or truncated):
                trace.append(dict(seconds=env.steps*env.dt,virtual_leg_m=info['virtual_leg_lengths_m'],
                    spring_m=springs,tilt_rad=info['tilt_rad'],hip_height_m=info['hip_height_m'],
                    displacement_m=info['planar_displacement_m'],torque_Nm=env.data.ctrl.tolist()))
            if done or truncated:
                break
        success=bool(last and last.get('combined_success',False))
        return dict(score=best-(.5 if acquired else 0.),acquired=acquired,success=success,
            seconds=env.steps*env.dt,reason=last['termination_reason'] or 'search_horizon',
            best_state_seconds=best_at,minimum_virtual_leg_m=minimum_leg,minimum_spring_m=minimum_spring,
            max_tilt_rad=maximum_tilt,max_displacement_m=maximum_displacement,reset=reset,
            warning_count=int(np.sum(env.data.warning.number)),trace=trace)
    except (RuntimeError,FloatingPointError) as error:
        return dict(score=10.,acquired=False,success=False,reason='numerical_failure',error=str(error))


def objective(parameters):
    return rollout(parameters)['score']


def main():
    global _ARGS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/robot/cad_getup_stand_round5_dynamic.yaml'))
    parser.add_argument('--standing-checkpoint',type=Path,required=True,help='Same-task actor, used only after genuine acquisition')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=12)
    parser.add_argument('--population-size',type=int,default=5,help='SciPy population multiplier, not absolute count')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--segments',type=int,default=4)
    parser.add_argument('--segment-seconds',type=float,default=1.)
    parser.add_argument('--horizon-seconds',type=float,default=5.)
    parser.add_argument('--seed',type=int,default=87)
    parser.add_argument('--initial-report',type=Path,help='Include a previous report best_parameters in the next search')
    args=parser.parse_args()
    if (not 1<=args.iterations<=200 or not 1<=args.population_size<=30 or not 1<=args.workers<=8
            or not 1<=args.segments<=10 or not np.isfinite(args.segment_seconds) or args.segment_seconds<=0
            or not np.isfinite(args.horizon_seconds) or not 0<args.horizon_seconds<=20):
        parser.error('Invalid search limits')
    cfg=load_config(args.config)
    if not cfg.get('getup',{}).get('dynamic_control',{}).get('enabled'):
        parser.error('This search requires the learned incremental dynamic get-up controller')
    if cfg.get('push',{}).get('enabled'):
        parser.error('Disable pushes when testing basic get-up feasibility')
    if not args.standing_checkpoint.is_file():
        parser.error('Standing checkpoint does not exist')
    _ARGS=args
    folder=ROOT/cfg['experiment']['output_directory']
    from climber_legged_rl.cad_dynamics import validate_artifacts
    from climber_legged_rl.mujoco_training import validate_policy_transfer
    validate_artifacts(args.config)
    validate_policy_transfer(args.standing_checkpoint,folder/'robot_experiment.xml',cfg)
    started=time.monotonic()
    history=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    progress=args.output.with_name(args.output.stem+'_progress.json')
    def callback(parameters,convergence):
        row=dict(generation=len(history)+1,score=float(objective(parameters)),
                 elapsed_seconds=time.monotonic()-started,parameters=parameters.tolist())
        history.append(row)
        progress.write_text(json.dumps(history,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='parameters'}),flush=True)
    options={}
    if args.initial_report:
        initial=np.asarray(json.loads(args.initial_report.read_text())['best_parameters'],dtype=float)
        if initial.shape!=(args.segments*3,) or not np.isfinite(initial).all() or np.any(abs(initial)>1):
            parser.error('Initial report does not match the action sequence dimensions or bounds')
        options['x0']=initial
    result=differential_evolution(objective,[(-1.,1.)]*(args.segments*3),seed=args.seed,
        maxiter=args.iterations,popsize=args.population_size,polish=False,tol=1e-5,
        workers=args.workers,updating='deferred',callback=callback,**options)
    import torch
    torch.set_num_threads(1)
    policy=PPO.load(args.standing_checkpoint,device='cpu')
    if policy.observation_space.shape!=(36,):
        raise ValueError('Post-acquisition checkpoint requires the 36-input combined task')
    offsets=cfg['getup']['rearward_offsets_m']
    validation=[rollout(result.x,offset=offset,validate=True,policy=policy) for offset in offsets]
    passed=all(row['success'] for row in validation)
    report=dict(simulation_only=True,passed=passed,config=str(args.config),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256((folder/'robot_experiment.xml').read_bytes()).hexdigest(),
        standing_checkpoint=str(args.standing_checkpoint),seed=args.seed,segments=args.segments,
        segment_seconds=args.segment_seconds,horizon_seconds=args.horizon_seconds,
        function_evaluations=int(result.nfev),best_score=float(result.fun),best_parameters=result.x.tolist(),
        validation=validation,history=history,elapsed_seconds=time.monotonic()-started,
        interpretation='Bounded symmetric action search only; failure does not prove physical impossibility. '
        'Search score is not a success rate. Full episode success requires real get-up and unchanged final standing criteria.')
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(passed=passed,validation_success_count=sum(r['success'] for r in validation),
                         validation_count=len(validation),output=str(args.output))),flush=True)
    if _ENV is not None:
        _ENV.close()
    return 0 if passed else 2


if __name__=='__main__':
    raise SystemExit(main())
