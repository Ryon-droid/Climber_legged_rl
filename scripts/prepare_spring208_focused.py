"""Discover fresh training failures, and measure complete PPO throughput."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'

import train_spring208_push_ab as trial
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv


def collect_case(spec):
    torch.set_num_threads(1)
    config=trial.load_config(trial.CFG)
    env=trial.make_mujoco_env(trial.ROOT/config['experiment']['output_directory']/'robot_experiment.xml',trial.CFG)
    parent=trial.ROOT/'outputs/push_exploration_ab18_20261111_1405/selected_policy.zip'
    policy=PPO.load(parent,device='cpu')
    base=env.unwrapped
    base.push.config=dict(base.push.config,episode_probability=float(spec['push']),force_N=spec['force_N'])
    base.push.config.pop('force_bands',None)
    xs=[];ys=[];trace=[]
    try:
        obs,_=env.reset(seed=spec['seed'],options={'reset_mode':'standing'})
        while True:
            action=policy.predict(obs,deterministic=True)[0]
            if spec['anchor'] and base.steps%4==0:
                xs.append(obs[:-3].copy());ys.append(np.clip(action,-1.,1.))
            if base.steps%20==0:
                pitch,roll=base._attitude()
                with torch.no_grad():
                    estimated=policy.policy.mlp_extractor.policy_net.estimate_velocity(torch.as_tensor(obs[None])).numpy()[0]
                trace.append(dict(seconds=base.steps*base.dt,pitch_deg=float(np.degrees(pitch)),
                    roll_deg=float(np.degrees(roll)),wheel_actions=action[4:].tolist(),
                    wheel_velocities=base.data.qvel[base.v_indices[4:]].tolist(),
                    velocity_label=obs[-3:].tolist(),velocity_estimate=estimated.tolist()))
            obs,_,done,truncated,info=env.step(action)
            if done or truncated:break
        outcome=dict(spec,success=bool(info['combined_success']),seconds=base.steps*base.dt,
            reason=info['termination_reason'],push_probability=float(spec['push']),
            push_events=info['push_metrics']['push_events'],peak_torque_Nm=info['peak_torque_Nm'],
            final_tilt_deg=float(np.degrees(info['tilt_rad'])),
            final_heading_deg=float(np.degrees(info['heading_error_rad'])),
            failed_before_first_push=bool(spec['push'] and not info['push_metrics']['push_count']),
            warnings=int(sum(base.data.warning.number)),trace=trace)
        return outcome,(np.asarray(xs,np.float32),np.asarray(ys,np.float32))
    finally:env.close()


def prepare(profile='focused18'):
    trial.configure_profile(profile);trial.setup()
    folder=trial.SOURCE
    folder.mkdir(parents=True,exist_ok=False)
    rehearsal=folder/'rehearsal';rehearsal.mkdir()
    specs=[dict(seed=1070000+i,anchor=i<24,push=i>=24,
        force_N=([2.,18.] if i<24 else [10.,15.] if i<40 else [15.,18.] if i<64 else [18.,18.]))
        for i in range(96)]
    if profile=='recovery_replay18':
        specs=[dict(seed=1140000+i,anchor=True,push=i>=24,
            force_N=([2.,10.] if i<40 else [10.,15.] if i<64 else [15.,18.] if i<80 else [18.,18.]))
            for i in range(96)]
    outcomes=[];xs=[];ys=[];cases=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        for outcome,arrays in pool.map(collect_case,specs):
            outcomes.append(outcome)
            if outcome['anchor'] and outcome['success']:
                xs.append(arrays[0]);ys.append(arrays[1])
            if outcome['push'] and not outcome['success']:
                cases.append(dict(seed=outcome['seed'],force_N=outcome['force_N'],
                    expected_push_events=outcome['push_events']))
            print(json.dumps({k:v for k,v in outcome.items() if k!='trace'}),flush=True)
    if not xs or not cases:raise RuntimeError('Missing successful standing data or hard training cases')
    x=np.concatenate(xs);y=np.concatenate(ys)
    config=trial.load_config(trial.CFG)
    xml=trial.ROOT/config['experiment']['output_directory']/'robot_experiment.xml'
    dataset=rehearsal/'successful_standing.npz'
    np.savez_compressed(dataset,observations=x,actions=y,weights=np.ones(len(x),np.float64))
    (rehearsal/'config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True))
    trial.write(dataset.with_suffix('.json'),dict(dataset_sha256=trial.sha(dataset),
        model_sha256=trial.sha(xml),parent_sha256=trial.PARENT_SHA,successful_data_only=True,
        runtime_reference_assistance=False,samples=len(x),actor_has_no_privileged_velocity=True,
        training_seed_range=[specs[0]['seed'],specs[-1]['seed']],outcomes=outcomes,profile=profile))
    trial.write(folder/'hard_training_resets.json',dict(cases=cases,source='fresh training discovery only'))
    trial.write(folder/'discovery.json',dict(parent_sha256=trial.PARENT_SHA,training_seed_range=[specs[0]['seed'],specs[-1]['seed']],
        episodes=len(outcomes),samples=len(x),hard_cases=len(cases),selection_or_test_imported=False,
        summary={name:sum(1 for e in outcomes if (e['reason'] or 'timeout')==name and not e['success'])
                 for name in sorted({e['reason'] or 'timeout' for e in outcomes if not e['success']})}))


def benchmark(folder):
    trial.configure_profile('focused18');trial.setup()
    results=[]
    for count in (4,6,8,8,6,4):
        trial.NUM_ENVS=count
        vector=SubprocVecEnv([trial.factory(folder,1080000+j) for j in range(count)],start_method='spawn')
        try:
            model,_=trial.learner(folder,'standing_only',vector)
            model.set_logger(configure(str(folder/f'benchmark_{count}'),['csv']))
            model.learn(total_timesteps=6144,reset_num_timesteps=True)
            started=time.monotonic()
            model.learn(total_timesteps=12288,reset_num_timesteps=True)
            seconds=time.monotonic()-started
            row=dict(num_envs=count,transitions=12288,seconds=seconds,
                transitions_per_second=12288/seconds,rollout_transitions=6144,
                includes_policy_and_ppo_and_rehearsal=True,warmup_discarded=True)
            results.append(row);print(json.dumps(row),flush=True)
        finally:vector.close()
    # Prefer fewer processes when a measured speed advantage is below 3%.
    aggregate=[dict(num_envs=count,transitions_per_second=
        sum(r['transitions'] for r in results if r['num_envs']==count)/
        sum(r['seconds'] for r in results if r['num_envs']==count)) for count in (4,6,8)]
    fastest=max(r['transitions_per_second'] for r in aggregate)
    chosen=min(r['num_envs'] for r in aggregate if r['transitions_per_second']>=fastest*.97)
    trial.write(folder/'concurrency_benchmark.json',dict(results=results,aggregate=aggregate,selected_num_envs=chosen,
        selection_rule='fewest processes within 3% of maximum measured throughput',
        hardware='RTX2080SUPER host; CPU MuJoCo and CPU PPO',saved_training_updates=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path)
    parser.add_argument('--profile',choices=['focused18','recovery_replay18'],default='focused18')
    args=parser.parse_args()
    if args.benchmark:benchmark(args.benchmark.resolve())
    else:prepare(args.profile)
