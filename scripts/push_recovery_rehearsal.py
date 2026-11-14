"""Training-only successful-policy anchors and fresh hard-reset discovery."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'scripts')]
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.standing_rehearsal import load_standing_rehearsal
from train_spring208_push import CFG,write
from prepare_spring208_phase3 import sha


class ScaledReward(gym.RewardWrapper):
    def __init__(self,env,factor):
        super().__init__(env);self.factor=factor
    def reward(self,reward):return reward*self.factor


def settings(folder,profile='anchored30'):
    dataset=Path(folder)/'rehearsal/successful_standing.npz'
    result=dict(enabled=True,dataset=str(dataset),dataset_sha256=sha(dataset),
        updates_per_rollout=4,reference_transitions=4096,batch_size=128,
        learning_rate=1e-6,max_grad_norm=1.,action_error_scales=[.025]*4+[.25]*2)
    if profile=='stability18':result.update(updates_per_rollout=16,batch_size=256)
    return result


def prepare(folder,profile='anchored30'):
    from train_spring208_push150 import PARENT,PARENT_SHA
    torch.set_num_threads(1)
    if sha(PARENT)!=PARENT_SHA:raise RuntimeError('Unexpected parent')
    validate_artifacts(CFG)
    destination=folder/'rehearsal';destination.mkdir(parents=True,exist_ok=True)
    config=load_config(CFG);xml=ROOT/config['experiment']['output_directory']/'robot_experiment.xml'
    env=make_mujoco_env(xml,CFG);policy=PPO.load(PARENT,device='cpu')
    xs=[];ys=[];ws=[];outcomes=[];hard=[];cases=[]
    stable=profile=='stability18'
    try:
        for i in range(128 if stable else 48):
            anchor=i<(96 if stable else 24);seed=(940000 if stable else 910000)+i
            if stable:
                push_probability=0. if i<32 else 1.
                force_range=[2.,10.] if i<64 else ([10.,15.] if i<96 else [15.,18.])
            else:
                push_probability=0. if i<8 else 1.
                force_range=[2.,15.] if anchor else [15.,18.]
            # All resets are newly generated training data. No held-out or
            # previous selection/test trajectory is imported into the replay.
            env.unwrapped.push.config=dict(env.unwrapped.push.config,episode_probability=push_probability,
                force_N=force_range)
            env.unwrapped.push.config.pop('force_bands',None)
            obs,_=env.reset(seed=seed,options={'reset_mode':'standing'})
            episode_x=[];episode_y=[];episode_w=[]
            while True:
                action=policy.predict(obs,deterministic=True)[0]
                if anchor and env.steps%4==0:
                    episode_x.append(obs[:-3].copy());episode_y.append(np.clip(action,-1.,1.))
                    active=env.unwrapped.push.active
                    near=active is not None and env.steps<=active['end_step']+200
                    episode_w.append(3. if near else 1.)
                obs,_,done,truncated,info=env.step(action)
                if done or truncated:break
            success=bool(info['combined_success'])
            outcomes.append(dict(seed=seed,anchor=anchor,success=success,reason=info['termination_reason'],
                push_count=info['push_metrics']['push_count'],seconds=env.steps*env.dt,
                force_N=force_range,push_probability=push_probability,
                push_events=info['push_metrics']['push_events']))
            if anchor and success:
                xs.extend(episode_x);ys.extend(episode_y);ws.extend(episode_w)
            elif not anchor and not success:
                hard.append(seed)
            if stable and not success and push_probability:
                cases.append(dict(seed=seed,force_N=force_range,
                    expected_push_events=info['push_metrics']['push_events']))
            print(json.dumps(outcomes[-1]),flush=True)
    finally:env.close()
    if len(xs)<1000 or not hard:raise RuntimeError('Insufficient successful anchors or hard resets')
    dataset=destination/'successful_standing.npz'
    np.savez_compressed(dataset,observations=np.asarray(xs,np.float32),actions=np.asarray(ys,np.float32),weights=np.asarray(ws,np.float64))
    (destination/'config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True))
    report=dict(dataset_sha256=sha(dataset),model_sha256=sha(xml),parent_sha256=PARENT_SHA,
        successful_data_only=True,runtime_reference_assistance=False,samples=len(xs),
        actor_has_no_privileged_velocity=True,outcomes=outcomes,
        training_seed_range=[940000,940127] if stable else [910000,910047],profile=profile)
    write(dataset.with_suffix('.json'),report)
    write(folder/'hard_training_resets.json',dict(seeds=hard,probability=.2 if stable else .3,
        source='fresh training discovery only',**(dict(cases=cases,exact_force_distribution=True) if stable else {})))
    # Reuse the repository's strict shape, actor information and provenance checks.
    load_standing_rehearsal(settings(folder,profile),config,xml)
    write(folder/'rehearsal_preparation.json',dict(passed=True,samples=len(xs),hard_training_seeds=hard,
        training_seeds=list(range(940000,940128)) if stable else list(range(910000,910048)),
        dataset_sha256=sha(dataset),model_sha256=sha(xml),profile=profile))
    print(json.dumps(dict(prepared=True,samples=len(xs),hard_resets=len(hard))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--profile',choices=['anchored30','stability18'],default='anchored30')
    args=parser.parse_args();prepare(args.folder.resolve(),args.profile)
