"""Native parity and command-sign checks for the pending velocity curriculum."""
import argparse
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'scripts')]
import numpy as np
import torch
from stable_baselines3 import PPO
from climber_legged_rl.mujoco_env import make_mujoco_env,_make_task_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.observation_history import wrap_observations
from spring208_velocity_adapter import VelocityCommandTask
from prepare_spring208_phase3 import PARENT,sha


def create():
    cfg=ROOT/'configs/robot/cad_spring208_phase3_nominal.yaml';config=load_config(cfg)
    model=ROOT/config['experiment']['output_directory']/'robot_experiment.xml'
    raw=_make_task_env(model,cfg)
    raw.push.config=dict(raw.push.config,episode_probability=0.)
    task=VelocityCommandTask(raw)
    return wrap_observations(task,config)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);policy=PPO.load(PARENT,device='cpu')
    a=create();cfg=ROOT/'configs/robot/cad_spring208_phase3_nominal.yaml';config=load_config(cfg)
    b=make_mujoco_env(ROOT/config['experiment']['output_directory']/'robot_experiment.xml',cfg)
    b.unwrapped.push.config=dict(b.unwrapped.push.config,episode_probability=0.)
    try:
        x,_=a.reset(seed=650001,options={'velocity_command':[0.,0.]})
        y,_=b.reset(seed=650001,options={'reset_mode':'standing'})
        np.testing.assert_array_equal(x,y)
        for _ in range(400):
            act=policy.predict(x,deterministic=True)[0]
            x,r,d,t,_=a.step(act);y,rr,dd,tt,_=b.step(act)
            np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(a.unwrapped.data.qpos,b.unwrapped.data.qpos)
            assert r==rr and (d,t)==(dd,tt)
    finally:a.close();b.close()
    rows=[]
    for command in ([.1,0.],[-.1,0.],[0.,.3],[0.,-.3]):
        a=create()
        try:
            obs,_=a.reset(seed=650002,options={'velocity_command':command})
            for _ in range(220):
                obs,_,d,t,info=a.step(policy.predict(obs,deterministic=True)[0])
                if d or t:raise RuntimeError('Plant failed during command interface check')
            np.testing.assert_allclose(obs[150:152],np.asarray(command)*[2.,.25],atol=1e-7)
            assert obs.shape==(183,) and obs[-4]==1.
            altered=obs.copy();altered[-3:]=[100.,-100.,50.]
            np.testing.assert_array_equal(policy.predict(obs,deterministic=True)[0],
                                          policy.predict(altered,deterministic=True)[0])
            rows.append(dict(command=command,command_slots=obs[150:152].tolist(),phase=float(obs[179])))
        finally:a.close()
    report=dict(passed=True,idle_physics_and_reward_bit_identical_400_steps=True,
        command_sign_checks=rows,actor_ignores_privileged_velocity=True,
        current_parent_sha256=sha(PARENT),training_started=False)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
