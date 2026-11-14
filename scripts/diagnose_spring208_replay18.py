"""Read-only action/reward audit on training-only reset cases."""
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'scripts')]
import numpy as np
import torch
from stable_baselines3 import PPO
from train_spring208_push import CFG,write
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_env import make_mujoco_env


def main():
    torch.set_num_threads(1)
    folder=ROOT/'outputs/push_recovery_replay18_20261113_1945'
    parent=PPO.load(folder/'selected_policy.zip',device='cpu')
    candidates=[PPO.load(folder/f'level18_attempt{i}_step245760_final.zip',device='cpu') for i in range(2)]
    cases=json.loads((ROOT/'outputs/push_replay18_preparation_20261111/hard_training_resets.json').read_text())['cases'][:8]
    config=load_config(CFG)
    env=make_mujoco_env(ROOT/config['experiment']['output_directory']/'robot_experiment.xml',CFG)
    rows=[]
    try:
        for case in cases:
            base=env.unwrapped
            base.push.config=dict(base.push.config,episode_probability=1.,force_N=case['force_N'])
            base.push.config.pop('force_bands',None)
            obs,_=env.reset(seed=case['seed'],options={'reset_mode':'standing'})
            differences=[[],[]];raw=[];states=[];terms={};count=0;post=0
            while True:
                with torch.no_grad():
                    raw_action=parent.policy.get_distribution(torch.as_tensor(obs[None])).distribution.mean.numpy()[0]
                action=np.clip(raw_action,-1.,1.)
                for i,policy in enumerate(candidates):
                    differences[i].append(policy.predict(obs,deterministic=True)[0]-action)
                raw.append(raw_action)
                obs,reward,done,truncated,info=env.step(action)
                if base.steps>=300:
                    post+=1
                    for key,value in info['reward_terms'].items():terms[key]=terms.get(key,0.)+value
                    if base.steps%20==0:
                        states.append(dict(seconds=base.steps*.01,tilt_deg=float(np.degrees(info['tilt_rad'])),
                            actions=action.tolist(),saturation=info['motor_saturation_fraction'],torque=info['motor_torque_Nm']))
                count+=1
                if done or truncated:break
            rows.append(dict(seed=case['seed'],success=bool(info['combined_success']),seconds=count*.01,
                raw_action_min=np.min(raw,axis=0).tolist(),raw_action_max=np.max(raw,axis=0).tolist(),
                clipping_fraction=np.mean(np.abs(raw)>1.,axis=0).tolist(),
                candidate_max_action_difference=[np.max(np.abs(d),axis=0).tolist() for d in differences],
                candidate_rms_action_difference=[np.sqrt(np.mean(np.square(d),axis=0)).tolist() for d in differences],
                mean_post_push_rewards={k:v/max(post,1) for k,v in terms.items()},last_states=states[-8:]))
        write(folder/'action_reward_audit.json',dict(training_only_seeds=True,cases=rows))
        print(json.dumps(rows,ensure_ascii=False))
    finally:env.close()


if __name__=='__main__':main()
