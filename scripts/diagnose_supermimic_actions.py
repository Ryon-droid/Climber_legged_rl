"""Diagnostic: compare learned one-step action sensitivities to native MuJoCo.

Used after the bounded pilot, never for its checkpoint selection or final tests.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def native_probe(folder):
    import mujoco
    from supermimic_pilot import make_env,pack,layout,write_json
    report=json.loads((folder/'dataset.json').read_text())
    schema=json.loads((folder/'schema.json').read_text());c=schema['controller']
    rows=[r for r in report['episodes'] if r['role']=='train' and r['noise']==0]
    selected=[]
    for offset in sorted(set(r['offset_m'] for r in rows)):
        row=next(r for r in rows if r['offset_m']==offset)
        arrays=np.load(row['path'])
        for t in (.2,.8,1.5,2.):
            i=min(int(round(t/.01)),len(arrays['actions'])-1)
            state=arrays['states'][i]
            if state[c+1]>.5:continue
            selected.append((state.copy(),arrays['actions'][i].copy()))
    env=make_env();task=env.unwrapped
    states=[];actions=[];next_q=[];next_v=[];rewards=[]
    # Same integration warm-start for both perturbation signs, so derivatives
    # describe the physical state/action map rather than history of solver calls.
    try:
        for state,action in selected:
            for motor in range(6):
                for sign in (-1.,1.):
                    perturbed=action.copy();perturbed[motor]+=sign*.001
                    perturbed=np.clip(perturbed,-1,1)
                    env.reset(seed=610001,options={'reset_mode':'folded'})
                    task.data.qpos[:]=state[:schema['nq']]
                    task.data.qvel[:]=state[schema['nq']:schema['history']]
                    task.steps=int(round(float(state[c])/.01));task.data.time=task.steps*.01
                    task.acquired=bool(state[c+1]>.5);task.settled_steps=int(round(float(state[c+2])/.01))
                    task.reset_hip_height=float(state[c+3]);task.reset_length_m=float(state[c+4])
                    task.hold_position=state[c+5:c+7].astype(float).copy();task.hold_heading=float(state[c+7])
                    task.action[:]=state[schema['latest']+18:schema['latest']+24]
                    task.data.qacc_warmstart[:]=0.
                    mujoco.mj_forward(task.model,task.data)
                    _,reward,done,truncated,info=task.step(perturbed)
                    states.append(state);actions.append(perturbed)
                    next_q.append(task.data.qpos.copy());next_v.append(task.data.qvel.copy())
                    rewards.append(reward)
        np.savez_compressed(folder/'action_probe.npz',states=np.array(states),actions=np.array(actions),
            next_qpos=np.array(next_q),next_qvel=np.array(next_v),rewards=np.array(rewards),epsilon=.001)
        write_json(folder/'action_probe_protocol.json',dict(states=len(selected),queries=len(states),
            epsilon=.001,role='post-pilot diagnostic; not final test or selection',
            solver_warm_start='zero and identical across both perturbation signs'))
    finally:env.close()


def model_probe(folder,cycle):
    import torch
    from supermimic_world_model import WorldModel
    torch.set_num_threads(1)
    g=json.loads((folder/'schema.json').read_text());a=np.load(folder/'action_probe.npz')
    payload=torch.load(folder/f'world_{cycle}.pt',map_location='cuda',weights_only=True)['model']
    model=WorldModel(g,payload['mean'].cpu().numpy(),payload['scale'].cpu().numpy(),payload['delta_scale'].cpu().numpy()).cuda()
    model.load_state_dict(payload)
    with torch.no_grad():
        predicted,predicted_reward,_=model(torch.tensor(a['states'],device='cuda'),torch.tensor(a['actions'],device='cuda'))
        predicted=predicted.cpu().numpy()
        predicted_reward=predicted_reward.cpu().numpy()
    select=g['q_indices'][:4]
    physical=np.concatenate([a['next_qpos'][:,select],a['next_qvel'][:,g['v_indices'][4:]]],-1)
    proxy=np.concatenate([predicted[:,select],predicted[:,[g['nq']+i for i in g['v_indices'][4:]]]],-1)
    observed=((physical[1::2]-physical[::2])/.002).reshape(-1,6,6)
    learned=((proxy[1::2]-proxy[::2])/.002).reshape(-1,6,6)
    scales=np.array([1.]*4+[.05]*2)
    observed=observed*scales;learned=learned*scales
    cosine=np.sum(observed*learned,axis=-1)/(np.linalg.norm(observed,axis=-1)*np.linalg.norm(learned,axis=-1)+1e-12)
    relative=np.linalg.norm(learned-observed,axis=-1)/(np.linalg.norm(observed,axis=-1)+1e-8)
    reward_native=((a['rewards'][1::2]-a['rewards'][::2])/.002).reshape(-1,6)
    reward_learned=((predicted_reward[1::2]-predicted_reward[::2])/.002).reshape(-1,6)
    reward_cos=np.sum(reward_native*reward_learned,axis=-1)/(np.linalg.norm(reward_native,axis=-1)*np.linalg.norm(reward_learned,axis=-1)+1e-12)
    result=dict(cycle=cycle,states=len(observed),
        median_action_response_cosine=float(np.median(cosine)),
        fraction_action_directions_positive_cosine=float(np.mean(cosine>0)),
        median_relative_jacobian_error=float(np.median(relative)),
        per_motor_median_cosine=np.median(cosine,axis=0).tolist(),
        median_reward_action_gradient_cosine=float(np.median(reward_cos)),
        note='Finite native perturbations on training states; post-pilot diagnosis, not a performance claim.')
    (folder/f'action_probe_model_{cycle}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['native','model'])
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--cycle',type=int,default=0)
    args=p.parse_args()
    if args.mode=='native':native_probe(args.folder)
    else:model_probe(args.folder,args.cycle)
