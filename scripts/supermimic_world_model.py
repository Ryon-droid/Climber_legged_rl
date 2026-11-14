"""GPU-only PyTorch world-model fitting/BPTT, isolated from the CPU simulator stack.

Small SuperMimic-inspired pilot: replay k=8; short H=16 imagination and critic.
Reference reward, physics, and native-policy testing are shared with PPO controls.
"""
import argparse
import json
import os
from pathlib import Path
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def sequential(weights,prefix):
    keys=sorted(int(k.split('.')[0]) for k in weights if k.startswith(prefix) and k.endswith('.weight')
                for k in [k[len(prefix):]] if k.split('.')[0].isdigit())
    layers=[]
    for i in keys:
        w=weights[prefix+str(i)+'.weight']
        while len(layers)<i:
            layers.append(nn.ELU() if prefix=='velocity_estimator.' else nn.ReLU())
        layers.append(nn.Linear(w.shape[1],w.shape[0]))
    if prefix in ('rising_net.','standing_net.current_net.'):
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class HistoryNet(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.current_net=sequential(weights,'standing_net.current_net.')
        w=weights['standing_net.past_projection.weight']
        self.past_projection=nn.Linear(w.shape[1],w.shape[0],bias=False)
    def forward(self,x):
        z=self.current_net[0](x[:,-36:])+self.past_projection(x[:,:-36])
        for layer in list(self.current_net)[1:]:z=layer(z)
        return z


class Actor(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.rising_net=sequential(weights,'rising_net.')
        w=weights['rising_head.weight'];self.rising_head=nn.Linear(w.shape[1],w.shape[0])
        self.standing_net=HistoryNet(weights)
        w=weights['standing_head.weight'];self.standing_head=nn.Linear(w.shape[1],w.shape[0])
        self.velocity_estimator=sequential(weights,'velocity_estimator.')
        self.register_buffer('sensor_indices',torch.tensor(weights['sensor_indices'],dtype=torch.long))
        self.load_state_dict({k:torch.tensor(v) for k,v in weights.items()})
        self.requires_grad_(False)
    def forward(self,obs):
        x=obs.clone().reshape(-1,5,36)
        x[:,:,28:30]=0.
        v=self.velocity_estimator(obs.index_select(1,self.sensor_indices))
        # Match the native actor's estimator stop-gradient, even during BPTT.
        x[:,-1,28:30]=2.*v.detach()[:,:2]
        x=x.reshape(-1,180)
        rise=self.rising_head(self.rising_net(x[:,-36:]))
        stand=self.standing_head(self.standing_net(x))
        return torch.where(x[:,-1:]> .95,stand,rise).clamp(-1,1)


def quat_matrix(q):
    q=F.normalize(q,dim=-1)
    w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],dim=-1).reshape(-1,3,3)


class WorldModel(nn.Module):
    def __init__(self,schema,mean,scale,delta_scale):
        super().__init__();self.schema=schema
        self.register_buffer('mean',torch.tensor(mean))
        self.register_buffer('scale',torch.tensor(scale))
        self.register_buffer('delta_scale',torch.tensor(delta_scale))
        self.register_buffer('indices',torch.tensor(schema['output_indices'],dtype=torch.long))
        self.register_buffer('axes',torch.tensor(schema['chassis_axes'],dtype=torch.float32))
        self.register_buffer('nominal',torch.tensor(schema['nominal_legs'],dtype=torch.float32))
        d=schema['dimensions'];o=len(schema['output_indices'])
        self.net=nn.Sequential(nn.Linear(d+6,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,o+2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def normalize(self,s):return ((s-self.mean)/self.scale).clamp(-20,20)
    def forward(self,s,a):
        g=self.schema;h=g['history'];l=g['latest'];m=g['metrics'];c=g['controller'];nq=g['nq']
        output=self.net(torch.cat([self.normalize(s),a],dim=-1))
        proposed=s.index_select(1,self.indices)+output[:,:-2]*self.delta_scale
        p=s.index_copy(1,self.indices,proposed)
        # Exact bookkeeping/observation heads prevent learning shortcut sensors.
        clock=s[:,c:c+1]+.01
        acquired=torch.maximum(s[:,c+1:c+2],p[:,c+1:c+2].clamp(0,1))
        settled=p[:,c+2:c+3].clamp(0,1)
        q=torch.cat([p[:,:3],F.normalize(p[:,3:7],dim=-1),p[:,7:nq]],-1)
        rotation=quat_matrix(q[:,3:7])@self.axes
        # Keep acquisition probability continuous inside the learned surrogate.
        phase=(.9*clock/2.).clamp(max=.9)*(1-acquired)+acquired
        latest=torch.cat([p[:,l:l+3],-rotation[:,2,:],torch.zeros_like(p[:,l+6:l+8]),
            q[:,g['q_indices'][:4]]-self.nominal,p[:,[nq+i for i in g['v_indices']]]*.05,
            a,p[:,l+24:l+28],torch.zeros_like(p[:,l+28:l+34]),
            (p[:,m+5:m+6]-g['target_hip_height'])*5.,phase],-1)
        springs=torch.stack([q[:,g['spring_q_indices'][j]]+g['spring_zero'][j] for j in range(2)],-1)
        metrics=torch.cat([p[:,m:m+2],springs,p[:,m+4:m+10]],-1)
        control=torch.cat([clock,acquired,settled,s[:,c+3:c+8]],-1)
        result=torch.cat([q,p[:,nq:h],s[:,h+36:l+36],latest,p[:,g['velocity']:m],metrics,control],-1)
        reward=output[:,-2]*20.
        reward=reward+500.*(acquired.flatten()-s[:,c+1]).clamp(min=0)
        termination=torch.sigmoid(output[:,-1])
        reward=reward-80.*termination
        return result,reward,termination


def episode_windows(rows,horizon):
    x=[];a=[];r=[];d=[];offset=[];returns=[]
    for row in rows:
        with np.load(row['path'],allow_pickle=False) as arrays:
            states=arrays['states'];actions=arrays['actions'];rewards=arrays['rewards'];dones=arrays['dones']
        if len(states)!=len(actions)+1:raise ValueError('Broken episode boundary')
        for start in range(0,len(actions)-horizon+1,4):
            # Never cross an episode reset, and keep terminal transitions labelled.
            x.append(states[start:start+horizon+1]);a.append(actions[start:start+horizon])
            r.append(rewards[start:start+horizon]);d.append(dones[start:start+horizon]);offset.append(row['offset_m'])
    if not x:raise ValueError('No continuous replay windows')
    return np.array(x),np.array(a),np.array(r),np.array(d),np.array(offset,np.float32)


def ref_features(s,g):
    l=g['latest'];m=g['metrics']
    return torch.cat([s[:,l+3:l+6],s[:,l+8:l+12],s[:,m:m+2],s[:,m+5:m+6],s[:,m+8:m+9]],-1)


def ref_reward(s,offsets,refs,g):
    which=(offsets[:,None]-refs['offsets'][None]).abs().argmin(1)
    t=(s[:,g['controller']]/.01).round().long().clamp(0,refs['features'].shape[1]-1)
    target=refs['features'][which,t]
    error=(ref_features(s,g)-target)/refs['scales']
    return 3.*torch.exp(-error.square().mean(-1))*(1-s[:,g['controller']+1])


def load_replay(folder,cycle):
    primary=json.loads((folder/'dataset.json').read_text())['episodes']
    rows=[r for r in primary if r['role']=='train']
    validation=[r for r in primary if r['role']=='validation']
    for i in range(cycle):
        path=folder/f'online_{i}.json'
        if path.exists():rows+=json.loads(path.read_text())['episodes']
    assert not set(r['seed'] for r in rows)&set(r['seed'] for r in validation)
    return rows,validation


def evaluate_world(model,rows,g,folder,cycle):
    # Validation contains whole held-out episodes; no overlapping train windows.
    arrays=episode_windows(rows,32)
    rng=np.random.default_rng(620000)
    first=arrays[0][:,0]
    c=g['controller']
    groups=[np.flatnonzero((first[:,c+1]<.5)&(first[:,c]<1.2)),
            np.flatnonzero((first[:,c+1]<.5)&(first[:,c]>=1.2)),
            np.flatnonzero(first[:,c+1]>=.5)]
    ids=np.concatenate([rng.choice(group,min(85,len(group)),replace=False) for group in groups if len(group)])
    truth,actions=arrays[0][ids],arrays[1][ids]
    device=model.mean.device
    state=torch.tensor(truth[:,0],device=device);predictions=[]
    with torch.no_grad():
        for t in range(32):
            state,_,_=model(state,torch.tensor(actions[:,t],device=device))
            predictions.append(state.cpu().numpy())
    pred=np.stack(predictions,axis=1)
    scores={}
    for horizon in (1,8,16,32):
        err=pred[:,horizon-1]-truth[:,horizon]
        m=g['metrics'];qi=g['q_indices'][:4]
        scores[str(horizon)]=dict(base_position_rmse_m=float(np.sqrt(np.mean(err[:,:3]**2))),
            active_joint_rmse_rad=float(np.sqrt(np.mean(err[:,qi]**2))),
            spring_rmse_m=float(np.sqrt(np.mean(err[:,m+2:m+4]**2))),
            roll_rmse_rad=float(np.sqrt(np.mean(err[:,m+6]**2))),
            latest_observation_rmse=float(np.sqrt(np.mean(err[:,g['latest']:g['latest']+36]**2))))
    persistence=truth[:,0]-truth[:,8]
    score=scores['16']
    # Prespecified engineering gate; native closed-chain checks are additional.
    passed=(score['base_position_rmse_m']<.02 and score['active_joint_rmse_rad']<.10
        and score['spring_rmse_m']<.003 and score['roll_rmse_rad']<np.deg2rad(3)
        and np.mean((pred[:,7,:3]-truth[:,8,:3])**2)<np.mean(persistence[:,:3]**2))
    np.savez_compressed(folder/f'world_predictions_{cycle}.npz',truth=truth,predictions=pred,actions=actions)
    report=dict(cycle=cycle,validation_episodes=len(rows),windows=len(ids),horizons=scores,
        prediction_gate_passed=bool(passed),native_closure_gate_pending=True,
        balanced_phase_windows=[min(85,len(group)) for group in groups],
        gate_horizon=16,
        persistence_base_8_rmse_m=float(np.sqrt(np.mean(persistence[:,:3]**2))))
    json_write(folder/f'world_validation_{cycle}.json',report)
    return report


def run(folder,cycle,fit_steps,policy_steps):
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(208345)
    if not torch.cuda.is_available():raise RuntimeError('Pilot GPU unavailable')
    device=torch.device('cuda')
    g=json.loads((folder/'schema.json').read_text());c=g['controller'];h=g['history']
    weights=dict(np.load(folder/'parent_actor.npz',allow_pickle=False))
    actor=Actor(weights).to(device);parent=Actor(weights).to(device)
    parity=np.load(folder/'actor_parity.npz')
    with torch.no_grad():
        err=(actor(torch.tensor(parity['observations'],device=device)).cpu().numpy()-parity['actions'])
    parity_error=float(np.abs(err).max())
    if parity_error>1e-5:raise RuntimeError('GPU actor differs from native actor: '+str(parity_error))
    rows,validation=load_replay(folder,cycle)
    arrays=episode_windows(rows,8)
    states,actions,rewards,dones,offsets=[torch.tensor(a,device=device) for a in arrays]
    all_states=np.concatenate([np.load(r['path'])['states'] for r in rows])
    mean=all_states.mean(0);scale=np.maximum(all_states.std(0),.05)
    indices=g['output_indices']
    delta=(arrays[0][:,1:,indices]-arrays[0][:,:-1,indices]).reshape(-1,len(indices))
    delta_scale=np.maximum(delta.std(0),.001)
    model=WorldModel(g,mean,scale,delta_scale).to(device)
    if cycle:
        previous=torch.load(folder/f'world_{cycle-1}.pt',map_location=device,weights_only=True)
        model.load_state_dict(previous['model'])
        actor.load_state_dict(previous['actor'])
    opt=torch.optim.Adam(model.parameters(),lr=2e-4)
    loss_trace=[]
    n=len(states)
    rising=torch.nonzero(states[:,0,c+1]<.5,as_tuple=False).flatten()
    standing=torch.nonzero(states[:,0,c+1]>=.5,as_tuple=False).flatten()
    for step in range(fit_steps):
        ids=torch.cat([group[torch.randint(len(group),(64,),device=device)]
                       for group in (rising,standing) if len(group)])
        p=states[ids,0];loss=0.
        # Progress from 1 to 8-step autoregressive fitting.
        horizon=1 if step<100 and cycle==0 else (4 if step<250 and cycle==0 else 8)
        for t in range(horizon):
            p,r,d=model(p,actions[ids,t])
            target=states[ids,t+1]
            loss=loss+((p[:,indices]-target[:,indices])/model.scale[indices]).square().mean()
            # Give kinematics/near-limit position precision a meaningful weight.
            loss=loss+.05*((p[:,:g['nq']]-target[:,:g['nq']])/.01).square().mean()
            loss=loss+.1*((r-rewards[ids,t])/20.).square().mean()
            loss=loss+.1*F.binary_cross_entropy(d.clamp(1e-6,1-1e-6),dones[ids,t])
        loss=loss/horizon
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%100==0:
            value=float(loss.detach());loss_trace.append(dict(step=step,loss=value))
            print(json.dumps(dict(world_cycle=cycle,update=step,loss=value,seconds=time.monotonic()-started)),flush=True)
    validation_report=evaluate_world(model,validation,g,folder,cycle)
    torch.save(dict(model=model.state_dict(),actor=actor.state_dict()),folder/f'world_{cycle}.pt')
    json_write(folder/f'world_fit_{cycle}.json',dict(seconds=time.monotonic()-started,
        fit_steps=fit_steps,training_episodes=len(rows),validation_episodes=len(validation),
        windows=n,actor_parity_error=parity_error,loss=loss_trace,device=str(device),
        peak_allocated_MB=torch.cuda.max_memory_allocated()/2**20,torch=torch.__version__))
    if not policy_steps:return
    # Require a CPU native geometry check before every policy optimization.
    native=json.loads((folder/f'world_geometry_{cycle}.json').read_text())
    if not validation_report['prediction_gate_passed'] or not native['passed']:
        raise RuntimeError('World model failed the prespecified validation gate')


def optimize(folder,cycle,updates):
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(208345+cycle)
    device='cuda';g=json.loads((folder/'schema.json').read_text());c=g['controller'];h=g['history']
    rows,_=load_replay(folder,cycle)
    weights=dict(np.load(folder/'parent_actor.npz',allow_pickle=False))
    actor=Actor(weights).to(device);parent=Actor(weights).to(device)
    payload=torch.load(folder/f'world_{cycle}.pt',map_location=device,weights_only=True)
    ms=payload['model']
    model=WorldModel(g,ms['mean'].cpu().numpy(),ms['scale'].cpu().numpy(),ms['delta_scale'].cpu().numpy()).to(device)
    model.load_state_dict(ms);model.requires_grad_(False)
    actor.load_state_dict(payload['actor'])
    actor.rising_net.requires_grad_(True);actor.rising_head.requires_grad_(True)
    parameters=[p for p in actor.parameters() if p.requires_grad]
    opt=torch.optim.Adam(parameters,lr=1e-6)
    refs={k:torch.tensor(v,device=device) for k,v in dict(np.load(folder/'references.npz')).items()}
    xs=[];offs=[];mc=[]
    for row in rows:
        a=np.load(row['path']);s=a['states'];reward=a['rewards']
        off=torch.full((len(s)-1,),row['offset_m'],device=device)
        with torch.no_grad():bonus=ref_reward(torch.tensor(s[1:],device=device),off,refs,g).cpu().numpy()
        ret=np.zeros(len(reward),np.float32);value=0.
        for i in reversed(range(len(reward))):value=float(reward[i]+bonus[i])*.02+.999*value;ret[i]=value
        keep=np.arange(0,len(reward),4)
        xs.append(s[keep]);offs.append(np.full(len(keep),row['offset_m'],np.float32));mc.append(ret[keep])
    s=torch.tensor(np.concatenate(xs),device=device);offsets=torch.tensor(np.concatenate(offs),device=device)
    targets=torch.tensor(np.concatenate(mc),device=device)
    critic=nn.Sequential(nn.Linear(g['dimensions'],128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,1)).to(device)
    critic_opt=torch.optim.Adam(critic.parameters(),lr=1e-3)
    # A critic initialized from real simulator returns, not model-generated success.
    for _ in range(800):
        ids=torch.randint(len(s),(256,),device=device)
        value=critic(model.normalize(s[ids])).flatten()
        loss=F.smooth_l1_loss(value,targets[ids])
        critic_opt.zero_grad(set_to_none=True);loss.backward();critic_opt.step()
    critic.requires_grad_(False)
    rising=torch.nonzero((s[:,c+1]<.5)&(s[:,c]<6.),as_tuple=False).flatten()
    if not len(rising):raise RuntimeError('Missing rising replay starts')
    history=[]
    for update in range(updates):
        ids=rising[torch.randint(len(rising),(128,),device=device)]
        p=s[ids];off=offsets[ids];survival=torch.ones(len(ids),device=device);objective=0.;anchor=0.
        replay_features=[];imagined_rewards=[];imagined_dones=[]
        for step in range(16):
            replay_features.append(model.normalize(p).detach())
            obs=p[:,h:h+180]
            a=actor(obs)
            with torch.no_grad():prior=parent(obs)
            anchor=anchor+((a-prior)/torch.tensor([.03]*4+[.1]*2,device=device)).square().mean()
            p,r,d=model(p,a)
            r=r+ref_reward(p,off,refs,g)
            imagined_rewards.append((r*.02).detach());imagined_dones.append(d.detach())
            objective=objective+(.999**step)*survival*r*.02
            survival=survival*(1-d)
        objective=objective+.999**16*survival*critic(model.normalize(p)).flatten()
        loss=-objective.mean()+.1*anchor/16
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=float(nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
        opt.step()
        # TD(lambda) critic targets on the same short imagined rollout.
        with torch.no_grad():
            features=torch.stack(replay_features)
            values=critic(features).squeeze(-1)
            boot=critic(model.normalize(p.detach())).flatten()
            gae_targets=[];tail=boot
            for step in reversed(range(16)):
                next_value=boot if step==15 else values[step+1]
                tail=imagined_rewards[step]+.999*(1-imagined_dones[step])*(.05*next_value+.95*tail)
                gae_targets.append(tail)
            gae_targets=torch.stack(gae_targets[::-1]).flatten()
            critic_inputs=features.flatten(0,1)
        critic.requires_grad_(True)
        for _ in range(2):
            shuffled=torch.randperm(len(critic_inputs),device=device)
            for batch in shuffled.split(512):
                value=critic(critic_inputs[batch]).flatten()
                critic_loss=F.smooth_l1_loss(value,gae_targets[batch])
                critic_opt.zero_grad(set_to_none=True);critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(),1.);critic_opt.step()
        critic.requires_grad_(False)
        if update%10==0:
            history.append(dict(update=update,loss=float(loss.detach()),gradient_norm=norm))
            print(json.dumps(dict(policy_cycle=cycle,**history[-1])),flush=True)
    frozen=[]
    original={k:torch.tensor(v,device=device) for k,v in weights.items()}
    for k,v in actor.state_dict().items():
        if not k.startswith(('rising_net.','rising_head.')) and not torch.equal(v,original[k]):frozen.append(k)
    if frozen:raise RuntimeError('Protected actor tensor changed: '+str(frozen))
    np.savez(folder/f'actor_cycle_{cycle}.npz',**{k:v.detach().cpu().numpy() for k,v in actor.state_dict().items()})
    torch.save(dict(model=model.state_dict(),actor=actor.state_dict()),folder/f'world_{cycle}.pt')
    json_write(folder/f'policy_fit_{cycle}.json',dict(seconds=time.monotonic()-started,updates=updates,
        horizon=16,critic='real-replay return warm start then imagined TD(lambda=.95); frozen during each policy gradient',
        frozen_tensors_identical=True,history=history,peak_allocated_MB=torch.cuda.max_memory_allocated()/2**20))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['fit','optimize','validate'])
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--cycle',type=int,default=0)
    p.add_argument('--fit-steps',type=int,default=1600);p.add_argument('--updates',type=int,default=80)
    args=p.parse_args()
    if args.mode=='fit':run(args.folder,args.cycle,args.fit_steps,0)
    elif args.mode=='optimize':optimize(args.folder,args.cycle,args.updates)
    else:
        torch.set_num_threads(1)
        schema=json.loads((args.folder/'schema.json').read_text())
        payload=torch.load(args.folder/f'world_{args.cycle}.pt',map_location='cuda',weights_only=True)['model']
        model=WorldModel(schema,payload['mean'].cpu().numpy(),payload['scale'].cpu().numpy(),payload['delta_scale'].cpu().numpy()).cuda()
        model.load_state_dict(payload)
        _,rows=load_replay(args.folder,args.cycle)
        print(json.dumps(evaluate_world(model,rows,schema,args.folder,args.cycle)),flush=True)


if __name__=='__main__':main()
