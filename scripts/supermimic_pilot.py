"""Native MuJoCo data, reference PPO, and paired evaluation for a bounded MBRL pilot.

This is a small SuperMimic-inspired adaptation, not an official reproduction.
No teacher controller or world model is used in native policy evaluation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'scripts')]
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_training import load_warm_start
from climber_legged_rl.spring_geometry import eye_center_distance
from climber_legged_rl.velocity_estimation import VelocityEstimationPPO
from prepare_spring208_phase3 import PARENT, PARENT_SHA, sha

CFG = ROOT/'configs/robot/cad_spring208_phase3_train.yaml'
SEEDS = dict(data=510000, validation=520000, selection=530000, test=540000)
DT = .01
REFERENCE_WEIGHT = 3.
REF_SCALE = np.array([.1]*3 + [.2]*4 + [.02]*3 + [.0872665], np.float32)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def make_env(split='train', offset=None):
    path = ROOT/f'configs/robot/cad_spring208_phase3_{split}.yaml'
    config = load_config(path)
    env = make_mujoco_env(ROOT/config['experiment']['output_directory']/'robot_experiment.xml', path)
    if offset is not None:
        env.pose_bank = [p for p in env.pose_bank if abs(p['rearward_offset_m']-offset)<1e-10]
        if not env.pose_bank:
            raise ValueError('Unknown prepared geometry')
    env.push.config = dict(env.push.config, episode_probability=0.)
    return env


def layout(env):
    nq, nv = env.model.nq, env.model.nv
    h = nq+nv
    v = h+180
    m = v+3
    c = m+10
    return dict(nq=nq, nv=nv, history=h, velocity=v, metrics=m, controller=c,
                dimensions=c+8, latest=h+144,
                output_indices=list(range(h))+list(range(h+144,v+3))+list(range(m,m+10))+[c+1,c+2],
                q_indices=env.q_indices.tolist(), v_indices=env.v_indices.tolist(),
                nominal_legs=env.nominal_legs.tolist(),chassis_axes=env.chassis_axes.tolist(),
                target_hip_height=float(env.target_hip_height),
                spring_q_indices=[int(env.model.jnt_qposadr[env.model.joint(s+'_gas_slide_joint').id])
                                  for s in ('left','right')],
                spring_zero=[float(np.linalg.norm(env.model.body(s+'_gas_rod').pos)) for s in ('left','right')])


def pack(env, obs, schema):
    task = env.unwrapped
    angular, velocity, _ = task._kinematics()
    from climber_legged_rl.stance_geometry import virtual_leg_geometry
    from climber_legged_rl.mujoco_model import closure_error
    shape = virtual_leg_geometry(task.model, task.data, task.motor_midpoints, task.wheel_centers, task.chassis_axes)
    pitch, roll = task._attitude()
    displacement, heading = task._hold_errors()
    metrics = [*shape['side_view_distance_m'],
        *[eye_center_distance(task.model,task.data,s) for s in ('left','right')],
        np.linalg.norm(closure_error(task.model,task.data),axis=1).max(),
        task.data.xpos[task.hip_ids,2].mean(),roll,pitch,heading,np.linalg.norm(displacement)]
    control = [task.steps*task.dt,float(task.acquired),task.settled_steps*task.dt,
               task.reset_hip_height,task.reset_length_m,*task.hold_position,task.hold_heading]
    state = np.r_[task.data.qpos, task.data.qvel, obs[:180], velocity, metrics, control].astype(np.float32)
    if state.shape!=(schema['dimensions'],) or not np.isfinite(state).all():
        raise RuntimeError('Invalid full training state')
    return state


def reference_features(state, schema):
    l, m = schema['latest'], schema['metrics']
    return np.concatenate([state[...,l+3:l+6], state[...,l+8:l+12],
                           state[...,m:m+2],state[...,m+5:m+6],state[...,m+8:m+9]],axis=-1)


def reference_reward(state, references, offset, schema):
    i = int(np.argmin(abs(references['offsets']-offset)))
    t = min(int(round(float(state[schema['controller']])/DT)),references['features'].shape[1]-1)
    error = (reference_features(state,schema)-references['features'][i,t])/REF_SCALE
    return REFERENCE_WEIGHT*float(np.exp(-np.mean(error**2)))*(1.-float(state[schema['controller']+1]))


def collect_episode(job):
    checkpoint, offset, seed, noise, destination, role = job
    torch.set_num_threads(1)
    env = make_env(offset=offset)
    policy = PPO.load(checkpoint,device='cpu')
    rng = np.random.default_rng(seed)
    schema = layout(env)
    states=[]; actions=[]; rewards=[]; dones=[]
    try:
        obs, reset = env.reset(seed=seed,options={'reset_mode':'folded'})
        states.append(pack(env,obs,schema))
        while True:
            action = policy.predict(obs,deterministic=True)[0]
            if not env.acquired and noise:
                action = np.clip(action+rng.normal(size=6)*np.array([.0015]*4+[.015]*2)*noise,-1,1)
            executed = action.astype(np.float32)
            obs,reward,done,truncated,info = env.step(executed)
            actions.append(executed.copy()); rewards.append(reward); dones.append(done)
            states.append(pack(env,obs,schema))
            if done or truncated:
                break
        path = Path(destination)/f'{role}_{seed}.npz'
        np.savez_compressed(path,states=np.array(states),actions=np.array(actions),
            rewards=np.array(rewards,np.float32),dones=np.array(dones,np.float32))
        return dict(path=str(path),offset_m=offset,seed=seed,noise=noise,role=role,
            samples=len(actions),success=bool(info['combined_success']),
            acquired=bool(env.acquired),getup_seconds=info['getup_seconds'],
            reason=info['termination_reason'] or 'timeout',sha256=sha(path),
            warnings=int(sum(env.data.warning.number)),model_sha256=sha(ROOT/load_config(CFG)['experiment']['output_directory']/'robot_experiment.xml'))
    finally:
        env.close()


def collect(folder, checkpoint=PARENT, cycle=-1):
    folder = Path(folder)
    data = folder/'data';data.mkdir(exist_ok=True)
    offsets = load_config(CFG)['getup']['rearward_offsets_m']
    if cycle<0:
        jobs=[(str(checkpoint),o,SEEDS['data']+i*100+j,(0.,0.,0.,0.,1.,3.)[j],str(data),'train')
              for i,o in enumerate(offsets) for j in range(6)]
        jobs += [(str(checkpoint),o,SEEDS['validation']+i*100+j,0.,str(data),'validation')
                 for i,o in enumerate(offsets) for j in range(2)]
    else:
        jobs=[(str(checkpoint),o,550000+cycle*10000+i*100+j,float(j),str(data),'online')
              for i,o in enumerate(offsets) for j in range(2)]
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        rows=list(pool.map(collect_episode,jobs))
    report=folder/('dataset.json' if cycle<0 else f'online_{cycle}.json')
    write_json(report,dict(seconds=time.monotonic()-started,episodes=rows,
        transitions=sum(r['samples'] for r in rows),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
    return report


def prepare(folder):
    folder=Path(folder);folder.mkdir(exist_ok=False,parents=True)
    if sha(PARENT)!=PARENT_SHA:
        raise RuntimeError('Unexpected parent checkpoint')
    for split in ('train','test','nominal'):
        validate_artifacts(ROOT/f'configs/robot/cad_spring208_phase3_{split}.yaml')
    env=make_env();schema=layout(env);env.close()
    write_json(folder/'schema.json',schema)
    policy=PPO.load(PARENT,device='cpu').policy
    arrays={k:v.detach().cpu().numpy() for k,v in policy.mlp_extractor.policy_net.state_dict().items()}
    np.savez(folder/'parent_actor.npz',**arrays)
    parent_cfg=load_config(CFG)
    source=ROOT/parent_cfg['experiment']['output_directory']
    write_json(folder/'protocol.json',dict(method='SuperMimic-inspired small-model pilot, not official reproduction',
        created_utc=time.time(),parent=str(PARENT),parent_sha256=sha(PARENT),
        physical_model_sha256=sha(source/'robot_experiment.xml'),policy_hz=100,physics_hz=1000,
        seeds=SEEDS,training_offsets_m=parent_cfg['getup']['rearward_offsets_m'],
        test_offsets_m=load_config(ROOT/'configs/robot/cad_spring208_phase3_test.yaml')['getup']['rearward_offsets_m'],
        target_fresh_success_fraction=.95,target_nominal_success_fraction=1.,
        physics_and_success_criteria_unchanged=True,standing_and_estimator_frozen=True,
        max_native_transitions_per_arm=200704,reference_weight=REFERENCE_WEIGHT,
        timing='Include dataset collection, model fit, online collection, optimization and selection; also report training-only timing.',
        small_sample_pilot=True,minimum_training_seeds_for_confirmatory_claim=3,
        final_test_not_used_for_selection=True,auto_promote=False))
    env=make_env()
    obs,_=env.reset(seed=123,options={'reset_mode':'folded'})
    env.close()
    rng=np.random.default_rng(992)
    samples=np.tile(obs[:180],(64,1));samples+=rng.normal(0,.01,samples.shape).astype(np.float32)
    samples[:,-1]=.5
    padded=np.pad(samples,((0,0),(0,3)))
    with torch.no_grad():
        outputs=policy.mlp_extractor.policy_net(torch.tensor(padded)).clamp(-1,1).numpy()
    np.savez(folder/'actor_parity.npz',observations=samples,actions=outputs)


def references(folder):
    folder=Path(folder);schema=json.loads((folder/'schema.json').read_text())
    report=json.loads((folder/'dataset.json').read_text())
    eligible=[r for r in report['episodes'] if r['role']=='train' and r['noise']==0 and r['success']]
    if not eligible:
        raise RuntimeError('No unassisted successful PPO trajectory for reference')
    offsets=np.array(load_config(CFG)['getup']['rearward_offsets_m'])
    features=[];sources=[]
    for o in offsets:
        candidates=sorted(eligible,key=lambda r:(abs(r['offset_m']-o),r['getup_seconds'],r['seed']))
        selected=candidates[0]
        s=np.load(selected['path'])['states']
        f=reference_features(s,schema)
        f=np.concatenate([f,np.repeat(f[-1:],max(0,2001-len(f)),axis=0)])[:2001]
        features.append(f);sources.append(selected)
    np.savez(folder/'references.npz',offsets=offsets,features=np.array(features),scales=REF_SCALE)
    write_json(folder/'reference_provenance.json',dict(sources=sources,
        unassisted_success_only=True,test_and_validation_excluded=True,
        all_failures_retained_for_dynamics=True,sha256=sha(folder/'references.npz')))


def evaluate_job(job):
    from assess_spring208_phase3 import case
    return case(job)


def evaluate(checkpoint,split='selection',label='candidate'):
    if split=='selection':
        offsets=load_config(CFG)['getup']['rearward_offsets_m']
        specs=[('train',o,'folded',False,SEEDS['selection']+j) for o in (offsets[0],offsets[3],offsets[-1]) for j in range(4)]
        specs += [('nominal',None,'folded',False,SEEDS['selection']+100+j) for j in range(4)]
    else:
        offsets=load_config(ROOT/'configs/robot/cad_spring208_phase3_test.yaml')['getup']['rearward_offsets_m']
        specs=[('test',o,'folded',False,SEEDS['test']+j) for o in offsets for j in range(16)]
        specs += [('nominal',None,'folded',False,SEEDS['test']+100+j) for j in range(8)]
        specs += [('nominal',None,'standing',push,SEEDS['test']+200+j) for push in (False,True) for j in range(8)]
    jobs=[(label,str(checkpoint),*spec) for spec in specs]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        rows=list(pool.map(evaluate_job,jobs))
    nominal=[r for r in rows if r['split']=='nominal' and r['mode']=='folded']
    folded=[r for r in rows if r['mode']=='folded' and r['split']!='nominal']
    standing=[r for r in rows if r['mode']=='standing']
    summary={}
    for key,group in [('folded',folded),('nominal',nominal),('standing',standing)]:
        if not group:continue
        times=[r['getup_seconds'] for r in group if r['success'] and r['getup_seconds'] is not None]
        summary[key]=dict(count=len(group),success=sum(r['success'] for r in group),
            median_getup_seconds=float(np.median(times)) if times else None,
            warnings=sum(sum(r['warnings']) for r in group),
            max_upper_stop_penetration_mm=max(max(r['upper_stop_penetration_mm']) for r in group),
            max_closure_m=max(r['max_closure_m'] for r in group),
            max_heading_deg=max(r['max_heading_deg'] for r in group))
    key=[sum(r['success'] for r in nominal),min(sum(r['success'] for r in folded if r['offset_m']==o)
        for o in sorted(set(r['offset_m'] for r in folded))),sum(r['success'] for r in folded)]
    return dict(summary=summary,key=key,episodes=rows,checkpoint=str(checkpoint),sha256=sha(checkpoint))


def import_actor(folder,weights,destination):
    model=VelocityEstimationPPO.load(PARENT,device='cpu')
    original={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
    data=np.load(weights,allow_pickle=False)
    actor=model.policy.mlp_extractor.policy_net
    current=actor.state_dict()
    changed=[]
    for key in data.files:
        if not key.startswith(('rising_net.','rising_head.')):
            if not np.array_equal(current[key].numpy(),data[key]):
                raise RuntimeError('Protected GPU actor tensor changed: '+key)
            continue
        current[key]=torch.tensor(data[key]);changed.append(key)
    actor.load_state_dict(current)
    for key,value in model.policy.state_dict().items():
        if not key.startswith(('mlp_extractor.policy_net.rising_net.','mlp_extractor.policy_net.rising_head.')):
            if not torch.equal(value,original[key]):raise RuntimeError('Protected tensor changed')
    model.policy.freeze_rising_actor=False;model.policy_kwargs['freeze_rising_actor']=False
    model.save(destination)
    return changed


def geometry_check(folder,cycle):
    import mujoco
    from climber_legged_rl.mujoco_model import closure_error
    folder=Path(folder)
    predictions=np.load(folder/f'world_predictions_{cycle}.npz')
    schema=json.loads((folder/'schema.json').read_text())
    env=make_env();nq=schema['nq'];m=schema['metrics']
    report={}
    try:
        for horizon in (1,8,16,32):
            errors=[];springs=[];heads=[]
            for state,truth in zip(predictions['predictions'][:,horizon-1],predictions['truth'][:,horizon]):
                env.data.qpos[:]=state[:nq]
                mujoco.mj_forward(env.model,env.data)
                errors.append(float(np.linalg.norm(closure_error(env.model,env.data),axis=1).max()))
                eye=np.array([eye_center_distance(env.model,env.data,s) for s in ('left','right')])
                springs.append(float(np.max(abs(eye-truth[m+2:m+4]))))
                heads.append(float(np.max(abs(eye-state[m+2:m+4]))))
            report[str(horizon)]=dict(closure_p95_m=float(np.quantile(errors,.95)),
                closure_max_m=max(errors),spring_prediction_p95_m=float(np.quantile(springs,.95)),
                spring_head_geometry_p95_m=float(np.quantile(heads,.95)))
    finally:env.close()
    score=report['16']
    passed=(score['closure_p95_m']<.003 and score['spring_prediction_p95_m']<.005
            and score['spring_head_geometry_p95_m']<.003)
    result=dict(passed=passed,horizons=report,source='Native MuJoCo forward kinematics of predicted qpos; not learned closure head')
    write_json(folder/f'world_geometry_{cycle}.json',result)
    return result


class ReferenceEnv(gym.Wrapper):
    def __init__(self,env,reference_path):
        super().__init__(env);self.schema=layout(env)
        self.refs=dict(np.load(reference_path,allow_pickle=False))
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        resetlegs=self.env.data.qpos[self.env.q_indices[:4]]
        self.offset=min(self.env.pose_bank,key=lambda p:np.linalg.norm(np.array(p['leg_reference_rad'])-resetlegs))['rearward_offset_m']
        return obs,info
    def step(self,action):
        obs,r,d,t,info=self.env.step(action)
        bonus=reference_reward(pack(self.env,obs,self.schema),self.refs,self.offset,self.schema)
        return obs,r+bonus,d,t,dict(info,reference_reward=bonus)


def ppo_factory(seed,reference_path):
    def factory():
        env=make_env()
        if reference_path:env=ReferenceEnv(env,reference_path)
        env.reset(seed=seed)
        return env
    return factory


class PilotPPOCallback(BaseCallback):
    def __init__(self,folder,initial,shared_seconds,interval=65536):
        super().__init__();self.folder=Path(folder);self.best=initial['key'];self.rows=[]
        self.started=time.monotonic();self.shared_seconds=shared_seconds;self.interval=interval;self.next_step=interval
    def assess_current(self):
        cp=self.folder/f'step_{self.num_timesteps}.zip';self.model.save(cp)
        result=evaluate(cp)
        if result['key']>self.best:
            self.best=result['key'];self.model.save(self.folder/'selected_policy.zip')
        self.rows.append(dict(steps=self.num_timesteps,seconds=time.monotonic()-self.started+self.shared_seconds,
                              selection=result))
        write_json(self.folder/'curve.json',self.rows)
        print(json.dumps(dict(ppo_steps=self.num_timesteps,key=result['key'],best=self.best)),flush=True)
    def _on_step(self):
        if self.num_timesteps<self.next_step:return True
        self.next_step+=self.interval
        self.assess_current()
        return True


def train_ppo(folder,reference=False,steps=200704,seed=208345):
    setup_start=time.monotonic()
    folder=Path(folder);arm=folder/('ppo_reference' if reference else 'ppo_current');arm.mkdir(exist_ok=False)
    dataset=json.loads((folder/'dataset.json').read_text())
    shared_seconds=dataset['seconds'] if reference else 0.
    if reference:
        steps=min(steps,((200704-dataset['transitions'])//2048)*2048)
    config=load_config(CFG);settings=dict(config['training'])
    settings.update(n_steps=512,batch_size=256,n_epochs=2,learning_rate=5e-7,
        freeze_rising_actor=False,freeze_standing_actor=True,freeze_action_std=True,
        reset_optimizer_on_warm_start=True)
    vector=SubprocVecEnv([ppo_factory(seed+i,str(folder/'references.npz') if reference else None) for i in range(4)],start_method='spawn')
    try:
        model,_,_=load_warm_start(PARENT,vector,settings,'cpu',seed)
        model.configure_estimator(dict(config['training']['velocity_estimator'],trainable=False))
        # Same stochastic action distribution as the current PPO configuration.
        with torch.no_grad():model.policy.log_std.copy_(torch.log(torch.tensor(config['training']['action_std'])))
        model.save(arm/'selected_policy.zip')
        initial=evaluate(PARENT);write_json(arm/'initial_selection.json',initial)
        interval=max(2048,((steps+3*2048-1)//(3*2048))*2048)
        callback=PilotPPOCallback(arm,initial,shared_seconds,interval)
        start=time.monotonic()
        model.learn(total_timesteps=steps,callback=callback,reset_num_timesteps=True,progress_bar=False)
        if not callback.rows or callback.rows[-1]['steps']!=model.num_timesteps:
            callback.assess_current()
        model.save(arm/'final_policy.zip')
        write_json(arm/'training.json',dict(seed=seed,steps=model.num_timesteps,
            seconds=time.monotonic()-start,shared_data_seconds=shared_seconds,
            inclusive_seconds=time.monotonic()-setup_start+shared_seconds,
            reference=reference,total_native_transitions=model.num_timesteps+(dataset['transitions'] if reference else 0),
            initial_selection=initial['key'],best_key=callback.best))
    finally:vector.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','collect','references','evaluate','import','ppo','geometry','schema'])
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=PARENT)
    parser.add_argument('--cycle',type=int,default=-1)
    parser.add_argument('--split',choices=['selection','test'],default='selection')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--reference',action='store_true')
    parser.add_argument('--steps',type=int,default=200704)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.mode=='prepare':prepare(args.folder)
    elif args.mode=='collect':collect(args.folder,args.checkpoint,args.cycle)
    elif args.mode=='references':references(args.folder)
    elif args.mode=='evaluate':write_json(args.output,evaluate(args.checkpoint,args.split))
    elif args.mode=='import':import_actor(args.folder,args.weights,args.output)
    elif args.mode=='ppo':train_ppo(args.folder,args.reference,args.steps)
    elif args.mode=='geometry':geometry_check(args.folder,args.cycle if args.cycle>=0 else 0)
    elif args.mode=='schema':
        env=make_env();write_json(args.folder/'schema.json',layout(env));env.close()


if __name__=='__main__':main()
