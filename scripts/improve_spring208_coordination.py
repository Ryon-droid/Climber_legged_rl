"""Training-only corrective demonstrations; native student receives no runtime teacher."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'scripts')]
import numpy as np
import torch
from stable_baselines3 import PPO
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.velocity_estimation import VelocityEstimationPPO

PARENT=ROOT/'outputs/mujoco_runs/spring208_phase1_20261106_1303/best/best_model.zip'
PARENT_SHA='28a44dcfc15928016ec809137bade5b127a31a5f2cc880ee96ca590b5bf4c454'
CFG=ROOT/'configs/robot/cad_spring208_phase3_train.yaml'
OUT=ROOT/'outputs/spring208_coordination_fix'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def teacher_action(parent,env,obs,recipe,ready):
    task=env.unwrapped
    if task.acquired:return parent.predict(obs,deterministic=True)[0],ready
    action=parent.predict(obs,deterministic=True)[0]
    # Make a training demonstration more coordinated without fixing both legs
    # to identical coordinates: feedback correction remains available in the actor.
    sym=recipe['symmetry']*max(0.,1.-task.steps*task.dt/recipe['symmetry_seconds'])
    mean=(action[:2]+action[2:4])/2
    action[:2]=(1.-sym)*action[:2]+sym*mean
    action[2:4]=(1.-sym)*action[2:4]+sym*mean
    from climber_legged_rl.stance_geometry import virtual_leg_geometry
    shape=virtual_leg_geometry(task.model,task.data,task.motor_midpoints,task.wheel_centers,task.chassis_axes)
    pitch,roll=task._attitude()
    ready=ready or (task.steps*task.dt>=recipe['handover_seconds']
        and max(shape['side_view_distance_m'])<.19 and abs(pitch)<.22 and abs(roll)<.12)
    if ready:
        # Ask the already learned standing actor for its target, then convert
        # the existing residual PD map into the rising actor's incremental map.
        teacher_obs=obs.copy()
        teacher_obs[:180].reshape(5,36)[:,-1]=1.
        stand=parent.predict(teacher_obs,deterministic=True)[0]
        target=np.clip(task.nominal_legs+stand[:4]*task.control['leg_action_scale_rad'],task.leg_low,task.leg_high)
        action[:4]=(target-task.data.qpos[task.q_indices[:4]])/task.dynamic['incremental_target_scale_rad']
        action[4:]=stand[4:]
    return np.clip(action,-1.,1.),ready


def rollout(job):
    recipe,offset,seed,checkpoint,mix,collect,split,mode,push=job
    torch.set_num_threads(1)
    cfg=ROOT/f'configs/robot/cad_spring208_phase3_{split}.yaml'
    config=load_config(cfg)
    env=make_mujoco_env(ROOT/config['experiment']['output_directory']/'robot_experiment.xml',cfg)
    if offset is not None:
        env.pose_bank=[p for p in env.pose_bank if abs(p['rearward_offset_m']-offset)<1e-10]
        if not env.pose_bank:raise RuntimeError('Missing prepared pose')
    env.unwrapped.push.config=dict(env.push.config,episode_probability=float(push))
    parent=PPO.load(PARENT,device='cpu') if recipe is not None else None
    student=PPO.load(checkpoint,device='cpu') if checkpoint else None
    xs=[];ys=[];ready=False;handover=None
    max_roll=0.;max_lr=0.
    try:
        obs,reset=env.reset(seed=seed,options={'reset_mode':mode})
        while True:
            if recipe is not None:
                label,ready=teacher_action(parent,env,obs,recipe,ready)
                if ready and handover is None:handover=env.steps*env.dt
                if collect and not env.acquired:
                    xs.append(obs.copy());ys.append(label.copy())
                action=label if student is None else mix*label+(1.-mix)*student.predict(obs,deterministic=True)[0]
            else:action=student.predict(obs,deterministic=True)[0]
            obs,_,done,truncated,info=env.step(action)
            _,roll=env._attitude()
            max_roll=max(max_roll,abs(float(np.degrees(roll))))
            q=env.data.qpos[env.q_indices[:4]]
            max_lr=max(max_lr,float(np.degrees(np.max(abs(q[:2]-q[2:])))))
            if done or truncated:
                row=dict(recipe=recipe,offset_m=offset,seed=seed,split=split,mode=mode,push=push,
                    success=bool(info['combined_success']),acquired=bool(env.acquired),
                    getup_seconds=info['getup_seconds'],seconds=env.steps*env.dt,
                    reason=info['termination_reason'] or 'timeout',handover_seconds=handover,
                    final_heading_deg=float(np.degrees(info['heading_error_rad'])),
                    final_displacement_m=info['planar_displacement_m'],
                    max_roll_deg=max_roll,max_lr_joint_difference_deg=max_lr,
                    warnings=int(sum(env.data.warning.number)),peak_torque_Nm=info['peak_torque_Nm'])
                return row,xs,ys
    finally:env.close()


def parallel(jobs):
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        return list(pool.map(rollout,jobs))


def probe():
    config=load_config(CFG)
    offsets=config['getup']['rearward_offsets_m']
    recipes=[dict(symmetry=sym,symmetry_seconds=1.,handover_seconds=seconds)
        for sym in (0.,.5,1.) for seconds in (.65,1.02,1.3)]
    jobs=[(recipe,offset,seed,None,1.,False,'train','folded',False)
        for recipe in recipes for offset,seed in [(offsets[0],451000),(offsets[0],451001),
        (offsets[3],451002),(offsets[-1],451003)]]
    result=parallel(jobs)
    rows=[r[0] for r in result]
    ranking=[]
    for recipe in recipes:
        group=[r for r in rows if r['recipe']==recipe]
        ranking.append(dict(recipe=recipe,success=sum(r['success'] for r in group),count=len(group),
            max_roll_deg=max(r['max_roll_deg'] for r in group)))
    ranking.sort(key=lambda r:(r['success'],-r['max_roll_deg']),reverse=True)
    OUT.mkdir(exist_ok=True)
    report=dict(parent_sha256=sha(PARENT),ranking=ranking,episodes=rows,
        teacher_only=True,note='Probe is not an autonomous student result')
    (OUT/'teacher_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def save(model,folder,name='final_policy.zip'):
    model.policy.freeze_rising_actor=False
    model.policy_kwargs['freeze_rising_actor']=False
    model.save(folder/name)


def distill(run_name):
    started=time.monotonic()
    probe_report=json.loads((OUT/'teacher_probe.json').read_text())
    if probe_report['parent_sha256']!=sha(PARENT):raise RuntimeError('Teacher probe parent changed')
    best_teacher=probe_report['ranking'][0]
    if best_teacher['success']<best_teacher['count']:raise RuntimeError('No teacher solved all probe cases')
    recipe=best_teacher['recipe']
    config=load_config(CFG)
    offsets=config['getup']['rearward_offsets_m']
    folder=ROOT/'outputs/mujoco_runs'/run_name
    folder.mkdir(exist_ok=False)
    source=ROOT/config['experiment']['output_directory']
    for name in ('robot_experiment.xml','model_report.json','getup_poses.json'):
        (folder/name).write_bytes((source/name).read_bytes())
    (folder/'config.yaml').write_bytes(CFG.read_bytes())
    model=VelocityEstimationPPO.load(PARENT,device='cpu')
    original={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
    model.policy.requires_grad_(False)
    actor=model.policy.mlp_extractor.policy_net
    parameters=list(actor.rising_net.parameters())+list(actor.rising_head.parameters())
    for p in parameters:p.requires_grad_(True)
    optimizer=torch.optim.Adam(parameters,lr=8e-6)
    torch.manual_seed(208344)
    xs=[];ys=[];weights=[];stages=[]
    save(model,folder)
    save(model,folder,'initial_policy.zip')
    selection_specs=[(offset,seed,'train') for offset in (offsets[0],offsets[3],offsets[-1])
        for seed in range(461000,461004)]
    selection_specs += [(None,462000,'nominal')]
    def assess(cp):
        rows=[r[0] for r in parallel([(None,o,s,str(cp),0.,False,split,'folded',False)
            for o,s,split in selection_specs])]
        hard=[r for r in rows if r['offset_m']==offsets[0]]
        nominal=[r for r in rows if r['split']=='nominal']
        key=(sum(r['success'] for r in nominal),sum(r['success'] for r in hard),sum(r['success'] for r in rows))
        return key,rows
    best_key,initial=assess(folder/'initial_policy.zip')
    save(model,folder,'best_policy.zip')
    for stage,mix in enumerate((1.,.75,.25,0.,0.)):
        specs=[(o,452000+stage*1000+j,'train') for j,o in enumerate(offsets)]
        specs += [(None,453000+stage,'nominal')]
        jobs=[(recipe,o,s,str(folder/'final_policy.zip') if stage else None,mix,True,split,'folded',False)
            for o,s,split in specs]
        gathered=parallel(jobs)
        for row,x,y in gathered:
            if stage==0 and not row['success']:continue
            if not len(x):continue
            xs.extend(x);ys.extend(y)
            # Emphasize coordination and heading correction near acquisition.
            for observation in x:
                phase=observation[:180].reshape(5,36)[-1,-1]
                weights.append(3. if phase>.4 else 1.)
        if not xs:raise RuntimeError('No successful teacher trajectories')
        tx=torch.tensor(np.asarray(xs),dtype=torch.float32)
        ty=torch.tensor(np.asarray(ys),dtype=torch.float32)
        probabilities=np.asarray(weights)/np.sum(weights)
        rng=np.random.default_rng(208344+stage)
        losses=[]
        for update in range(1500 if stage==0 else 1000):
            ids=rng.choice(len(tx),256,p=probabilities)
            prediction=actor(tx[ids]).clamp(-1,1)
            loss=((prediction-ty[ids])/torch.tensor([.1]*4+[.25]*2)).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step();losses.append(float(loss.detach()))
        save(model,folder)
        key,selection=assess(folder/'final_policy.zip')
        if key>best_key:
            best_key=key;save(model,folder,'best_policy.zip')
        record=dict(stage=stage,teacher_mix=mix,demonstration_outcomes=[r[0] for r in gathered],
            samples=len(xs),loss_first=float(np.mean(losses[:50])),loss_last=float(np.mean(losses[-50:])),
            selection_key=key,best_key=best_key,selection_episodes=selection)
        stages.append(record)
        (folder/'stages.json').write_text(json.dumps(stages,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k not in ('demonstration_outcomes','selection_episodes')}),flush=True)
    best=VelocityEstimationPPO.load(folder/'best_policy.zip',device='cpu')
    changed=[]
    for name,value in best.policy.state_dict().items():
        if name.startswith(('mlp_extractor.policy_net.rising_net.','mlp_extractor.policy_net.rising_head.')):
            if not torch.equal(value,original[name]):changed.append(name)
        elif not torch.equal(value,original[name]):raise RuntimeError('Protected tensor changed: '+name)
    # Keep the last candidate separate; publish the selected candidate for assessment.
    save(best,folder,'selected_policy.zip')
    np.savez_compressed(folder/'corrective_labels.npz',observations=np.asarray(xs),actions=np.asarray(ys),weights=np.asarray(weights))
    result=dict(simulation_only=True,method='corrective_demonstrations_and_dagger',
        runtime_teacher_assistance=False,parent=str(PARENT),parent_sha256=sha(PARENT),
        selected=str(folder/'selected_policy.zip'),selected_sha256=sha(folder/'selected_policy.zip'),
        model_sha256=sha(source/'robot_experiment.xml'),initial_selection=initial,
        best_selection_key=best_key,stages=stages,changed_rising_tensors=changed,
        standing_estimator_critic_std_unchanged=True,seconds=time.monotonic()-started,
        note='Teacher mixing and modified observations are training only; selections use native student observations.')
    (folder/'coordination_training_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('TRAINING_REPORT='+str(folder/'coordination_training_report.json'),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['probe','distill'])
    parser.add_argument('--run-name')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if sha(PARENT)!=PARENT_SHA:raise RuntimeError('Unexpected parent')
    validate_artifacts(CFG)
    if args.mode=='probe':probe()
    else:
        if not args.run_name or Path(args.run_name).name!=args.run_name:raise ValueError('Require unique run name')
        distill(args.run_name)


if __name__=='__main__':main()
