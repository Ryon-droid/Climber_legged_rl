"""Bounded standing/push PPO, retaining the accepted rising actor and mechanics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'scripts')]
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_training import load_warm_start
from climber_legged_rl.spring_geometry import eye_center_distance
from prepare_spring208_phase3 import PARENT,PARENT_SHA,sha

CFG=ROOT/'configs/robot/cad_spring208_phase3_train.yaml'
NOMINAL=ROOT/'configs/robot/cad_spring208_phase3_nominal.yaml'
TEST=ROOT/'configs/robot/cad_spring208_phase3_test.yaml'
TRAIN_SEED=610000
SELECT_SEED=620000
TEST_SEED=630000


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def factory(seed,log_path):
    def create():
        config=load_config(CFG)
        env=make_mujoco_env(ROOT/config['experiment']['output_directory']/'robot_experiment.xml',CFG)
        base=env.unwrapped
        base.getup['stand_probability']=1.
        base.push.config=dict(base.push.config,episode_probability=.85,
            force_bands=[dict(force_N=[2.,5.],probability=.2),
                         dict(force_N=[5.,10.],probability=.3),
                         dict(force_N=[10.,15.],probability=.5)])
        # Train the same task, concentrating samples on its upper force band.
        # Broad recovery terms already exist in the accepted configuration.
        env.reset(seed=seed)
        return Monitor(env,filename=log_path)
    return create


def evaluate_case(job):
    checkpoint,spec,physics_measurement=job
    torch.set_num_threads(1)
    config_path=Path(spec['config'])
    config=load_config(config_path)
    env=make_mujoco_env(ROOT/config['experiment']['output_directory']/'robot_experiment.xml',config_path)
    base=env.unwrapped
    from climber_legged_rl.motor_authority import apply_standing_motor_limit
    motor_envelope=apply_standing_motor_limit(env,spec.get('standing_leg_torque_limit_Nm'),
                                             spec.get('standing_wheel_kv'))
    push=dict(base.push.config,episode_probability=float(spec['push']))
    if spec.get('force_N'):
        push['force_N']=spec['force_N'];push.pop('force_bands',None)
    base.push.config=push
    if spec.get('offset_m') is not None:
        env.pose_bank=[p for p in env.pose_bank if abs(p['rearward_offset_m']-spec['offset_m'])<1e-10]
        if not env.pose_bank:raise RuntimeError('Missing evaluation geometry')
    if spec.get('balance_curriculum_stage'):
        from climber_legged_rl.push_balance_curriculum import PushBalanceCurriculum
        env=PushBalanceCurriculum(env,spec['balance_curriculum_stage'])
    policy=PPO.load(checkpoint,device='cpu')
    low=np.full(2,np.inf);high=np.full(2,-np.inf);upper=np.zeros(2);lower=np.zeros(2)
    samples=0
    def springs():
        nonlocal samples
        lengths=np.array([eye_center_distance(base.model,base.data,s) for s in ('left','right')])
        bounds=np.array([config['spring']['eye_center_limits_m'][s] for s in ('left','right')])
        np.minimum(low,lengths,out=low);np.maximum(high,lengths,out=high)
        np.maximum(upper,lengths-bounds[:,1],out=upper)
        np.maximum(lower,bounds[:,0]-lengths,out=lower);samples+=1
    if physics_measurement:
        observe=base._observe_motor_torque
        def measured(torque):
            observe(torque);springs()
        base._observe_motor_torque=measured
    try:
        obs,reset=env.reset(seed=spec['seed'],options={'reset_mode':spec['mode']})
        measure_quality = spec.get('measure_recovery_quality', False)
        previous_action = previous_delta = None
        action_curvature_sum = angular_speed_squared = 0.
        min_clearance = float('inf')
        clearance_samples = quality_samples = 0
        if measure_quality:
            from climber_legged_rl.recovery_clearance_reward import minimum_base_clearance
        if physics_measurement:springs()
        maximum=dict(displacement_m=0.,heading_deg=0.,tilt_deg=0.,closure_m=0.)
        pre_push_maximum=dict(displacement_m=0.,heading_deg=0.,tilt_deg=0.)
        warnings=np.array(base.data.warning.number).copy()
        while True:
            action=policy.predict(obs,deterministic=True)[0]
            obs,reward,done,truncated,info=env.step(action)
            if measure_quality and info.get('getup_acquired', True):
                clipped = np.clip(action, -1., 1.)
                delta = None if previous_action is None else clipped-previous_action
                if delta is not None and previous_delta is not None:
                    action_curvature_sum += float(np.mean((delta-previous_delta)**2))
                previous_action = clipped.copy()
                previous_delta = None if delta is None else delta.copy()
                angular_speed_squared += float(np.sum(base._kinematics()[0][:2]**2))
                quality_samples += 1
                if base.steps % 10 == 0 or done or truncated:
                    min_clearance = min(min_clearance, minimum_base_clearance(base))
                    clearance_samples += 1
            maximum['displacement_m']=max(maximum['displacement_m'],info['planar_displacement_m'])
            maximum['heading_deg']=max(maximum['heading_deg'],float(np.degrees(abs(info['heading_error_rad']))))
            maximum['tilt_deg']=max(maximum['tilt_deg'],float(np.degrees(info['tilt_rad'])))
            maximum['closure_m']=max(maximum['closure_m'],info['closure_error_m'])
            if spec['mode']=='standing' and base.steps*base.dt<=base.push.config['first_push_seconds']:
                pre_push_maximum['displacement_m']=max(pre_push_maximum['displacement_m'],info['planar_displacement_m'])
                pre_push_maximum['heading_deg']=max(pre_push_maximum['heading_deg'],float(np.degrees(abs(info['heading_error_rad']))))
                pre_push_maximum['tilt_deg']=max(pre_push_maximum['tilt_deg'],float(np.degrees(info['tilt_rad'])))
            if done or truncated:break
        if physics_measurement:
            springs()
            if samples!=base.steps*config['control']['decimation']+2:
                raise RuntimeError('Incomplete physics-step spring measurement')
        events=copy.deepcopy(info['push_metrics'])
        if (spec['push'] and events['push_count']==0
                and base.steps*base.dt>base.push.config['first_push_seconds']+base.dt):
            raise RuntimeError('Push case ended without an actual applied pulse')
        return dict(spec=spec,motor_envelope=motor_envelope,success=bool(info['combined_success']),
            **(dict(strict_success=info['strict_success'],strict_push_metrics=info['strict_push_metrics'])
               if 'strict_success' in info else {}),
            acquired=bool(info['getup_acquired']),seconds=base.steps*base.dt,
            failed_before_first_push=bool(spec['push'] and events['push_count']==0),
            reason=info['termination_reason'] or 'timeout',getup_seconds=info['getup_seconds'],
            maximum=maximum,pre_push_maximum=pre_push_maximum,push_metrics=events,
            warnings=int(sum(np.array(base.data.warning.number)-warnings)),
            spring_upper_mm=(upper*1000).tolist() if physics_measurement else None,
            spring_lower_mm=(lower*1000).tolist() if physics_measurement else None,
            spring_sample_count=samples,peak_torque_Nm=info['peak_torque_Nm'],
            recovery_quality=(dict(action_curvature_rms=float(np.sqrt(action_curvature_sum/max(quality_samples-2,1))),
                tilt_rate_rms_rad_s=float(np.sqrt(angular_speed_squared/max(quality_samples,1))),
                minimum_base_clearance_mm=min_clearance*1000 if clearance_samples else None,
                clearance_sample_count=clearance_samples) if measure_quality else None),
            final_errors=dict(position_m=info['planar_displacement_m'],
                heading_deg=float(np.degrees(abs(info['heading_error_rad']))),
                speed_m_s=info['planar_speed_m_s'],yaw_rate_rad_s=abs(info['yaw_rate_rad_s'])))
    finally:env.close()


def specs(split):
    seed=SELECT_SEED if split=='selection' else TEST_SEED
    standing=8 if split=='selection' else 32
    band_count=8 if split=='selection' else 24
    rows=[dict(group='standing',config=str(NOMINAL),mode='standing',push=False,seed=seed+i)
          for i in range(standing)]
    for j,band in enumerate(([2.,5.],[5.,10.],[10.,15.])):
        rows += [dict(group='push_'+str(j),config=str(NOMINAL),mode='standing',push=True,
                      seed=seed+100+j*100+i,force_N=band) for i in range(band_count)]
    rows += [dict(group='nominal_getup',config=str(NOMINAL),mode='folded',push=False,seed=seed+500+i)
             for i in range(2 if split=='selection' else 4)]
    poses=load_config(CFG if split=='selection' else TEST)['getup']['rearward_offsets_m']
    if split=='selection':poses=[poses[0],poses[len(poses)//2],poses[-1]]
    rows += [dict(group='difficult_getup',config=str(CFG if split=='selection' else TEST),
                  mode='folded',push=False,seed=seed+600+i,offset_m=offset)
             for offset in poses for i in range(4 if split=='selection' else 16)]
    # A previous diagnostic failure is a regression case, not a fresh test case.
    rows += [dict(group='regression_standing',config=str(NOMINAL),mode='standing',push=push,
                  seed=540203) for push in (False,True)]
    return rows


def summarize(rows):
    groups={}
    for group in sorted({r['spec']['group'] for r in rows}):
        items=[r for r in rows if r['spec']['group']==group]
        groups[group]=dict(count=len(items),success=sum(r['success'] for r in items),
            warnings=sum(r['warnings'] for r in items),
            push_count=sum(r['push_metrics']['push_count'] for r in items),
            recovered_push_count=sum(r['push_metrics']['recovered_push_count'] for r in items),
            max_heading_deg=max(r['maximum']['heading_deg'] for r in items),
            max_displacement_m=max(r['maximum']['displacement_m'] for r in items),
            max_recovery_seconds=max((r['push_metrics']['max_recovery_seconds'] or 0. for r in items)))
    return groups


class Assessor:
    def __init__(self):
        self.pool=ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'))
    def evaluate(self,checkpoint,split,physics=False):
        started=time.monotonic()
        rows=list(self.pool.map(evaluate_case,[(str(checkpoint),s,physics) for s in specs(split)]))
        return dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),split=split,
            episodes=rows,summary=summarize(rows),seconds=time.monotonic()-started,
            physics_step_spring_measurement=physics)
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)


def identity(row):
    s=row['spec'];return s['group'],s['seed'],s.get('offset_m')


def eligible(result,parent):
    s=result['summary'];baseline={identity(r):r for r in parent['episodes']}
    regressions=[identity(r) for r in result['episodes']
                 if r['spec']['group'] in ('standing','nominal_getup','difficult_getup')
                 and baseline[identity(r)]['success'] and not r['success']]
    return not regressions and not any(r['warnings'] for r in result['episodes'])


def score(result):
    s=result['summary'];push=[s['push_'+str(i)] for i in range(3)]
    return [min(r['success']/r['count'] for r in push),sum(r['success'] for r in push),
            s['standing']['success'],s['regression_standing']['success'],
            -sum(r['max_heading_deg'] for r in push)]


def passed(result,parent):
    s=result['summary']
    return (eligible(result,parent)
        and all(s['push_'+str(i)]['success']/s['push_'+str(i)]['count']>=.95 for i in range(3))
        and s['standing']['success']/s['standing']['count']>=.95
        and s['nominal_getup']['success']==s['nominal_getup']['count']
        and not any(max((r.get('spring_upper_mm') or [0.])+ (r.get('spring_lower_mm') or [0.]))>1.
                    for r in result['episodes']))


def protected(model,parent):
    changes=[];standing=[]
    for name,value in parent.policy.state_dict().items():
        other=model.policy.state_dict()[name]
        if name.startswith(('mlp_extractor.policy_net.rising_','mlp_extractor.policy_net.velocity_estimator.')):
            if not torch.equal(value,other):changes.append(name)
        elif name.startswith(('mlp_extractor.policy_net.standing_',)) and not torch.equal(value,other):
            standing.append(name)
    if changes:raise RuntimeError('Protected policy changed: '+str(changes))
    return dict(rising_and_estimator_identical=True,changed_standing_tensors=standing)


class SelectionCallback(BaseCallback):
    def __init__(self,folder,assessor,baseline,parent,stage,global_steps,status):
        super().__init__();self.folder=folder;self.assessor=assessor;self.baseline=baseline
        self.parent=parent;self.stage=stage;self.global_steps=global_steps;self.status=status
        self.next_step=65536;self.last_step=0;self.goal=False
        self.best=json.loads((folder/'selected_selection.json').read_text())
        self.curve=[]
    def assess(self):
        self.last_step=self.num_timesteps
        checkpoint=self.folder/f'stage_{self.stage}_step_{self.num_timesteps}.zip'
        contract=protected(self.model,self.parent);self.model.save(checkpoint)
        self.status('selection',stage=self.stage,steps=self.global_steps+self.num_timesteps)
        result=self.assessor.evaluate(checkpoint,'selection')
        write(checkpoint.with_suffix('.json'),result)
        accepted=eligible(result,self.baseline) and score(result)>score(self.best)
        if accepted:
            self.best=result;shutil.copy2(checkpoint,self.folder/'selected_policy.zip')
            write(self.folder/'selected_selection.json',result)
        self.goal=passed(result,self.baseline)
        record=dict(stage=self.stage,steps=self.global_steps+self.num_timesteps,
            key=score(result),accepted=accepted,selection_gate=self.goal,summary=result['summary'],contract=contract)
        self.curve.append(record);write(self.folder/f'curve_{self.stage}.json',self.curve)
        self.status('training',**record)
    def _on_step(self):
        if (self.folder/'STOP').exists():return False
        if self.num_timesteps>=self.next_step:
            self.next_step+=65536;self.assess()
            if self.goal:return False
        return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--mode',choices=['run','check'],default='run')
    p.add_argument('--resume-failed',action='store_true');p.add_argument('--start-stage',type=int,default=0)
    args=p.parse_args()
    folder=args.folder.resolve();folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    if sha(PARENT)!=PARENT_SHA:raise RuntimeError('Unexpected parent')
    for cfg in (CFG,NOMINAL,TEST):validate_artifacts(cfg)
    if args.mode=='check':
        a=factory(TRAIN_SEED,None)();v=SubprocVecEnv([factory(TRAIN_SEED+1,None)],start_method='spawn')
        try:
            settings=dict(load_config(CFG)['training'],freeze_rising_actor=True,freeze_standing_actor=False,
                          learning_rate=3e-6,reset_optimizer_on_warm_start=True)
            m,_,_=load_warm_start(PARENT,v,settings,'cpu',TRAIN_SEED)
            m.configure_estimator(dict(settings['velocity_estimator'],trainable=False))
            actor=m.policy.mlp_extractor.policy_net
            assert all(p.requires_grad for p in actor.standing_net.parameters())
            assert not any(p.requires_grad for p in actor.rising_net.parameters())
            assert not any(p.requires_grad for p in actor.velocity_estimator.parameters())
            obs,_=a.reset(seed=TRAIN_SEED,options={'reset_mode':'standing'})
            a.unwrapped.push.config=dict(a.unwrapped.push.config,episode_probability=1.)
            obs,_=a.reset(seed=TRAIN_SEED,options={'reset_mode':'standing'})
            seen=0
            for _ in range(340):
                obs,_,done,truncated,info=a.step(m.predict(obs,deterministic=True)[0])
                seen+=int(np.linalg.norm(info['push_force_world_N'])>0.)
                if done or truncated:break
            assert seen>0,'No physical push observed'
            result=dict(passed=True,standing_trainable=True,rising_frozen=True,estimator_frozen=True,
                        applied_force_steps=seen,physical_xml_sha256=sha(ROOT/load_config(CFG)['experiment']['output_directory']/'robot_experiment.xml'))
            write(folder/'preflight.json',result);print(json.dumps(result),flush=True)
        finally:a.close();v.close()
        return
    lock=(ROOT/'outputs/spring208_phase1_launch/workflow.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=None
    if (folder/'workflow.json').exists():
        previous=json.loads((folder/'workflow.json').read_text())
        if not args.resume_failed or previous['state']!='failed':
            raise RuntimeError('Refusing to reuse workflow')
        shutil.copy2(folder/'workflow.json',folder/'interrupted_workflow.json')
    if not json.loads((folder/'preflight.json').read_text())['passed']:raise RuntimeError('Preflight missing')
    start=time.monotonic()-(previous['elapsed_seconds'] if previous else 0.);state=dict(pid=os.getpid(),started_unix=previous['started_unix'] if previous else time.time(),max_steps=786432,
        simulation_only=True,auto_hardware_deployment=False)
    def status(phase,**extra):
        state.update(state=phase,elapsed_seconds=time.monotonic()-start,**extra)
        write(folder/'workflow.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)
    assessor=Assessor();parent=PPO.load(PARENT,device='cpu');total=previous['steps'] if previous else 0
    try:
        if not previous:write(folder/'protocol.json',dict(parent=str(PARENT),parent_sha256=sha(PARENT),
            scripts_sha256=sha(__file__),train_seed=TRAIN_SEED,selection_seed=SELECT_SEED,test_seed=TEST_SEED,
            selection_specs=specs('selection'),test_specs=specs('test'),
            training_reset='standing_only',training_push_probability=.85,
            training_force_band_probabilities=[.2,.3,.5],task_criteria_unchanged=True,
            target='Each force band >=95%, unpushed standing >=95%, nominal getup 100%, no lost baseline getup cases, zero numerical warnings, spring stop penetration <=1 mm',
            max_steps=786432,final_test_not_used_for_checkpoint_selection=True,
            preliminary_single_seed=True,locomotion_starts_only_after_push_gate=True))
        if previous:
            baseline=json.loads((folder/'parent_selection.json').read_text())
            parent_test=json.loads((folder/'parent_test.json').read_text())
            write(folder/'resume.json',dict(reason=previous['error'],start_stage=args.start_stage,
                consumed_steps=total,script_sha256=sha(__file__),
                correction='Early failure before the first scheduled push is a failed episode, not an evaluator exception.',
                restart_from='selected_policy.zip',final_tests_not_used_for_selection=True))
        else:
            status('baseline_selection');baseline=assessor.evaluate(PARENT,'selection')
            write(folder/'parent_selection.json',baseline);write(folder/'selected_selection.json',baseline)
            shutil.copy2(PARENT,folder/'selected_policy.zip')
            status('baseline_test');parent_test=assessor.evaluate(PARENT,'test',physics=True)
            write(folder/'parent_test.json',parent_test)
        if passed(parent_test,parent_test):
            status('completed',gate_passed=True,training_steps=0,note='Parent already meets the tested push gate')
            return
        stages=[(3e-6,.005,.05),(1e-6,.003,.03),(1e-5,.005,.05),(3e-6,.01,.1)]
        for stage,(lr,leg_std,wheel_std) in enumerate(stages):
            if stage<args.start_stage:continue
            if (folder/'STOP').exists():break
            settings=copy.deepcopy(load_config(CFG)['training'])
            settings.update(n_steps=512,batch_size=256,n_epochs=3,learning_rate=lr,
                freeze_rising_actor=True,freeze_standing_actor=False,freeze_action_std=True,
                reset_optimizer_on_warm_start=True,target_kl=.006,clip_range=.04)
            vector=SubprocVecEnv([factory(TRAIN_SEED+stage*1000+i,str(folder/f'monitor_{stage}_{i}.csv'))
                                 for i in range(4)],start_method='spawn')
            try:
                model,_,_=load_warm_start(folder/'selected_policy.zip',vector,settings,'cpu',TRAIN_SEED+stage)
                model.configure_estimator(dict(settings['velocity_estimator'],trainable=False))
                model.configure_gradient_clipping(True)
                model.configure_standing_rehearsal(dict(enabled=False),load_config(CFG),
                    ROOT/load_config(CFG)['experiment']['output_directory']/'robot_experiment.xml',TRAIN_SEED)
                with torch.no_grad():model.policy.log_std.copy_(torch.log(torch.tensor([leg_std]*4+[wheel_std]*2)))
                model.set_logger(configure(str(folder/f'logs_{stage}'),['csv']))
                protected(model,parent)
                callback=SelectionCallback(folder,assessor,baseline,parent,stage,total,status)
                write(folder/f'settings_{stage}.json',dict(settings,action_std=[leg_std]*4+[wheel_std]*2))
                status('training',stage=stage,steps=total,learning_rate=lr)
                model.learn(total_timesteps=196608,callback=callback,reset_num_timesteps=True,progress_bar=False)
                # The rollout callback precedes PPO's final optimizer update.
                # Always assess those final updated weights too.
                callback.assess()
                model.save(folder/f'final_stage_{stage}.zip');total+=model.num_timesteps
                if callback.goal:break
            finally:vector.close()
        status('final_independent_test',steps=total)
        final=assessor.evaluate(folder/'selected_policy.zip','test',physics=True)
        write(folder/'selected_test.json',final)
        contract=protected(PPO.load(folder/'selected_policy.zip',device='cpu'),parent)
        gate=passed(final,parent_test)
        report=dict(parent_summary=parent_test['summary'],selected_summary=final['summary'],
            gate_passed=gate,new_training_steps=total,seconds=time.monotonic()-start,
            selected_checkpoint=str(folder/'selected_policy.zip'),selected_sha256=sha(folder/'selected_policy.zip'),
            parent_sha256=sha(PARENT),contract=contract,simulation_only=True,
            next_step='low_speed_velocity_curriculum' if gate else 'review_push_failures_before_velocity',
            target_note='Difficult getup >=95% remains a separate unsolved goal; this gate only requires no regression there.')
        write(folder/'assessment.json',report)
        if sha(PARENT)!=PARENT_SHA:raise RuntimeError('Parent file changed')
        status('completed',steps=total,gate_passed=gate,report=str(folder/'assessment.json'))
    except BaseException as error:
        status('failed',error=repr(error));raise
    finally:assessor.close()


if __name__=='__main__':main()
