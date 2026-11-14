"""Bounded 150 N impulse curriculum; identical plant and native recovery scoring."""
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
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_training import load_warm_start
from train_spring208_push import CFG,NOMINAL,TEST,evaluate_case,summarize,protected,write
from prepare_spring208_phase3 import sha

PARENT=ROOT/'outputs/push_recovery_20261109_1900/selected_policy.zip'
PARENT_SHA='4ac664b6b185969a95a12692f04c0bf7456773d52f8f78ad4cec922f90f759eb'
LEVELS=[30.,50.,80.,110.,150.]
TRAIN_SEED=710000
SELECT_SEED=720000
TEST_SEED=730000
ATTEMPT_STEPS=262144
TARGET=150.
PROFILE='original150'
SELECTION_INTERVAL=131072
MAX_SECONDS=6000
ATTEMPTS=((3e-6,.01,.10),(1e-6,.02,.20))
REHEARSAL_FOLDER=None


def configure_profile(profile):
    global PROFILE,LEVELS,TARGET,TRAIN_SEED,SELECT_SEED,TEST_SEED
    global ATTEMPT_STEPS,SELECTION_INTERVAL,MAX_SECONDS,ATTEMPTS
    PROFILE=profile
    if profile=='fine30':
        LEVELS=[18.,22.,26.,30.];TARGET=30.
        TRAIN_SEED=810000;SELECT_SEED=820000;TEST_SEED=830000
        ATTEMPT_STEPS=196608;SELECTION_INTERVAL=65536;MAX_SECONDS=3600
        ATTEMPTS=((3e-6,.005,.05),(1e-5,.01,.10))
    elif profile=='anchored30':
        LEVELS=[18.,22.,26.,30.];TARGET=30.
        TRAIN_SEED=910100;SELECT_SEED=920000;TEST_SEED=930000
        ATTEMPT_STEPS=262144;SELECTION_INTERVAL=65536;MAX_SECONDS=5400
        ATTEMPTS=((1e-5,.02,.20),(3e-5,.03,.30))
    elif profile=='stability18':
        LEVELS=[15.,18.];TARGET=18.
        TRAIN_SEED=940100;SELECT_SEED=950000;TEST_SEED=960000
        ATTEMPT_STEPS=196608;SELECTION_INTERVAL=65536;MAX_SECONDS=2400
        ATTEMPTS=((2e-6,.003,.03),(1e-6,.005,.05))


def configure_training_extras(model,folder,seed):
    config=load_config(CFG);xml=ROOT/config['experiment']['output_directory']/'robot_experiment.xml'
    if PROFILE in ('anchored30','stability18'):
        from push_recovery_rehearsal import settings
        previous=float(getattr(model,'training_reward_scale',1.))
        with torch.no_grad():
            model.policy.value_net.weight.mul_(.01/previous)
            model.policy.value_net.bias.mul_(.01/previous)
        model.training_reward_scale=.01
        return model.configure_standing_rehearsal(settings(folder,PROFILE),config,xml,seed)
    return model.configure_standing_rehearsal(dict(enabled=False),config,xml,seed)


def factory(seed,level,monitor):
    fine=PROFILE in ('fine30','anchored30','stability18')
    anchored=PROFILE in ('anchored30','stability18')
    stability=PROFILE=='stability18'
    replay_folder=REHEARSAL_FOLDER
    def create():
        config=load_config(CFG)
        env=make_mujoco_env(ROOT/config['experiment']['output_directory']/'robot_experiment.xml',CFG)
        base=env.unwrapped;base.getup['stand_probability']=1.
        base.push.config=dict(base.push.config,episode_probability=.9,force_N=[2.,level],
            force_bands=[dict(force_N=[2.,15.],probability=.2),
                         dict(force_N=[15.,level],probability=.3),
                         dict(force_N=[max(15.,.75*level),level],probability=.5)])
        if stability:
            base.push.config.update(episode_probability=.5,force_bands=[
                dict(force_N=[2.,10.],probability=.4),
                dict(force_N=[10.,15.],probability=.3),
                dict(force_N=[15.,level],probability=.3)])
        # Wider recovery penalties supply a signal outside precise hold rewards.
        # Acceptance, motor envelope and physics remain the native baseline's.
        if not fine:
            base.balance_recovery.settings=dict(base.balance_recovery.settings,
                position_weight=2.,velocity_weight=3.,heading_weight=1.,yaw_rate_weight=1.,
                tilt_weight=5.,tilt_rate_weight=1.)
        env.reset(seed=seed)
        if anchored:
            from climber_legged_rl.push_reset_replay import PushResetReplay
            from push_recovery_rehearsal import ScaledReward
            replay=json.loads((Path(replay_folder)/'hard_training_resets.json').read_text())
            if stability:
                from climber_legged_rl.push_reset_replay import ExactPushResetReplay
                cases=[c for c in replay['cases'] if c['force_N'][1]<=level]
                if cases:env=ExactPushResetReplay(env,cases,.2,seed+1000003)
            else:
                env=PushResetReplay(env,replay,seed+1000003)
            env=ScaledReward(env,.01)
        return Monitor(env,filename=monitor)
    return create


def specifications(level,split):
    final=split=='test';seed=TEST_SEED if final else SELECT_SEED
    count=24 if final else 12
    rows=[dict(group='level',config=str(NOMINAL),mode='standing',push=True,
               force_N=[max(15.,.7*level),level],seed=seed+100+i) for i in range(count)]
    if final:
        rows += [dict(group=f'exact{int(TARGET)}',config=str(NOMINAL),mode='standing',push=True,
                      force_N=[TARGET,TARGET],seed=seed+200+i) for i in range(24)]
    rows += [dict(group='accepted15',config=str(NOMINAL),mode='standing',push=True,
                  force_N=[10.,15.],seed=seed+300+i) for i in range(24 if final else 8)]
    rows += [dict(group='standing',config=str(NOMINAL),mode='standing',push=False,
                  seed=seed+i) for i in range(32 if final else 8)]
    rows += [dict(group='nominal_getup',config=str(NOMINAL),mode='folded',push=False,
                  seed=seed+400+i) for i in range(4 if final else 2)]
    offsets=load_config(TEST if final else CFG)['getup']['rearward_offsets_m']
    if not final:offsets=[offsets[0],offsets[3],offsets[-1]]
    rows += [dict(group='difficult_getup',config=str(TEST if final else CFG),mode='folded',push=False,
                  seed=seed+500+i,offset_m=o) for o in offsets for i in range(16 if final else 4)]
    return rows


class Assessor:
    def __init__(self):self.pool=ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'))
    def evaluate(self,checkpoint,level,split='selection',physics=False):
        started=time.monotonic()
        rows=list(self.pool.map(evaluate_case,[(str(checkpoint),s,physics) for s in specifications(level,split)]))
        return dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),level_N=level,split=split,
            episodes=rows,summary=summarize(rows),seconds=time.monotonic()-started,
            physics_step_spring_measurement=physics)
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)


def identity(row):
    s=row['spec'];return s['group'],s['seed'],s.get('offset_m')


def retained(result,reference):
    before={identity(r):r for r in reference['episodes']}
    lost=[identity(r) for r in result['episodes']
          if r['spec']['group'] in ('standing','nominal_getup','difficult_getup','accepted15')
          and before[identity(r)]['success'] and not r['success']]
    s=result['summary'];b=reference['summary']
    return (not lost and not any(r['warnings'] for r in result['episodes'])
            and s['accepted15']['success']>=b['accepted15']['success'])


def score(result):
    s=result['summary'];r=s['level']
    pushed=[x for x in result['episodes'] if x['spec']['group']=='level']
    # Partial recovery and survival break zero-success ties during hard stages.
    deadlines=[min(x['seconds'],20.) for x in pushed]
    return [r['success'],r['recovered_push_count'],sum(deadlines),
            s['accepted15']['success'],-sum(x['final_errors']['position_m'] for x in pushed),
            -sum(x['final_errors']['heading_deg'] for x in pushed)]


def pass_level(result,baseline,threshold=.9):
    s=result['summary'];r=s['level']
    return (retained(result,baseline) and r['success']/r['count']>=threshold
            and s['standing']['success']/s['standing']['count']>=.95
            and s['nominal_getup']['success']==s['nominal_getup']['count'])


class Selection(BaseCallback):
    def __init__(self,folder,assessor,level,attempt,parent,baseline,best,status,consumed):
        super().__init__();self.folder=folder;self.assessor=assessor;self.level=level;self.attempt=attempt
        self.parent=parent;self.baseline=baseline;self.best=best;self.status=status;self.consumed=consumed
        self.next=SELECTION_INTERVAL;self.curve=[];self.goal=False
        self.deadline=time.monotonic()+MAX_SECONDS
    def assess(self,final=False):
        suffix='_final' if final else ''
        cp=self.folder/f'level{int(self.level)}_attempt{self.attempt}_step{self.num_timesteps}{suffix}.zip'
        protected(self.model,self.parent);self.model.save(cp)
        self.status('selection',level_N=self.level,attempt=self.attempt,steps=self.consumed+self.num_timesteps)
        result=self.assessor.evaluate(cp,self.level)
        write(cp.with_suffix('.json'),result)
        accept=retained(result,self.baseline) and score(result)>score(self.best)
        if accept:
            self.best=result;shutil.copy2(cp,self.folder/'selected_policy.zip')
            write(self.folder/'selected_selection.json',result)
        self.goal=pass_level(self.best,self.baseline)
        row=dict(steps=self.consumed+self.num_timesteps,level_N=self.level,attempt=self.attempt,
            accepted=accept,level_passed=self.goal,key=score(result),summary=result['summary'],
            checkpoint=str(cp),final_optimizer_state=final)
        self.curve.append(row);write(self.folder/f'curve_{int(self.level)}_{self.attempt}.json',self.curve)
        self.status('training',**row)
    def _on_step(self):
        if (self.folder/'STOP').exists() or time.monotonic()>=self.deadline:return False
        if self.num_timesteps>=self.next:
            self.next+=SELECTION_INTERVAL;self.assess()
            if self.goal:return False
        return True


def preflight(folder):
    checks=[]
    exact_check=None
    if PROFILE=='stability18':
        from climber_legged_rl.push_reset_replay import ExactPushResetReplay
        cases=json.loads((folder/'hard_training_resets.json').read_text())['cases']
        case=next(c for c in cases if c['expected_push_events'])
        replay_env=factory(TRAIN_SEED,TARGET,None)()
        wrapper=replay_env
        while not isinstance(wrapper,ExactPushResetReplay):wrapper=wrapper.env
        wrapper.cases=[case];wrapper.probability=.9;wrapper.rng=np.random.default_rng(12)
        try:
            obs,reset_info=replay_env.reset()
            if not reset_info['training_exact_force_replay']:raise RuntimeError('Exact replay not selected')
            reference=PPO.load(PARENT,device='cpu')
            while True:
                obs,_,done,truncated,info=replay_env.step(reference.predict(obs,deterministic=True)[0])
                if done or truncated:break
            actual=info['push_metrics']['push_events'];expected=case['expected_push_events']
            if len(actual)!=len(expected):raise RuntimeError('Exact replay event count changed')
            for a,b in zip(actual,expected):
                for key in ('start_step','end_step','duration_seconds','force_world_N'):
                    if not np.array_equal(a[key],b[key]):raise RuntimeError('Exact replay differs: '+key)
            exact_check=dict(seed=case['seed'],force_N=case['force_N'],events_matched=len(actual),
                seconds=replay_env.unwrapped.steps*replay_env.unwrapped.dt)
        finally:replay_env.close()
    for level in (LEVELS[0],TARGET):
        env=factory(TRAIN_SEED,level,None)()
        try:
            env.unwrapped.push.config=dict(env.unwrapped.push.config,episode_probability=1.,
                force_N=[level,level],force_bands=[dict(force_N=[level,level],probability=1.)])
            if PROFILE=='stability18':
                wrapper=env
                while hasattr(wrapper,'env'):
                    if isinstance(wrapper,ExactPushResetReplay):
                        wrapper.original=copy.deepcopy(env.unwrapped.push.config)
                        break
                    wrapper=wrapper.env
            # A previously verified standing reset isolates force application
            # from unrelated rare failures before the scheduled first pulse.
            obs,_=env.reset(seed=630000,options={'reset_mode':'standing'})
            policy=PPO.load(PARENT,device='cpu');seen=[]
            for _ in range(320):
                obs,_,d,t,info=env.step(policy.predict(obs,deterministic=True)[0])
                force=np.linalg.norm(info['push_force_world_N'])
                if force:seen.append(force)
                if d or t:break
            if not seen or not np.allclose(seen,level):
                raise RuntimeError(f'Physical force check failed: level={level}, measured={seen}, steps={env.unwrapped.steps}')
            checks.append(dict(level_N=level,measured_force_N=float(max(seen)),pulse_steps=len(seen),
                warnings=int(sum(env.unwrapped.data.warning.number))))
        finally:env.close()
    v=DummyVecEnv([factory(TRAIN_SEED+1,LEVELS[0],None)])
    try:
        settings=dict(load_config(CFG)['training'],freeze_rising_actor=True,freeze_standing_actor=False,
            reset_optimizer_on_warm_start=True,learning_rate=3e-6,n_steps=256,batch_size=128,n_epochs=1)
        if PROFILE=='stability18':settings['learning_rate']=ATTEMPTS[0][0]
        model,_,_=load_warm_start(PARENT,v,settings,'cpu',TRAIN_SEED)
        model.configure_estimator(dict(settings['velocity_estimator'],trainable=False))
        actor=model.policy.mlp_extractor.policy_net
        assert all(p.requires_grad for p in actor.standing_net.parameters())
        assert not any(p.requires_grad for p in actor.rising_net.parameters())
        assert not any(p.requires_grad for p in actor.velocity_estimator.parameters())
        obs=v.reset();samples=0
        for _ in range(4):
            action=model.predict(obs,deterministic=True)[0];obs,_,_,_=v.step(action);samples+=1
        model.configure_gradient_clipping(True)
        replay_contract=configure_training_extras(model,folder,TRAIN_SEED)
        leg_std,wheel_std=(ATTEMPTS[0][1:] if PROFILE=='stability18' else (.01,.1))
        with torch.no_grad():model.policy.log_std.copy_(torch.log(torch.tensor([leg_std]*4+[wheel_std]*2)))
        model.set_logger(configure(str(folder/'preflight_optimizer'),['csv']))
        reference=PPO.load(PARENT,device='cpu')
        model.learn(total_timesteps=1024,reset_num_timesteps=True,progress_bar=False)
        contract=protected(model,reference)
        if not contract['changed_standing_tensors']:raise RuntimeError('Standing optimizer did not update')
        if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
            raise RuntimeError('Nonfinite optimizer output')
        result=dict(passed=True,force_checks=checks,rising_frozen=True,estimator_frozen=True,
            standing_trainable=True,parent_sha256=sha(PARENT),native_step_checks=samples,
            discarded_optimizer_smoke_steps=1024,optimizer_contract=contract,
            model_sha256=sha(ROOT/load_config(CFG)['experiment']['output_directory']/'robot_experiment.xml'),
            profile=PROFILE,script_sha256=sha(__file__),rehearsal=replay_contract,
            exact_training_replay=exact_check,
            training_reward_scale=float(getattr(model,'training_reward_scale',1.)))
        write(folder/'preflight.json',result);print(json.dumps(result),flush=True)
    finally:v.close()


def main():
    global REHEARSAL_FOLDER
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--profile',choices=['original150','fine30','anchored30','stability18'],default='original150')
    args=parser.parse_args();configure_profile(args.profile)
    folder=args.folder.resolve();folder.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    REHEARSAL_FOLDER=str(folder)
    if sha(PARENT)!=PARENT_SHA:raise RuntimeError('Unexpected accepted checkpoint')
    for path in (CFG,NOMINAL,TEST):validate_artifacts(path)
    if args.check:preflight(folder);return
    lock=(ROOT/'outputs/spring208_phase1_launch/workflow.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'workflow.json').exists():raise RuntimeError('Refusing to overwrite a run')
    checked=json.loads((folder/'preflight.json').read_text())
    if not (checked['passed'] and checked['profile']==PROFILE and checked['script_sha256']==sha(__file__)):
        raise RuntimeError('Preflight missing or stale')
    start=time.monotonic();state=dict(pid=os.getpid(),started_unix=time.time(),simulation_only=True,
        profile=PROFILE,target_force_N=TARGET,long_term_target_N=150.,
        target_success_fraction=.9,max_steps=ATTEMPT_STEPS*len(ATTEMPTS)*len(LEVELS),
        max_training_seconds=MAX_SECONDS,auto_hardware_deployment=False)
    def status(phase,**extra):
        state.update(state=phase,elapsed_seconds=time.monotonic()-start,**extra)
        write(folder/'workflow.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)
    total=0;assessor=Assessor();parent=PPO.load(PARENT,device='cpu');levels=[]
    try:
        write(folder/'protocol.json',dict(parent=str(PARENT),parent_sha256=PARENT_SHA,
            script_sha256=sha(__file__),evaluator_sha256=sha(ROOT/'scripts/train_spring208_push.py'),
            profile=PROFILE,training_levels_N=LEVELS,attempts_per_level=len(ATTEMPTS),steps_per_attempt=ATTEMPT_STEPS,
            selection_interval=SELECTION_INTERVAL,attempt_settings=ATTEMPTS,
            recovery_rewards='native accepted 15 N' if PROFILE in ('fine30','anchored30','stability18') else 'broader recovery penalties',
            training_reward_scale=.01 if PROFILE in ('anchored30','stability18') else 1.,
            successful_standing_rehearsal=PROFILE in ('anchored30','stability18'),
            hard_training_resets=(json.loads((folder/'hard_training_resets.json').read_text()) if PROFILE in ('anchored30','stability18') else None),
            rehearsal_helper_sha256=sha(ROOT/'scripts/push_recovery_rehearsal.py') if PROFILE in ('anchored30','stability18') else None,
            training_seed=TRAIN_SEED,selection_seed=SELECT_SEED,test_seed=TEST_SEED,
            pulse_seconds=[.05,.10],force_is_total_3D_magnitude=True,elevation_deg=[-15.,15.],
            direction='8 balanced world-frame azimuth sectors, +/-10 degrees jitter',
            target=f'{TARGET:g} N fixed magnitude and {max(15.,.7*TARGET):g}-{TARGET:g} N random pulses >=90%; standing >=95%; accepted15 and baseline getup successes retained',
            native_recovery_seconds=4.,settle_seconds=.5,final_position_m=.05,final_heading_deg=5.,
            max_transient_position_m=.2,max_transient_heading_deg=20.,
            model_and_motor_limits_unchanged=True,rising_and_estimator_frozen=True,
            final_test_not_used_for_selection=True,final_physics_step_sampling=True,
            estimator_and_inertia_not_hardware_identified=True,
            user_accepted_previous_15N_policy=True,automatic_velocity_training=False))
        shutil.copy2(PARENT,folder/'selected_policy.zip')
        for level in LEVELS:
            if time.monotonic()-start>MAX_SECONDS or (folder/'STOP').exists():break
            status('level_baseline',level_N=level,steps=total)
            baseline=assessor.evaluate(PARENT,level)
            write(folder/f'parent_level{int(level)}.json',baseline)
            best=baseline if sha(folder/'selected_policy.zip')==PARENT_SHA else assessor.evaluate(folder/'selected_policy.zip',level)
            write(folder/'selected_selection.json',best)
            level_ok=pass_level(best,baseline)
            for attempt,(rate,leg_std,wheel_std) in enumerate(ATTEMPTS):
                if level_ok or time.monotonic()-start>MAX_SECONDS or (folder/'STOP').exists():break
                vector=SubprocVecEnv([factory(TRAIN_SEED+int(level)*100+attempt*10+i,level,
                    str(folder/f'monitor_{int(level)}_{attempt}_{i}.csv')) for i in range(4)],start_method='spawn')
                try:
                    settings=copy.deepcopy(load_config(CFG)['training'])
                    settings.update(n_steps=1024,batch_size=512,n_epochs=3,learning_rate=rate,
                        freeze_rising_actor=True,freeze_standing_actor=False,freeze_action_std=True,
                        reset_optimizer_on_warm_start=True,target_kl=.006,clip_range=.04)
                    if PROFILE=='stability18':
                        settings.update(n_epochs=2,target_kl=.003,clip_range=.02)
                    model,_,_=load_warm_start(folder/'selected_policy.zip',vector,settings,'cpu',TRAIN_SEED+int(level)+attempt)
                    model.configure_estimator(dict(settings['velocity_estimator'],trainable=False))
                    model.configure_gradient_clipping(True)
                    configure_training_extras(model,folder,TRAIN_SEED+attempt)
                    with torch.no_grad():model.policy.log_std.copy_(torch.log(torch.tensor([leg_std]*4+[wheel_std]*2)))
                    model.set_logger(configure(str(folder/f'logs_{int(level)}_{attempt}'),['csv']))
                    protected(model,parent)
                    callback=Selection(folder,assessor,level,attempt,parent,baseline,best,status,total)
                    callback.deadline=start+MAX_SECONDS
                    write(folder/f'settings_{int(level)}_{attempt}.json',dict(settings,
                        action_std=[leg_std]*4+[wheel_std]*2,force_N=level))
                    status('training',level_N=level,attempt=attempt,steps=total)
                    model.learn(total_timesteps=ATTEMPT_STEPS,callback=callback,reset_num_timesteps=True,progress_bar=False)
                    callback.assess(final=True);total+=model.num_timesteps;best=callback.best
                    level_ok=pass_level(best,baseline)
                finally:vector.close()
            levels.append(dict(force_N=level,passed=level_ok,summary=best['summary']))
            write(folder/'levels.json',levels)
            if not level_ok:break
            shutil.copy2(folder/'selected_policy.zip',folder/f'passed_level{int(level)}.zip')
        status('final_independent_test',steps=total,evaluation_force_N=TARGET)
        before=assessor.evaluate(PARENT,TARGET,'test',True);write(folder/'parent_test.json',before)
        after=assessor.evaluate(folder/'selected_policy.zip',TARGET,'test',True);write(folder/'selected_test.json',after)
        exact=after['summary'][f'exact{int(TARGET)}']
        physics_ok=(not any(r['warnings'] for r in after['episodes']) and
            not any(max(r['spring_upper_mm']+r['spring_lower_mm'])>1. for r in after['episodes']))
        passed=(pass_level(after,before) and exact['success']/exact['count']>=.9 and physics_ok)
        report=dict(target_force_N=TARGET,target_passed=passed,
            target_150N_passed=bool(passed and TARGET==150.),steps=total,seconds=time.monotonic()-start,
            parent_summary=before['summary'],selected_summary=after['summary'],levels=levels,
            protected_actor=protected(PPO.load(folder/'selected_policy.zip',device='cpu'),parent),
            physical_checks_passed=physics_ok,simulation_only=True,
            selected_checkpoint=str(folder/'selected_policy.zip'),selected_sha256=sha(folder/'selected_policy.zip'),
            parent_sha256=sha(PARENT),automatic_velocity_training=False,
            note=f'A {TARGET:g} N 0.05-0.10 s pulse is not continuous pushing. Unfinished curriculum is reported, never silently promoted.')
        write(folder/'assessment.json',report)
        if sha(PARENT)!=PARENT_SHA:raise RuntimeError('Parent changed')
        status('completed',steps=total,target_passed=passed,
            target_150N_passed=bool(passed and TARGET==150.),report=str(folder/'assessment.json'))
    except BaseException as error:
        status('failed',error=repr(error));raise
    finally:assessor.close()


if __name__=='__main__':main()
