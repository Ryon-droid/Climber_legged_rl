"""Paired PPO pilots comparing successful-action rehearsal constraints."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'scripts')]
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_training import load_warm_start
from climber_legged_rl.push_reset_replay import ExactPushResetReplay
from climber_legged_rl.standing_rehearsal import load_standing_rehearsal
from push_recovery_rehearsal import ScaledReward
import train_spring208_push150 as common
from train_spring208_push import CFG,protected,write
from prepare_spring208_phase3 import sha

SOURCE=ROOT/'outputs/push_stability18_20261110_2115'
PARENT=SOURCE/'selected_policy.zip'
PARENT_SHA='c602614a55b711518a60fdfd826365b56e87d7076e740d3ae2fe4d93d259532f'
SEED=970000
STEPS=131072
INTERVAL=65536
SECONDS=1500
ARMS=('all_success','standing_only')
PROFILE='recovery_ab18'
LEARNING_RATE=3e-6
ACTION_STD=[.006]*4+[.06]*2
EPOCHS=2
CLIP_RANGE=.02
TARGET_KL=.003
NUM_ENVS=4
ROLLOUT_TRANSITIONS=4096
HARD_PROBABILITY=.4
FOCUSED_PROFILES=('focused18','anchor_ab18','recovery_replay18','wheel_recovery18')


def configure_profile(profile):
    global PROFILE,ARMS,SEED,STEPS,SECONDS,LEARNING_RATE,ACTION_STD,EPOCHS,CLIP_RANGE,TARGET_KL
    global SOURCE,PARENT,PARENT_SHA,INTERVAL,ROLLOUT_TRANSITIONS,HARD_PROBABILITY
    if profile not in ('recovery_ab18','exploration_ab18',*FOCUSED_PROFILES):
        raise ValueError('Unknown paired training profile')
    PROFILE=profile
    ARMS=('all_success','standing_only');SEED=970000;STEPS=131072;SECONDS=1500
    LEARNING_RATE=3e-6;ACTION_STD=[.006]*4+[.06]*2;EPOCHS=2;CLIP_RANGE=.02;TARGET_KL=.003
    SOURCE=ROOT/'outputs/push_stability18_20261110_2115';PARENT=SOURCE/'selected_policy.zip'
    PARENT_SHA='c602614a55b711518a60fdfd826365b56e87d7076e740d3ae2fe4d93d259532f'
    INTERVAL=65536;ROLLOUT_TRANSITIONS=4096;HARD_PROBABILITY=.4
    if profile=='exploration_ab18':
        ARMS=('no_rehearsal','standing_only');SEED=1001000;STEPS=262144;SECONDS=2100
        LEARNING_RATE=1e-5;ACTION_STD=[.02]*4+[.20]*2;EPOCHS=4;CLIP_RANGE=.05;TARGET_KL=.006
    elif profile in FOCUSED_PROFILES:
        SOURCE=ROOT/'outputs/push_focus18_preparation_20261111'
        PARENT=ROOT/'outputs/push_exploration_ab18_20261111_1405/selected_policy.zip'
        PARENT_SHA='a444d90aee7013cc8dc6f05f183040f0a9137c1ca11d1457b27fe56b9cf10fe3'
        ARMS=('standing_only',);SEED=1080000;STEPS=491520;SECONDS=2100
        LEARNING_RATE=1e-5;ACTION_STD=[.02]*4+[.20]*2;EPOCHS=4;CLIP_RANGE=.05;TARGET_KL=.006
        INTERVAL=98304;ROLLOUT_TRANSITIONS=6144;HARD_PROBABILITY=.5
        if profile=='anchor_ab18':
            ARMS=('weak_anchor','full_anchor');SEED=1110000;STEPS=245760;SECONDS=2400
            INTERVAL=61440
        elif profile=='recovery_replay18':
            SOURCE=ROOT/'outputs/push_replay18_preparation_20261111'
            ARMS=('standing_only','all_success');SEED=1150000;STEPS=245760;SECONDS=2400
            LEARNING_RATE=3e-6;ACTION_STD=[.01]*4+[.10]*2;CLIP_RANGE=.03;TARGET_KL=.003
            INTERVAL=61440
        elif profile=='wheel_recovery18':
            SOURCE=ROOT/'outputs/push_replay18_preparation_20261111'
            ARMS=('wheel_head','all_actor');SEED=1180000;STEPS=245760;SECONDS=2400
            LEARNING_RATE=3e-5;ACTION_STD=[.0015]*4+[.15]*2;CLIP_RANGE=.03;TARGET_KL=.003
            INTERVAL=61440


def setup():
    torch.set_num_threads(1)
    common.configure_profile('stability18')
    common.SELECT_SEED=1010000 if PROFILE=='exploration_ab18' else 980000
    common.TEST_SEED=1020000 if PROFILE=='exploration_ab18' else 990000
    if PROFILE=='focused18':common.SELECT_SEED=1090000;common.TEST_SEED=1100000
    if PROFILE=='anchor_ab18':common.SELECT_SEED=1120000;common.TEST_SEED=1130000
    if PROFILE=='recovery_replay18':common.SELECT_SEED=1160000;common.TEST_SEED=1170000
    if PROFILE=='wheel_recovery18':common.SELECT_SEED=1190000;common.TEST_SEED=1200000
    common.TARGET=18.
    if sha(PARENT)!=PARENT_SHA:raise RuntimeError('Unexpected parent checkpoint')
    for path in (CFG,common.NOMINAL,common.TEST):validate_artifacts(path)


def rehearsal_settings(folder,arm):
    dataset=folder/arm/'successful_standing.npz'
    return dict(enabled=arm!='no_rehearsal',dataset=str(dataset),dataset_sha256=sha(dataset),
        updates_per_rollout=(4 if PROFILE in ('recovery_replay18','wheel_recovery18') else 2 if arm=='weak_anchor' else 16),reference_transitions=4096,batch_size=256,
        learning_rate=1e-6,max_grad_norm=1.,action_error_scales=[.025]*4+[.25]*2)


def prepare(folder):
    config=load_config(CFG);xml=ROOT/config['experiment']['output_directory']/'robot_experiment.xml'
    report=json.loads((SOURCE/'rehearsal/successful_standing.json').read_text())
    with np.load(SOURCE/'rehearsal/successful_standing.npz',allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in ('observations','actions','weights')}
    mask=np.zeros(len(arrays['actions']),dtype=bool);cursor=0
    for e in report['outcomes']:
        if not (e['anchor'] and e['success']):continue
        count=int(round(e['seconds']/.01))/4
        if count!=int(count):raise RuntimeError('Nonintegral source sample count')
        count=int(count)
        if e['push_probability']==0.:mask[cursor:cursor+count]=True
        cursor+=count
    if cursor!=len(mask) or mask.sum()<1000:raise RuntimeError('Source episode/sample alignment failed')
    for arm in ARMS:
        destination=folder/arm;destination.mkdir(parents=True,exist_ok=False)
        all_success=arm in ('all_success','wheel_head','all_actor')
        selected=np.ones(len(mask),dtype=bool) if all_success else mask
        dataset=destination/'successful_standing.npz'
        np.savez_compressed(dataset,**{k:v[selected] for k,v in arrays.items()})
        shutil.copy2(SOURCE/'rehearsal/config.yaml',destination/'config.yaml')
        metadata=dict(report,dataset_sha256=sha(dataset),samples=int(selected.sum()),
            source_dataset_sha256=sha(SOURCE/'rehearsal/successful_standing.npz'),
            filter='all successful anchor episodes' if all_success else 'successful no-push episodes only',
            reused_training_only=True)
        write(dataset.with_suffix('.json'),metadata)
        load_standing_rehearsal(rehearsal_settings(folder,arm),config,xml)
    cases=json.loads((SOURCE/'hard_training_resets.json').read_text())['cases']
    no_push=[dict(seed=e['seed'],force_N=[2.,15.],push=False,expected_push_events=[])
        for e in report['outcomes'] if not e['success'] and e['push_probability']==0.]
    if not no_push and PROFILE not in FOCUSED_PROFILES:raise RuntimeError('No training-only unstable standing reset')
    # Half of the hard pool is no-push standing failure; half has actual pushes.
    balanced=cases+no_push if PROFILE in FOCUSED_PROFILES else cases+no_push*len(cases)
    write(folder/'hard_training_resets.json',dict(cases=balanced,probability=HARD_PROBABILITY,
        source='prior training discovery only; no selection/test cases'))
    write(folder/'preparation.json',dict(passed=True,parent_sha256=PARENT_SHA,
        source_training_seeds=report['training_seed_range'],all_success_samples=len(mask),
        standing_only_samples=int(mask.sum()),hard_push_cases=len(cases),hard_standing_cases=len(no_push),
        selection_seed=common.SELECT_SEED,test_seed=common.TEST_SEED,profile=PROFILE))


def factory(folder,seed,monitor=None):
    cases=json.loads((folder/'hard_training_resets.json').read_text())['cases']
    focused=PROFILE in FOCUSED_PROFILES;hard_probability=HARD_PROBABILITY
    def create():
        config=load_config(CFG)
        env=make_mujoco_env(ROOT/config['experiment']['output_directory']/'robot_experiment.xml',CFG)
        base=env.unwrapped;base.getup['stand_probability']=1.
        base.push.config=dict(base.push.config,episode_probability=.6,force_N=[2.,18.],
            force_bands=[dict(force_N=[2.,10.],probability=.2),
                         dict(force_N=[10.,15.],probability=.3),
                         dict(force_N=[15.,18.],probability=.5)])
        if focused:
            base.push.config.update(episode_probability=.75,force_bands=[
                dict(force_N=[2.,10.],probability=.1),
                dict(force_N=[10.,15.],probability=.2),
                dict(force_N=[15.,18.],probability=.3),
                dict(force_N=[18.,18.],probability=.4)])
            base.balance_recovery.settings=dict(base.balance_recovery.settings,
                tilt_weight=8.,tilt_rate_weight=2.,heading_weight=1.5,yaw_rate_weight=1.)
        env.reset(seed=seed)
        env=ExactPushResetReplay(env,cases,hard_probability,seed+1000003)
        return Monitor(ScaledReward(env,.01),filename=monitor)
    return create


def restrict_to_wheel_head(policy):
    """Freeze features and leg outputs; both PPO and rehearsal obey the mask.

    Hooks are training-only and are reinstalled by learner, never needed to play
    a checkpoint. Adam is reset before this function and uses no weight decay.
    """
    actor=policy.mlp_extractor.policy_net
    actor.standing_net.requires_grad_(False)
    actor.standing_head.requires_grad_(True)
    def wheel_rows(gradient):
        result=torch.zeros_like(gradient)
        result[4:]=gradient[4:]
        return result
    actor.standing_head.weight.register_hook(wheel_rows)
    actor.standing_head.bias.register_hook(wheel_rows)


def wheel_contract(policy,reference):
    current=policy.mlp_extractor.policy_net;before=reference.mlp_extractor.policy_net
    assert all(torch.equal(value,before.standing_net.state_dict()[name])
        for name,value in current.standing_net.state_dict().items()),'Standing features changed'
    assert torch.equal(current.standing_head.weight[:4],before.standing_head.weight[:4]),'Leg output weights changed'
    assert torch.equal(current.standing_head.bias[:4],before.standing_head.bias[:4]),'Leg output biases changed'
    changed=not (torch.equal(current.standing_head.weight[4:],before.standing_head.weight[4:])
        and torch.equal(current.standing_head.bias[4:],before.standing_head.bias[4:]))
    return dict(features_and_leg_outputs_identical=True,wheel_outputs_changed=changed)


def learner(folder,arm,vector,smoke=False):
    settings=copy.deepcopy(load_config(CFG)['training'])
    if ROLLOUT_TRANSITIONS%vector.num_envs:raise ValueError('Rollout batch must divide environment count')
    smoke_steps=1024 if PROFILE in ('anchor_ab18','recovery_replay18','wheel_recovery18') else 256
    settings.update(n_steps=smoke_steps if smoke else ROLLOUT_TRANSITIONS//vector.num_envs,batch_size=128 if smoke else 512,
        n_epochs=EPOCHS,learning_rate=LEARNING_RATE,freeze_rising_actor=True,freeze_standing_actor=False,
        freeze_action_std=True,reset_optimizer_on_warm_start=True,target_kl=TARGET_KL,clip_range=CLIP_RANGE)
    model,_,_=load_warm_start(PARENT,vector,settings,'cpu',SEED)
    model.configure_estimator(dict(settings['velocity_estimator'],trainable=False))
    model.configure_gradient_clipping(True)
    if arm=='wheel_head':restrict_to_wheel_head(model.policy)
    scale=float(getattr(model,'training_reward_scale',1.))
    with torch.no_grad():
        model.policy.value_net.weight.mul_(.01/scale);model.policy.value_net.bias.mul_(.01/scale)
        model.policy.log_std.copy_(torch.log(torch.tensor(ACTION_STD)))
    model.training_reward_scale=.01
    model.configure_standing_rehearsal(rehearsal_settings(folder,arm),load_config(CFG),
        ROOT/load_config(CFG)['experiment']['output_directory']/'robot_experiment.xml',SEED)
    return model,settings


def check(folder):
    reference=PPO.load(PARENT,device='cpu');results=[];initial_actions=[]
    for arm in ARMS:
        vector=DummyVecEnv([factory(folder,SEED)])
        try:
            model,settings=learner(folder,arm,vector,True)
            initial_actions.append(model.predict(vector.reset(),deterministic=True)[0])
            model.set_logger(configure(str(folder/arm/'preflight_optimizer'),['csv']))
            model.learn(total_timesteps=1024,progress_bar=False)
            contract=protected(model,reference)
            wheel=wheel_contract(model.policy,reference.policy) if arm=='wheel_head' else None
            if wheel is not None and not wheel['wheel_outputs_changed']:raise RuntimeError('Wheel head did not update')
            if not contract['changed_standing_tensors']:raise RuntimeError('Standing did not update')
            enabled=getattr(model,'_standing_rehearsal',None) is not None
            if enabled!=(arm!='no_rehearsal'):raise RuntimeError('Rehearsal arm was not applied')
            if enabled:
                expected=max(1,round(rehearsal_settings(folder,arm)['updates_per_rollout']*
                    model.n_steps*vector.num_envs/4096))
                if model._standing_rehearsal.last_updates!=expected:raise RuntimeError('Rehearsal update budget differs')
            if not all(torch.isfinite(p).all() for p in model.policy.parameters()):raise RuntimeError('Nonfinite update')
            results.append(dict(arm=arm,contract=contract,wheel_contract=wheel,discarded_steps=1024,rehearsal_enabled=enabled,
                rehearsal_updates_per_4096=rehearsal_settings(folder,arm)['updates_per_rollout'],
                actual_smoke_rehearsal_updates=(model._standing_rehearsal.last_updates if enabled else 0),
                learning_rate=settings['learning_rate'],action_std=model.policy.log_std.exp().detach().tolist()))
        finally:vector.close()
    for action in initial_actions[1:]:np.testing.assert_array_equal(initial_actions[0],action)
    # Verify the training-only failed standing reset really disables force.
    env=factory(folder,SEED)();wrapper=env
    while not isinstance(wrapper,ExactPushResetReplay):wrapper=wrapper.env
    reward_check=None
    if PROFILE in FOCUSED_PROFILES:
        expected=dict(tilt_weight=8.,tilt_rate_weight=2.,heading_weight=1.5,yaw_rate_weight=1.)
        actual={key:env.unwrapped.balance_recovery.settings[key] for key in expected}
        if actual!=expected:raise RuntimeError('Focused recovery rewards were not applied')
        reward_check=actual
    wrapper.cases=[c for c in wrapper.cases if c.get('push') is False]
    if not wrapper.cases:wrapper.cases=[dict(seed=1070000,force_N=[2.,18.],push=False)]
    wrapper.probability=.9;wrapper.rng=np.random.default_rng(12)
    try:
        obs,info=env.reset();actual=0.
        while True:
            obs,_,done,truncated,info=env.step(reference.predict(obs,deterministic=True)[0])
            actual=max(actual,float(np.linalg.norm(info['push_force_world_N'])))
            if done or truncated:break
        if actual!=0.:raise RuntimeError('Standing hard case applied a push')
        standing=dict(seed=wrapper.cases[0]['seed'],seconds=env.unwrapped.steps*env.unwrapped.dt,
            reason=info['termination_reason'],maximum_force_N=actual)
    finally:env.close()
    env=factory(folder,SEED)();wrapper=env
    while not isinstance(wrapper,ExactPushResetReplay):wrapper=wrapper.env
    wrapper.original.update(episode_probability=1.,force_N=[18.,18.])
    wrapper.original.pop('force_bands',None)
    try:
        obs,_=env.reset(seed=630000,options={'reset_mode':'standing'});seen=[]
        for _ in range(320):
            obs,_,done,truncated,info=env.step(reference.predict(obs,deterministic=True)[0])
            magnitude=float(np.linalg.norm(info['push_force_world_N']))
            if magnitude:seen.append(magnitude)
            if done or truncated:break
        if not seen or not np.allclose(seen,18.):raise RuntimeError('Physical 18 N pulse not applied')
        force=dict(measured_force_N=max(seen),pulse_steps=len(seen),warnings=int(sum(env.unwrapped.data.warning.number)))
    finally:env.close()
    write(folder/'preflight.json',dict(passed=True,profile=PROFILE,script_sha256=sha(__file__),
        replay_sha256=sha(ROOT/'source/climber_legged_rl/push_reset_replay.py'),
        parent_sha256=PARENT_SHA,arms=results,num_envs=NUM_ENVS,
        rollout_transitions=ROLLOUT_TRANSITIONS,paired_initial_actions_identical=True if len(ARMS)>1 else None,
        training_only_standing_replay=standing,force_check=force,recovery_reward_overrides=reward_check))


def meaningful(result,baseline):
    groups=('standing','accepted15','level','nominal_getup','difficult_getup')
    if 'exact18' in baseline['summary']:
        groups=(*groups,'exact18')
        if result['summary']['exact18']['success']<baseline['summary']['exact18']['success']:return False
    return common.retained(result,baseline) and any(
        result['summary'][g]['success']>baseline['summary'][g]['success'] for g in groups)


def selection_score(result):
    original=common.score(result)
    if 'exact18' in result['summary']:
        exact=result['summary']['exact18']
        return [exact['success'],*original,exact['recovered_push_count']]
    return original


class Assessor(common.Assessor):
    def evaluate(self,checkpoint,level,split='selection',physics=False):
        specs=common.specifications(level,split)
        if PROFILE in ('anchor_ab18','recovery_replay18','wheel_recovery18') and split=='selection':
            specs += [dict(group='exact18',config=str(common.NOMINAL),mode='standing',push=True,
                force_N=[18.,18.],seed=common.SELECT_SEED+200+i) for i in range(12)]
        started=time.monotonic()
        rows=list(self.pool.map(common.evaluate_case,[(str(checkpoint),s,physics) for s in specs]))
        return dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),level_N=level,split=split,
            episodes=rows,summary=common.summarize(rows),seconds=time.monotonic()-started,
            physics_step_spring_measurement=physics)


class Selection(BaseCallback):
    def __init__(self,folder,arm,assessor,parent,baseline,best,status,start,consumed):
        super().__init__();self.folder=folder;self.arm=arm;self.assessor=assessor
        self.parent=parent;self.baseline=baseline;self.best=best;self.status=status
        self.start=start;self.consumed=consumed;self.next=INTERVAL;self.curve=[]
    def assess(self,final=False):
        suffix='_final' if final else ''
        cp=self.folder/f'level18_attempt{ARMS.index(self.arm)}_step{self.num_timesteps}{suffix}.zip'
        protected(self.model,self.parent);self.model.save(cp)
        if self.arm=='wheel_head':wheel_contract(self.model.policy,self.parent.policy)
        self.status('selection',arm=self.arm,attempt=ARMS.index(self.arm),steps=self.consumed+self.num_timesteps)
        result=self.assessor.evaluate(cp,18.);write(cp.with_suffix('.json'),result)
        accept=meaningful(result,self.baseline) and selection_score(result)>selection_score(self.best)
        if accept:
            self.best=result;shutil.copy2(cp,self.folder/'selected_policy.zip')
            write(self.folder/'selected_selection.json',result)
        self.curve.append(dict(steps=self.consumed+self.num_timesteps,arm=self.arm,
            attempt=ARMS.index(self.arm),accepted=accept,checkpoint=str(cp),
            summary=result['summary'],final_optimizer_state=final))
        write(self.folder/f'curve_18_{ARMS.index(self.arm)}.json',self.curve)
        self.status('training',arm=self.arm,steps=self.consumed+self.num_timesteps,accepted=accept,summary=result['summary'])
    def _on_step(self):
        if (self.folder/'STOP').exists() or time.monotonic()-self.start>SECONDS:return False
        if self.num_timesteps>=self.next:self.next+=INTERVAL;self.assess()
        return True


def run(folder):
    lock=(ROOT/'outputs/spring208_phase1_launch/workflow.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'workflow.json').exists():raise RuntimeError('Refusing to overwrite a run')
    checked=json.loads((folder/'preflight.json').read_text())
    if not checked['passed'] or checked['script_sha256']!=sha(__file__):raise RuntimeError('Stale preflight')
    if checked.get('profile','recovery_ab18')!=PROFILE:raise RuntimeError('Preflight profile differs')
    if checked.get('num_envs',4)!=NUM_ENVS:raise RuntimeError('Preflight environment count differs')
    if checked['replay_sha256']!=sha(ROOT/'source/climber_legged_rl/push_reset_replay.py'):raise RuntimeError('Replay changed')
    start=time.monotonic();state=dict(pid=os.getpid(),started_unix=time.time(),profile=PROFILE,
        target_force_N=18.,long_term_target_N=150.,level_N=18.,max_steps=STEPS*len(ARMS),
        max_training_seconds=SECONDS,simulation_only=True,auto_hardware_deployment=False)
    def status(phase,**extra):
        state.update(state=phase,elapsed_seconds=time.monotonic()-start,**extra)
        write(folder/'workflow.json',state);print(json.dumps(state),flush=True)
    assessor=Assessor();parent=PPO.load(PARENT,device='cpu');total=0
    try:
        write(folder/'protocol.json',dict(profile=PROFILE,parent=str(PARENT),parent_sha256=PARENT_SHA,
            arms=list(ARMS),one_variable=('trainable parameters: wheel output rows only vs full standing actor' if PROFILE=='wheel_recovery18' else
                ('standing-only vs successful standing and push-recovery rehearsal' if PROFILE=='recovery_replay18' else
                ('standing rehearsal updates: 2 vs 16 per 4096 transitions' if PROFILE=='anchor_ab18' else
                ('single-arm targeted recovery continuation' if PROFILE=='focused18' else
                ('successful rehearsal disabled vs no-push episodes only'
                if PROFILE=='exploration_ab18' else 'successful rehearsal data: all episodes vs no-push episodes only'))))),
            paired_training_seed=SEED,selection_seed=common.SELECT_SEED,test_seed=common.TEST_SEED,
            steps_per_arm=STEPS,learning_rate=LEARNING_RATE,action_std=ACTION_STD,
            n_epochs=EPOCHS,clip_range=CLIP_RANGE,target_kl=TARGET_KL,
            native_rewards_and_acceptance=PROFILE not in FOCUSED_PROFILES,native_acceptance=True,
            recovery_reward_overrides=(dict(tilt_weight=8.,tilt_rate_weight=2.,heading_weight=1.5,yaw_rate_weight=1.)
                if PROFILE in FOCUSED_PROFILES else {}),
            fixed18_in_selection=PROFILE in ('anchor_ab18','recovery_replay18','wheel_recovery18'),
            selection_order='fixed18 success first' if PROFILE in ('anchor_ab18','recovery_replay18','wheel_recovery18') else 'random force band success first',
            num_envs=NUM_ENVS,rollout_transitions=ROLLOUT_TRANSITIONS,
            reward_scale=.01,model_and_motor_limits_unchanged=True,
            exact_hard_reset_probability=HARD_PROBABILITY,holdout_not_replayed=True,
            acceptance_requires_success_count_gain=True,final_test_not_used_for_selection=True,
            current_stage_passed_required_for_promotion=True,automatic_velocity_training=False))
        shutil.copy2(PARENT,folder/'selected_policy.zip')
        status('level_baseline',steps=0)
        baseline=assessor.evaluate(PARENT,18.);write(folder/'parent_level18.json',baseline)
        best=baseline;write(folder/'selected_selection.json',baseline)
        for i,arm in enumerate(ARMS):
            if (folder/'STOP').exists() or time.monotonic()-start>SECONDS:break
            # Both arms start from exactly the same parent and training seeds.
            vector=SubprocVecEnv([factory(folder,SEED+j,str(folder/f'monitor_18_{i}_{j}.csv')) for j in range(NUM_ENVS)],start_method='spawn')
            try:
                model,settings=learner(folder,arm,vector)
                write(folder/f'settings_18_{i}.json',dict(settings,arm=arm,rehearsal=rehearsal_settings(folder,arm)))
                model.set_logger(configure(str(folder/f'logs_18_{i}'),['csv']))
                callback=Selection(folder,arm,assessor,parent,baseline,best,status,start,total)
                status('training',arm=arm,attempt=i,steps=total)
                model.learn(total_timesteps=STEPS,callback=callback,reset_num_timesteps=True,progress_bar=False)
                callback.assess(final=True);total+=model.num_timesteps;best=callback.best
            finally:vector.close()
        status('final_independent_test',steps=total)
        before=assessor.evaluate(PARENT,18.,'test',True);write(folder/'parent_test.json',before)
        identical=sha(folder/'selected_policy.zip')==sha(PARENT)
        if identical:
            after=copy.deepcopy(before)
            after.update(checkpoint=str(folder/'selected_policy.zip'),reused_identical_checkpoint_evaluation=True)
        else:after=assessor.evaluate(folder/'selected_policy.zip',18.,'test',True)
        write(folder/'selected_test.json',after)
        physics_ok=not any(e['warnings'] or max(e['spring_upper_mm']+e['spring_lower_mm'])>1. for e in after['episodes'])
        passed=common.pass_level(after,before) and after['summary']['exact18']['success']/24>=.9 and physics_ok
        write(folder/'assessment.json',dict(target_force_N=18.,target_passed=passed,target_150N_passed=False,
            steps=total,seconds=time.monotonic()-start,parent_summary=before['summary'],selected_summary=after['summary'],
            selected_checkpoint=str(folder/'selected_policy.zip'),selected_sha256=sha(folder/'selected_policy.zip'),
            parent_sha256=PARENT_SHA,physical_checks_passed=physics_ok,simulation_only=True,
            selected_identical_to_parent=identical,
            protected_actor=protected(PPO.load(folder/'selected_policy.zip',device='cpu'),parent),
            automatic_velocity_training=False))
        status('completed',steps=total,target_passed=passed,target_150N_passed=False)
    except BaseException as error:status('failed',error=repr(error));raise
    finally:assessor.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--profile',choices=['recovery_ab18','exploration_ab18',*FOCUSED_PROFILES],default='recovery_ab18')
    parser.add_argument('--num-envs',type=int,choices=[4,6,8],default=4)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--check',action='store_true')
    args=parser.parse_args();configure_profile(args.profile);NUM_ENVS=args.num_envs
    setup();folder=args.folder.resolve();folder.mkdir(parents=True,exist_ok=True)
    if args.prepare:prepare(folder)
    elif args.check:check(folder)
    else:run(folder)
