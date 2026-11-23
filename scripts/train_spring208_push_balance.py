"""Bounded six-actuator PPO: balance first, strict positional scoring in parallel."""
import argparse
import copy
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'source'), str(ROOT / 'scripts')]
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.push_balance_curriculum import PushBalanceCurriculum, STAGES
from climber_legged_rl.push_reset_replay import ExactPushResetReplay
from climber_legged_rl.motor_authority import apply_standing_motor_limit
from climber_legged_rl.recovery_force_curriculum import RecoveryForceCurriculum, FORCE_SCHEDULE
from climber_legged_rl.early_recovery_reward import EarlyRecoveryReward, EARLY_RECOVERY_SETTINGS, BALANCED_RECOVERY_SETTINGS
from climber_legged_rl.recovery_clearance_reward import RecoveryClearanceReward, SETTINGS as CLEARANCE_SETTINGS
from push_recovery_rehearsal import ScaledReward
import train_spring208_push_ab as ab
import train_spring208_push150 as common
from train_spring208_push import CFG, evaluate_case, summarize, protected, write
from prepare_spring208_phase3 import sha
from directional_recovery_replay import build_directional_replay, failure_weight

PARENT = ROOT / 'outputs/push_wheel_recovery18_20261113_2020/selected_policy.zip'
PARENT_SHA = 'cd6d6eb1674e4a0753c91ad710f8b679c368256cd9402007c1262ca29f39a144'
STEPS = 491520
INTERVAL = 98304
MAX_SECONDS = 2100
TRAIN_SEED = 1210000
STAGE = 'balance'
PROFILE = 'rated_balance18'
DAMPED_WHEEL_GAIN = .4
RECOVERY_WEIGHTS = dict(tilt_weight=8., tilt_rate_weight=2., velocity_weight=3., yaw_rate_weight=1.)
REPEAT_FORCE_SCHEDULE = ((0, 15.), (12288, 16.5), (24576, 18.))
QUALITY_PROFILES = ('damped_recovery24', 'catch_recovery24')
TARGET24_PROFILES = ('focused_recovery24', *QUALITY_PROFILES)
PROGRESSIVE_PROFILES = ('progressive_balance30', *TARGET24_PROFILES)
RETENTION_PROFILES = ('standing_retention18', *PROGRESSIVE_PROFILES)
RATED_PROFILES = ('rated_balance18', 'reactive_balance18', 'recovery_curriculum18',
                  'repeat_recovery18', 'early_recovery18', 'balanced_recovery18', 'wheel_response18',
                  'directional_recovery18', 'short_directional18', *RETENTION_PROFILES)
DIRECTIONAL_PROFILES = ('directional_recovery18', 'short_directional18', *RETENTION_PROFILES)
CONTROLLER_RESPONSE_PROFILES = ('wheel_response18', *DIRECTIONAL_PROFILES)
BALANCED_SELECTION_PROFILES = ('balanced_recovery18', *CONTROLLER_RESPONSE_PROFILES)
MIXED_RECOVERY_PROFILES = ('early_recovery18', *BALANCED_SELECTION_PROFILES)


def target_force():
    if PROFILE in TARGET24_PROFILES:
        return 24.
    return 30. if PROFILE == 'progressive_balance30' else 18.


def mixed_force_range():
    return [22., 24.] if PROFILE in TARGET24_PROFILES else [18., 30.] if PROFILE == 'progressive_balance30' else [15., 18.]


def target_group():
    return f'exact{target_force():g}'


def compare_original_controller():
    return PROFILE in CONTROLLER_RESPONSE_PROFILES and PROFILE not in PROGRESSIVE_PROFILES


def early_reward_settings():
    if PROFILE == 'balanced_recovery18':
        return dict(BALANCED_RECOVERY_SETTINGS)
    return dict(EARLY_RECOVERY_SETTINGS) if PROFILE == 'early_recovery18' else None


def force_schedule():
    if PROFILE == 'catch_recovery24':
        return ((0, 18.), (6144, 21.), (12288, 24.))
    if PROFILE in TARGET24_PROFILES:
        return ((0, 22.), (6144, 24.))
    if PROFILE == 'progressive_balance30':
        return ((0, 18.), (3072, 22.), (9216, 26.), (15360, 30.))
    if PROFILE == 'standing_retention18':
        return ((0, 15.), (3072, 18.))
    if PROFILE == 'short_directional18':
        return ((0, 12.), (6144, 15.), (12288, 18.))
    if PROFILE in ('repeat_recovery18', 'early_recovery18', 'balanced_recovery18'):
        return REPEAT_FORCE_SCHEDULE
    return FORCE_SCHEDULE if PROFILE in ('recovery_curriculum18', *CONTROLLER_RESPONSE_PROFILES) else ()


def failure_sampling_weight(events):
    if PROFILE in DIRECTIONAL_PROFILES:
        return failure_weight(events)
    if PROFILE in MIXED_RECOVERY_PROFILES:
        if len(events) >= 2 and events[0]['recovered'] and not events[-1]['recovered']:
            return 3
        if events and all(e['recovered'] for e in events):
            return 2
        return 3 if events and events[-1]['force_world_N'][0] < 0 else 1
    if PROFILE == 'repeat_recovery18':
        if len(events) >= 2 and events[0]['recovered'] and not events[-1]['recovered']:
            return 6
        if events and all(e['recovered'] for e in events):
            return 4
    if PROFILE in ('reactive_balance18', 'recovery_curriculum18', 'repeat_recovery18'):
        return 3 if events and events[-1]['force_world_N'][0] < 0 else 1
    return 1


def motor_options():
    options = dict(standing_leg_torque_limit_Nm=20.) if PROFILE in RATED_PROFILES else {}
    if PROFILE in CONTROLLER_RESPONSE_PROFILES:
        options['standing_wheel_kv'] = DAMPED_WHEEL_GAIN if PROFILE == 'damped_recovery24' else .4
    return options


def clearance_reward_settings():
    if PROFILE not in QUALITY_PROFILES:
        return None
    settings = copy.deepcopy(CLEARANCE_SETTINGS)
    if PROFILE == 'catch_recovery24':
        # Let the legs/wheels catch the body before rewarding a stop. This is
        # an experiment in reward priorities, never a replacement controller.
        settings.update(clearance_guard_m=.030, clearance_weight=4.,
            curvature_weight=0., transient_leg_length_fraction=.25,
            transient_linear_tracking_fraction=.25, transient_velocity_fraction=.25)
    return settings


def sustained_profile():
    return PROFILE in ('sustained_balance18', *RATED_PROFILES)


def setup(stage, profile='rated_balance18', wheel_kv=None):
    global STAGE, PROFILE, TRAIN_SEED, PARENT, PARENT_SHA, STEPS, INTERVAL, MAX_SECONDS, DAMPED_WHEEL_GAIN
    if wheel_kv is not None and (profile != 'damped_recovery24' or wheel_kv not in (.3, .4, .5)):
        raise ValueError('Gain override is supported only for the damped recovery comparison')
    DAMPED_WHEEL_GAIN = .4 if wheel_kv is None else float(wheel_kv)
    STAGE = stage
    PROFILE = profile
    STEPS, INTERVAL, MAX_SECONDS = 491520, 98304, 2100
    TRAIN_SEED = 1240000 if sustained_profile() else 1210000
    PARENT = ROOT / 'outputs/push_wheel_recovery18_20261113_2020/selected_policy.zip'
    PARENT_SHA = 'cd6d6eb1674e4a0753c91ad710f8b679c368256cd9402007c1262ca29f39a144'
    if profile in ('reactive_balance18', 'recovery_curriculum18'):
        PARENT = ROOT / 'outputs/push_rated_balance18_20261115/selected_policy.zip'
        PARENT_SHA = '48daf747afffef24ffffa1e44ddef3197963e505c6a68f9be888b171eea5e9e2'
        TRAIN_SEED = 1300000 if profile == 'recovery_curriculum18' else 1270000
    elif profile in ('repeat_recovery18', *MIXED_RECOVERY_PROFILES):
        PARENT = ROOT / 'outputs/push_recovery_curriculum18_20261115/selected_policy.zip'
        PARENT_SHA = '49c09b5c72dd43f380b2814e6884f3954d1726904cb49cae14376eaad0b6c4b1'
        TRAIN_SEED = (1460000 if profile == 'directional_recovery18' else 1430000 if profile == 'wheel_response18' else 1390000 if profile == 'balanced_recovery18'
                      else 1360000 if profile == 'early_recovery18' else 1330000)
    if profile in ('short_directional18', *RETENTION_PROFILES):
        PARENT = ROOT / 'outputs/push_directional_recovery18_20261116/selected_policy.zip'
        PARENT_SHA = 'ec7fee00619e4240a2c6d93c1a247ec74f972a47b34deb3a8ad8e2a1478f696e'
        TRAIN_SEED = 1490000
        STEPS, INTERVAL, MAX_SECONDS = 196608, 49152, 1500
        if profile == 'standing_retention18':
            TRAIN_SEED = 1520000
            MAX_SECONDS = 2400
        elif profile == 'progressive_balance30':
            TRAIN_SEED = 1550000
            STEPS, INTERVAL, MAX_SECONDS = 294912, 73728, 3000
        elif profile == 'focused_recovery24':
            PARENT = ROOT / 'outputs/push_progressive_balance30_20261116/selected_policy.zip'
            PARENT_SHA = 'c8b5da09547f70957031d358ae067d0c00761b7b45b949e26971044683ddd314'
            TRAIN_SEED = 1580000
            STEPS, INTERVAL, MAX_SECONDS = 294912, 73728, 3000
        elif profile == 'damped_recovery24':
            PARENT = ROOT / 'outputs/push_focused_recovery24_20261116/selected_policy.zip'
            PARENT_SHA = 'c8b5da09547f70957031d358ae067d0c00761b7b45b949e26971044683ddd314'
            TRAIN_SEED = 1620000
            STEPS, INTERVAL, MAX_SECONDS = 294912, 73728, 3600
        elif profile == 'catch_recovery24':
            # The previous candidate regressed on independent 18N cases.
            PARENT = ROOT / 'outputs/push_focused_recovery24_20261116/selected_policy.zip'
            PARENT_SHA = 'c8b5da09547f70957031d358ae067d0c00761b7b45b949e26971044683ddd314'
            TRAIN_SEED = 1650000
            STEPS, INTERVAL, MAX_SECONDS = 294912, 73728, 3600
    ab.configure_profile('wheel_recovery18')
    ab.PARENT = PARENT
    ab.PARENT_SHA = PARENT_SHA
    ab.ARMS = ('all_actor',)
    ab.NUM_ENVS = 8
    ab.SEED = TRAIN_SEED
    ab.LEARNING_RATE = 3e-5 if profile == 'reactive_balance18' else 1e-5
    ab.ACTION_STD = ([.03] * 4 + [.20] * 2 if sustained_profile()
                     else [.015] * 4 + [.15] * 2)
    if profile == 'reactive_balance18':
        ab.ACTION_STD = [.08] * 4 + [.30] * 2
    elif profile == 'recovery_curriculum18':
        ab.ACTION_STD = [.04] * 4 + [.20] * 2
    elif profile == 'repeat_recovery18':
        ab.LEARNING_RATE = 3e-6
        ab.ACTION_STD = [.025] * 4 + [.15] * 2
    elif profile in MIXED_RECOVERY_PROFILES:
        ab.LEARNING_RATE = 5e-6
        ab.ACTION_STD = [.04] * 4 + [.20] * 2
    if profile in ('short_directional18', *RETENTION_PROFILES):
        ab.LEARNING_RATE = 3e-6
        ab.ACTION_STD = [.03] * 4 + [.18] * 2
    if profile == 'damped_recovery24':
        ab.LEARNING_RATE = 1e-5
        ab.ACTION_STD = [.05] * 4 + [.18] * 2
    elif profile == 'catch_recovery24':
        ab.LEARNING_RATE = 2e-5
        ab.ACTION_STD = [.08] * 4 + [.22] * 2
    ab.setup()
    common.TARGET = target_force()
    common.SELECT_SEED = 1250000 if sustained_profile() else 1220000
    common.TEST_SEED = 1260000 if sustained_profile() else 1230000
    if profile == 'reactive_balance18':
        common.SELECT_SEED, common.TEST_SEED = 1280000, 1290000
    elif profile == 'recovery_curriculum18':
        common.SELECT_SEED, common.TEST_SEED = 1310000, 1320000
    elif profile == 'repeat_recovery18':
        common.SELECT_SEED, common.TEST_SEED = 1340000, 1350000
    elif profile == 'early_recovery18':
        common.SELECT_SEED, common.TEST_SEED = 1370000, 1380000
    elif profile == 'balanced_recovery18':
        common.SELECT_SEED, common.TEST_SEED = 1400000, 1410000
    elif profile == 'wheel_response18':
        common.SELECT_SEED, common.TEST_SEED = 1440000, 1450000
    elif profile == 'directional_recovery18':
        common.SELECT_SEED, common.TEST_SEED = 1470000, 1480000
    elif profile == 'short_directional18':
        common.SELECT_SEED, common.TEST_SEED = 1500000, 1510000
    elif profile == 'standing_retention18':
        common.SELECT_SEED, common.TEST_SEED = 1530000, 1540000
    elif profile == 'progressive_balance30':
        common.SELECT_SEED, common.TEST_SEED = 1560000, 1570000
    elif profile == 'focused_recovery24':
        common.SELECT_SEED, common.TEST_SEED = 1590000, 1600000
    elif profile == 'damped_recovery24':
        common.SELECT_SEED, common.TEST_SEED = 1630000, 1640000
    elif profile == 'catch_recovery24':
        common.SELECT_SEED, common.TEST_SEED = 1660000, 1670000


def replay_probability():
    return .5 if PROFILE in DIRECTIONAL_PROFILES else .35 if PROFILE in MIXED_RECOVERY_PROFILES else .5


def retention_weights():
    if PROFILE in RETENTION_PROFILES:
        return dict(standing=7, fixed_negative_x=2, mixed_negative_x=3, mixed_other=2)
    return None


def training_push_probability():
    return .65 if PROFILE in RETENTION_PROFILES else .75


def rehearsal_settings(folder):
    settings = ab.rehearsal_settings(folder, 'all_actor')
    if sustained_profile():
        settings['updates_per_rollout'] = 1
    return settings


def learner(folder, vector, smoke=False):
    model, settings = ab.learner(folder, 'all_actor', vector, smoke=smoke)
    config = load_config(CFG)
    model.configure_standing_rehearsal(rehearsal_settings(folder), config,
        ROOT / config['experiment']['output_directory'] / 'robot_experiment.xml', TRAIN_SEED)
    return model, settings


def prepare(folder):
    ab.prepare(folder)
    if not sustained_profile():
        return
    # Discover failures only from new training seeds, never the displayed test
    # clips or selection cases. Include delayed falls after an event recovered.
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    discovery_count = 192 if PROFILE in DIRECTIONAL_PROFILES else 96 if PROFILE in MIXED_RECOVERY_PROFILES else 48
    no_push_count = 16 if PROFILE in DIRECTIONAL_PROFILES else 8
    if PROFILE in RETENTION_PROFILES:
        discovery_count, no_push_count = 256, 64
    upper = target_force()
    lower = mixed_force_range()[0]
    specs = [dict(group='training_discovery', config=str(CFG), mode='standing',
        push=i >= no_push_count, force_N=[lower, upper] if PROFILE in BALANCED_SELECTION_PROFILES and i % 2 else [upper, upper], seed=TRAIN_SEED + 1000 + i,
        balance_curriculum_stage=STAGE, **motor_options()) for i in range(discovery_count)]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        rows = list(pool.map(evaluate_case, [(str(PARENT), s, False) for s in specs]))
    cases = []
    negative_x = 0
    for row in rows:
        if row['success']:
            continue
        events = row['push_metrics']['push_events']
        delayed = bool(events and all(e['recovered'] for e in events))
        targeted = bool(events and events[-1]['force_world_N'][0] < 0)
        negative_x += int(targeted)
        case = dict(seed=row['spec']['seed'], force_N=list(row['spec']['force_N']),
            push=row['spec']['push'], expected_push_events=events,
            source='fresh training discovery', failure_reason=row['reason'],
            delayed_failure_after_recovery=delayed)
        case['failed_after_first_recovery'] = bool(len(events) >= 2 and events[0]['recovered'] and not events[-1]['recovered'])
        copies = failure_sampling_weight(events)
        cases.extend(copy.deepcopy(case) for _ in range(copies))
    if not cases or any(r['warnings'] for r in rows):
        raise RuntimeError('No usable training failures or numerical warnings during discovery')
    replay_metadata = {}
    if PROFILE in DIRECTIONAL_PROFILES:
        cases, replay_metadata = build_directional_replay(rows, TRAIN_SEED,
            (TRAIN_SEED + 1000, TRAIN_SEED + 999 + discovery_count), retention_weights())
    write(folder / 'training_discovery.json', dict(training_seed_range=[TRAIN_SEED+1000, TRAIN_SEED+999+discovery_count],
        holdouts_not_replayed=True, outcomes=rows,
        failed_cases=sum(not row['success'] for row in rows), weighted_replay_entries=len(cases),
        replay_composition=replay_metadata,
        negative_x_failure_cases=negative_x,
        repeated_push_failures=sum(not row['success'] and len(row['push_metrics']['push_events']) >= 2
            and row['push_metrics']['push_events'][0]['recovered']
            and not row['push_metrics']['push_events'][-1]['recovered'] for row in rows),
        delayed_failures=sum(not row['success'] and bool(row['push_metrics']['push_events'])
            and all(e['recovered'] for e in row['push_metrics']['push_events']) for row in rows), **motor_options()))
    write(folder / 'hard_training_resets.json', dict(cases=cases, probability=replay_probability(),
        replay_composition=replay_metadata,
        negative_x_failure_sampling_weight=6 if PROFILE in DIRECTIONAL_PROFILES else 3 if PROFILE in ('reactive_balance18', 'recovery_curriculum18', 'repeat_recovery18', *MIXED_RECOVERY_PROFILES) else 1,
        failed_after_first_recovery_sampling_weight=6 if PROFILE in DIRECTIONAL_PROFILES else 3 if PROFILE in MIXED_RECOVERY_PROFILES else 6 if PROFILE == 'repeat_recovery18' else None,
        delayed_failure_sampling_weight=4 if PROFILE in DIRECTIONAL_PROFILES else 2 if PROFILE in MIXED_RECOVERY_PROFILES else 4 if PROFILE == 'repeat_recovery18' else None,
        sampling_weights_combination='maximum applicable weight',
        source=('fresh training-only failures and successful retention; no selection/test seeds'
                if PROFILE in DIRECTIONAL_PROFILES else 'fresh training-only failures; no previous test seeds'),
        **motor_options()))


def source_hashes():
    return {name: sha(ROOT / name) for name in (
        'scripts/train_spring208_push_balance.py', 'scripts/train_spring208_push.py',
        'scripts/train_spring208_push_ab.py', 'source/climber_legged_rl/push_balance_curriculum.py',
        'scripts/directional_recovery_replay.py',
        'source/climber_legged_rl/push_reset_replay.py', 'source/climber_legged_rl/motor_authority.py',
        'source/climber_legged_rl/recovery_force_curriculum.py',
        'source/climber_legged_rl/early_recovery_reward.py',
        'source/climber_legged_rl/recovery_clearance_reward.py',
        'source/climber_legged_rl/policy_getup.py')}


def factory(folder, seed, stage, monitor=None):
    replay = json.loads((folder / 'hard_training_resets.json').read_text())
    cases = replay['cases']
    hard_probability = replay.get('probability', .5)
    sustained = sustained_profile()
    limits = motor_options()
    schedule = force_schedule()
    early_settings = early_reward_settings()
    balanced = PROFILE in BALANCED_SELECTION_PROFILES
    push_probability = training_push_probability()
    upper = target_force()
    clearance_settings = clearance_reward_settings()
    bands = ([dict(force_N=[10., 15.], probability=.1),
              dict(force_N=[18., 18.], probability=.2),
              dict(force_N=mixed_force_range(), probability=.4),
              dict(force_N=[upper, upper], probability=.3)] if PROFILE in PROGRESSIVE_PROFILES else
             [dict(force_N=[2., 10.], probability=.1),
              dict(force_N=[10., 15.], probability=.2),
              dict(force_N=[15., 18.], probability=.4 if balanced else .2 if sustained else .3),
              dict(force_N=[18., 18.], probability=.3 if balanced else .5 if sustained else .4)])
    def create():
        config = load_config(CFG)
        env = make_mujoco_env(ROOT / config['experiment']['output_directory'] / 'robot_experiment.xml', CFG)
        base = env.unwrapped
        apply_standing_motor_limit(env, **limits)
        base.getup['stand_probability'] = 1.
        base.push.config = dict(base.push.config, episode_probability=push_probability,
                               force_N=[2., upper], force_bands=copy.deepcopy(bands))
        if sustained:
            base.balance_recovery.settings = dict(base.balance_recovery.settings, **RECOVERY_WEIGHTS)
        env = PushBalanceCurriculum(env, stage, sustained_reward=2. if sustained else 0.)
        if early_settings is not None:
            env = EarlyRecoveryReward(env, early_settings)
        if clearance_settings is not None:
            env = RecoveryClearanceReward(env, clearance_settings)
        env.reset(seed=seed)
        env = ExactPushResetReplay(env, cases, hard_probability, seed + 1000003)
        if schedule:
            env = RecoveryForceCurriculum(env, schedule)
        return Monitor(ScaledReward(env, .01), filename=monitor,
                       info_keywords=('strict_success', 'training_force_cap_N') if schedule else ('strict_success',))
    return create


class Assessor(common.Assessor):
    def evaluate(self, checkpoint, split='selection', physics=False, motor_overrides=None):
        started = time.monotonic()
        specs = common.specifications(target_force(), split)
        if PROFILE in PROGRESSIVE_PROFILES:
            specs = [dict(s, force_N=mixed_force_range()) if s['group'] == 'level' else s for s in specs]
        if split == 'selection' and PROFILE in BALANCED_SELECTION_PROFILES:
            # More fresh selection cases; test seeds remain entirely separate.
            level = [s for s in specs if s['group'] == 'level']
            specs += [dict(s, seed=s['seed'] + len(level)) for s in level]
        if split == 'selection' and PROFILE in RETENTION_PROFILES:
            # Fresh standing seeds only; independent test seeds remain untouched.
            standing = [s for s in specs if s['group'] == 'standing']
            small = [s for s in specs if s['group'] == 'accepted15']
            specs += [dict(standing[0], seed=common.SELECT_SEED + i)
                      for i in range(len(standing), 64)]
            specs += [dict(small[0], seed=common.SELECT_SEED + 300 + i)
                      for i in range(len(small), 24)]
        if split == 'selection':
            specs += [dict(group=target_group(), config=str(common.NOMINAL), mode='standing', push=True,
                           force_N=[target_force(), target_force()], seed=common.SELECT_SEED + 200 + i)
                      for i in range(24 if PROFILE in BALANCED_SELECTION_PROFILES else 12)]
        if PROFILE in PROGRESSIVE_PROFILES:
            base_seed = common.SELECT_SEED if split == 'selection' else common.TEST_SEED
            specs += [dict(group='exact18', config=str(common.NOMINAL), mode='standing', push=True,
                           force_N=[18., 18.], seed=base_seed + 700 + i) for i in range(24)]
        limits = motor_options() if motor_overrides is None else motor_overrides
        specs = [dict(s, balance_curriculum_stage=STAGE, **limits) for s in specs]
        if PROFILE in QUALITY_PROFILES:
            specs = [dict(s, measure_recovery_quality=True) for s in specs]
        rows = list(self.pool.map(evaluate_case, [(str(checkpoint), s, physics) for s in specs]))
        strict = [dict(r, success=r['strict_success'], push_metrics=r['strict_push_metrics']) for r in rows]
        return dict(checkpoint=str(checkpoint), sha256=sha(checkpoint), level_N=target_force(), split=split,
            balance_curriculum_stage=STAGE, **limits, episodes=rows, summary=summarize(rows),
            strict_summary=summarize(strict), seconds=time.monotonic() - started,
            physics_step_spring_measurement=physics)


def eligible(result, baseline):
    # Protect standing/getup under the current curriculum. Strict position loss
    # is recorded, but deliberately does not veto stage-1 balance improvements.
    groups = ('standing', 'nominal_getup', 'difficult_getup')
    if PROFILE in ('repeat_recovery18', *MIXED_RECOVERY_PROFILES):
        groups += ('accepted15',)
    if PROFILE in BALANCED_SELECTION_PROFILES:
        groups += ('level', target_group())
    if PROFILE in PROGRESSIVE_PROFILES:
        groups += ('exact18',)
    return (not any(e['warnings'] for e in result['episodes'])
            and all(result['summary'][g]['success'] >= baseline['summary'][g]['success']
                    for g in groups))


def score(result):
    s = result['summary']
    pushed = [e for e in result['episodes'] if e['spec']['group'] in (target_group(), 'level')]
    if PROFILE in QUALITY_PROFILES:
        return [s[target_group()]['success'], s['level']['success'], s['accepted15']['success'],
                sum(e['push_metrics']['recovered_push_count'] for e in pushed),
                sum(e['seconds'] for e in pushed),
                -sum(e['recovery_quality']['tilt_rate_rms_rad_s'] for e in pushed),
                -sum(e['recovery_quality']['action_curvature_rms'] for e in pushed)]
    return [s[target_group()]['success'], s['level']['success'], s['accepted15']['success'],
            sum(e['push_metrics']['recovered_push_count'] for e in pushed),
            sum(e['seconds'] for e in pushed),
            -sum(e['final_errors']['speed_m_s'] for e in pushed),
            -sum(e['final_errors']['yaw_rate_rad_s'] for e in pushed)]


def ready(result):
    s = result['summary']
    return (not any(e['warnings'] for e in result['episodes'])
            and all(s[g]['success'] / s[g]['count'] >= .9 for g in (target_group(), 'level', 'accepted15'))
            and s['standing']['success'] / s['standing']['count'] >= .95
            and s['nominal_getup']['success'] == s['nominal_getup']['count'])


def preflight(folder):
    reference = PPO.load(PARENT, device='cpu')
    vector = DummyVecEnv([factory(folder, TRAIN_SEED, STAGE)])
    try:
        model, settings = learner(folder, vector, smoke=True)
        original = copy.deepcopy(model.policy.mlp_extractor.policy_net.standing_head.state_dict())
        model.set_logger(configure(str(folder / 'preflight_optimizer'), ['csv']))
        model.learn(total_timesteps=1024, progress_bar=False)
        contract = protected(model, reference)
        head = model.policy.mlp_extractor.policy_net.standing_head.state_dict()
        rows = [any(not torch.equal(head[k][i], original[k][i]) for k in head) for i in range(6)]
        if not all(rows) or not all(torch.isfinite(p).all() for p in model.policy.parameters()):
            raise RuntimeError('All six output rows must update finitely')
        expected = max(1, round(rehearsal_settings(folder)['updates_per_rollout'] * model.n_steps / 4096))
        if model._standing_rehearsal.last_updates != expected:
            raise RuntimeError('Rehearsal update budget was not applied')
    finally:
        vector.close()
    spec = dict(group=target_group(), config=str(common.NOMINAL), mode='standing', push=True,
                force_N=[target_force(), target_force()], seed=630000, **motor_options())
    native = evaluate_case((str(PARENT), spec, True))
    stage = evaluate_case((str(PARENT), dict(spec, balance_curriculum_stage=STAGE), True))
    for key in ('seconds', 'maximum', 'final_errors', 'spring_upper_mm', 'spring_lower_mm', 'warnings'):
        if native[key] != stage[key]:
            raise RuntimeError('Wrapper changed physical rollout: ' + key)
    if native['success'] != stage['strict_success'] or native['push_metrics'] != stage['strict_push_metrics']:
        raise RuntimeError('Native strict accounting changed')
    forces = [np.linalg.norm(e['force_world_N']) for e in stage['push_metrics']['push_events']]
    if not forces or not np.allclose(forces, target_force()) or stage['warnings']:
        raise RuntimeError(f'{target_force():g} N physical smoke check failed')
    training_env = factory(folder, TRAIN_SEED, STAGE)()
    try:
        training_env.reset(seed=TRAIN_SEED, options={'reset_mode': 'standing'})
        wrapper = training_env
        early_reward_present = False
        clearance_reward_present = False
        while not isinstance(wrapper, PushBalanceCurriculum):
            early_reward_present |= isinstance(wrapper, EarlyRecoveryReward)
            clearance_reward_present |= isinstance(wrapper, RecoveryClearanceReward)
            if isinstance(wrapper, EarlyRecoveryReward):
                assert wrapper.settings == early_reward_settings()
            if isinstance(wrapper, RecoveryClearanceReward):
                assert wrapper.settings == clearance_reward_settings()
            wrapper = wrapper.env
        if early_reward_present != (early_reward_settings() is not None):
            raise RuntimeError('Training recovery reward wrapper is missing or unexpected')
        if clearance_reward_present != (PROFILE in QUALITY_PROFILES):
            raise RuntimeError('Training clearance reward wrapper is missing or unexpected')
        expected_limit = motor_options().get('standing_leg_torque_limit_Nm', 10.)
        np.testing.assert_array_equal(training_env.unwrapped.torque_limits, [expected_limit] * 4 + [2.] * 2)
        assert training_env.unwrapped.control['wheel_kv'] == motor_options().get('standing_wheel_kv', .2)
        if sustained_profile():
            assert wrapper.sustained_reward == 2.
            assert all(training_env.unwrapped.balance_recovery.settings[k] == v for k,v in RECOVERY_WEIGHTS.items())
        if force_schedule():
            cap = force_schedule()[0][1]
            assert training_env.unwrapped.push.config['force_N'][1] == cap
            assert all(b['force_N'][1] <= cap for b in training_env.unwrapped.push.config['force_bands'])
        replay_wrapper = training_env
        while not isinstance(replay_wrapper, ExactPushResetReplay):
            replay_wrapper = replay_wrapper.env
        assert replay_wrapper.original['episode_probability'] == training_push_probability()
        if early_reward_present:
            _, reward, _, _, info = training_env.step(np.zeros(6))
            if not np.isfinite([reward, info['predicted_tilt_rad'], info['reward_terms']['early_recovery_risk']]).all():
                raise RuntimeError('Nonfinite early recovery reward')
    finally:
        training_env.close()
    write(folder / 'preflight.json', dict(passed=True, source_hashes=source_hashes(), stage=STAGE, profile=PROFILE,
        parent_sha256=PARENT_SHA, discarded_steps=1024, num_envs=8,
        all_six_output_rows_updated=rows, contract=contract,
        physical_rollout_identical=True, native_strict_accounting_identical=True,
        force_N=forces, settings_learning_rate=settings['learning_rate'],
        action_std=ab.ACTION_STD, rehearsal_updates_per_4096=rehearsal_settings(folder)['updates_per_rollout'],
        reward_overrides=RECOVERY_WEIGHTS if sustained_profile() else {},
        early_recovery_reward=early_reward_settings(),
        training_reward_wrapper_applied=early_reward_present,
        clearance_reward_wrapper_applied=clearance_reward_present,
        clearance_reward=clearance_reward_settings(),
        sustained_reward=2. if sustained_profile() else 0.,
        training_force_schedule_per_worker=list(force_schedule()),
        random_training_push_probability=training_push_probability(),
        successful_retention_weights=retention_weights(),
        selection_force_curriculum=False, **motor_options()))


class Selection(BaseCallback):
    def __init__(self, folder, assessor, reference, baseline, status, start, original_controller=None):
        super().__init__()
        self.folder = folder
        self.assessor = assessor
        self.reference = reference
        self.baseline = self.best = baseline
        self.original_controller = original_controller
        self.status = status
        self.start = start
        self.next = INTERVAL
        self.curve = []
        self.last_assessed = -1
        self.ready_streak = 0

    def assess(self):
        if self.last_assessed == self.num_timesteps:
            return
        self.last_assessed = self.num_timesteps
        checkpoint = self.folder / f'level{target_force():g}_attempt0_step{self.num_timesteps}.zip'
        protected(self.model, self.reference)
        self.model.save(checkpoint)
        self.status('selection', steps=self.num_timesteps)
        result = self.assessor.evaluate(checkpoint)
        write(checkpoint.with_suffix('.json'), result)
        accept = eligible(result, self.baseline) and score(result) > score(self.best)
        if PROFILE in BALANCED_SELECTION_PROFILES:
            accept = accept and eligible(result, self.best)
        if self.original_controller is not None:
            accept = accept and eligible(result, self.original_controller)
        if accept:
            self.best = result
            shutil.copy2(checkpoint, self.folder / 'selected_policy.zip')
            write(self.folder / 'selected_selection.json', result)
        retained_original = (self.original_controller is None
                             or eligible(result, self.original_controller))
        self.ready_streak = (self.ready_streak + 1 if eligible(result, self.baseline)
                             and retained_original and ready(result) else 0)
        self.curve.append(dict(steps=self.num_timesteps, attempt=0, arm='all_actor',
            accepted=accept, checkpoint=str(checkpoint), summary=result['summary'],
            strict_summary=result['strict_summary'], balance_curriculum_stage=STAGE,
            consecutive_ready_checks=self.ready_streak))
        write(self.folder / f'curve_{target_force():g}_0.json', self.curve)
        self.status('training', steps=self.num_timesteps, accepted=accept,
                    summary=result['summary'], strict_summary=result['strict_summary'])

    def _on_step(self):
        if (self.folder / 'STOP').exists() or time.monotonic() - self.start >= MAX_SECONDS:
            return False
        if self.num_timesteps >= self.next:
            self.next += INTERVAL
            self.assess()
        return True


def run(folder):
    lock = (ROOT / 'outputs/spring208_phase1_launch/workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'workflow.json').exists():
        raise RuntimeError('Refusing to overwrite an existing run')
    checked = json.loads((folder / 'preflight.json').read_text())
    if (not checked['passed'] or checked['source_hashes'] != source_hashes() or checked['stage'] != STAGE
            or checked.get('profile', 'balance_first18') != PROFILE
            or any(checked.get(k) != v for k,v in motor_options().items())):
        raise RuntimeError('Preflight is stale')
    start = time.monotonic()
    state = dict(pid=os.getpid(), started_unix=time.time(), profile=PROFILE,
        balance_curriculum_stage=STAGE, target_force_N=target_force(), long_term_target_N=150., level_N=target_force(),
        max_steps=STEPS, max_training_seconds=MAX_SECONDS, arm='all_actor', attempt=0,
        simulation_only=True, auto_hardware_deployment=False, **motor_options())
    def status(phase, **extra):
        state.update(state=phase, elapsed_seconds=time.monotonic() - start, **extra)
        write(folder / 'workflow.json', state)
        print(json.dumps(state), flush=True)
    write(folder / 'protocol.json', dict(profile=PROFILE, parent=str(PARENT), parent_sha256=PARENT_SHA,
        arms=['all_actor'], steps_per_arm=STEPS, num_envs=8, rollout_transitions=6144,
        stage=STAGE, stage_criteria=asdict(STAGES[STAGE]), curriculum=[asdict(s) for s in STAGES.values()],
        native_acceptance=STAGE == 'strict', strict_metrics_reported_separately=True,
        selection_requires_native_position_retention=False, training_seed=TRAIN_SEED,
        selection_seed=common.SELECT_SEED, test_seed=common.TEST_SEED,
        learning_rate=ab.LEARNING_RATE, action_std=ab.ACTION_STD, reward_scale=.01,
        rehearsal=rehearsal_settings(folder),
        reward_overrides=RECOVERY_WEIGHTS if sustained_profile() else {},
        sustained_reward=2. if sustained_profile() else 0.,
        fresh_training_discovery=sustained_profile(),
        negative_x_failure_sampling_weight=6 if PROFILE in DIRECTIONAL_PROFILES else 3 if PROFILE in ('reactive_balance18', 'recovery_curriculum18', 'repeat_recovery18', *MIXED_RECOVERY_PROFILES) else 1,
        failed_after_first_recovery_sampling_weight=6 if PROFILE in DIRECTIONAL_PROFILES else 3 if PROFILE in MIXED_RECOVERY_PROFILES else 6 if PROFILE == 'repeat_recovery18' else None,
        delayed_failure_sampling_weight=4 if PROFILE in DIRECTIONAL_PROFILES else 2 if PROFILE in MIXED_RECOVERY_PROFILES else 4 if PROFILE == 'repeat_recovery18' else None,
        hard_replay_probability=replay_probability(),
        successful_training_retention_replay=PROFILE in DIRECTIONAL_PROFILES,
        early_recovery_reward=early_reward_settings(),
        evaluation_uses_early_recovery_reward=False,
        clearance_reward=clearance_reward_settings(),
        evaluation_uses_clearance_reward=False,
        sampling_weights_combination='maximum applicable weight',
        protect_accepted15=PROFILE in ('repeat_recovery18', *MIXED_RECOVERY_PROFILES),
        protect_level_and_exact18=PROFILE in BALANCED_SELECTION_PROFILES,
        target_force_N=target_force(), protect_exact18=PROFILE in PROGRESSIVE_PROFILES,
        mixed_force_range_N=mixed_force_range(),
        training_levels_N=[f for _, f in force_schedule()],
        prior_18N_user_accepted=PROFILE == 'progressive_balance30',
        prior_18N_automatic_gate_waived_by_user=PROFILE == 'progressive_balance30',
        selection_push_cases_per_group=24 if PROFILE in BALANCED_SELECTION_PROFILES else 12,
        selection_standing_cases=64 if PROFILE in RETENTION_PROFILES else 8,
        selection_small_push_cases=24 if PROFILE in RETENTION_PROFILES else 8,
        random_training_push_probability=training_push_probability(),
        successful_retention_weights=retention_weights(),
        training_force_band_probabilities=[.1, .2, .4, .3] if PROFILE in BALANCED_SELECTION_PROFILES else None,
        controller_tuning_experiment=compare_original_controller(),
        original_controller_gain_comparison=compare_original_controller(),
        rising_wheel_gain_unchanged=True, wheel_torque_limits_Nm=[2., 2.],
        training_force_schedule_per_worker=list(force_schedule()),
        training_force_schedule_global_approx=[[s * 8, f] for s, f in force_schedule()],
        force_schedule_applies_at_episode_reset=True, selection_force_curriculum=False,
        force_scaled_replays_are_not_exact=True,
        model_and_motor_limits_unchanged=not bool(motor_options()),
        model_geometry_unchanged=True, motor_limit_overrides=motor_options(),
        **motor_options(),
        parent_baseline_uses_same_motor_limits=True,
        historical_rehearsal_torque_limit_Nm=10., historical_rehearsal_wheel_kv=.2,
        training_only_replay=True,
        promotion_requires_two_validation_checks_and_independent_test=True,
        automatic_stage_promotion=False, automatic_velocity_training=False,
        expected_minutes=[35, 55] if PROFILE in PROGRESSIVE_PROFILES else [18, 30] if PROFILE == 'short_directional18' else [25, 40] if PROFILE in ('directional_recovery18', 'standing_retention18') else [20, 35] if PROFILE in BALANCED_SELECTION_PROFILES else [20, 30],
        conservative_minutes=70 if PROFILE in PROGRESSIVE_PROFILES else 40 if PROFILE == 'short_directional18' else 50 if PROFILE in ('directional_recovery18', 'standing_retention18') else 45, source_hashes=source_hashes()))
    for name in source_hashes():
        shutil.copy2(ROOT / name, folder / Path(name).name)
    shutil.copy2(PARENT, folder / 'selected_policy.zip')
    reference = PPO.load(PARENT, device='cpu')
    assessor = Assessor()
    try:
        status('level_baseline', steps=0)
        baseline = assessor.evaluate(PARENT)
        write(folder / f'parent_level{target_force():g}.json', baseline)
        write(folder / 'selected_selection.json', baseline)
        original_controller = None
        if compare_original_controller():
            original_controller = assessor.evaluate(PARENT,
                motor_overrides=dict(standing_leg_torque_limit_Nm=20., standing_wheel_kv=.2))
            write(folder / 'original_controller_selection.json', original_controller)
        vector = SubprocVecEnv([factory(folder, TRAIN_SEED + i, STAGE,
            str(folder / f'monitor_{target_force():g}_0_{i}.csv')) for i in range(8)], start_method='spawn')
        try:
            model, settings = learner(folder, vector)
            write(folder / f'settings_{target_force():g}_0.json', dict(settings, rehearsal=rehearsal_settings(folder)))
            model.set_logger(configure(str(folder / f'logs_{target_force():g}_0'), ['csv']))
            callback = Selection(folder, assessor, reference, baseline, status, start, original_controller)
            status('training', steps=0)
            model.learn(total_timesteps=STEPS, callback=callback, reset_num_timesteps=True, progress_bar=False)
            callback.assess()
            total = model.num_timesteps
        finally:
            vector.close()
        status('final_independent_test', steps=total)
        before = assessor.evaluate(PARENT, 'test', True)
        write(folder / 'parent_test.json', before)
        identical = sha(folder / 'selected_policy.zip') == PARENT_SHA
        after = copy.deepcopy(before) if identical else assessor.evaluate(folder / 'selected_policy.zip', 'test', True)
        after.update(checkpoint=str(folder / 'selected_policy.zip'))
        write(folder / 'selected_test.json', after)
        original_controller_test = None
        if compare_original_controller():
            original_controller_test = assessor.evaluate(PARENT, 'test', True,
                motor_overrides=dict(standing_leg_torque_limit_Nm=20., standing_wheel_kv=.2))
            write(folder / 'original_controller_test.json', original_controller_test)
        physics = not any(e['warnings'] or max(e['spring_upper_mm'] + e['spring_lower_mm']) > 1.
                          for e in after['episodes'])
        independent_retained = eligible(after, before) and physics
        rejected_candidate = None
        if PROFILE == 'catch_recovery24' and not independent_retained:
            # Preserve evidence of the rejected candidate and retain the parent
            # as the advertised checkpoint. Do not search on independent tests.
            rejected_candidate = str(folder / 'rejected_candidate.zip')
            shutil.copy2(folder / 'selected_policy.zip', rejected_candidate)
            write(folder / 'rejected_candidate_test.json', after)
            shutil.copy2(PARENT, folder / 'selected_policy.zip')
            write(folder / 'selected_selection.json', baseline)
            after = copy.deepcopy(before)
            after.update(checkpoint=str(folder / 'selected_policy.zip'))
            write(folder / 'selected_test.json', after)
            identical = True
            physics = not any(e['warnings'] or max(e['spring_upper_mm'] + e['spring_lower_mm']) > 1.
                              for e in after['episodes'])
        stage_passed = ready(after) and eligible(after, before) and physics and callback.ready_streak >= 2
        if PROFILE == 'catch_recovery24':
            stage_passed = stage_passed and independent_retained
        if original_controller_test is not None:
            stage_passed = stage_passed and eligible(after, original_controller_test)
        target_passed = stage_passed if PROFILE in PROGRESSIVE_PROFILES else False
        write(folder / 'assessment.json', dict(target_force_N=target_force(), target_passed=target_passed,
            target_150N_passed=False, balance_curriculum_stage=STAGE, stage_passed=stage_passed,
            promotion_ready=stage_passed, automatic_stage_promotion=False,
            steps=total, seconds=time.monotonic() - start, parent_summary=before['summary'],
            selected_summary=after['summary'], parent_strict_summary=before['strict_summary'],
            selected_strict_summary=after['strict_summary'], physical_checks_passed=physics,
            selected_checkpoint=str(folder / 'selected_policy.zip'), selected_sha256=sha(folder / 'selected_policy.zip'),
            parent_sha256=PARENT_SHA, selected_identical_to_parent=identical,
            independent_candidate_retained=independent_retained,
            rejected_candidate_checkpoint=rejected_candidate,
            original_controller_summary=original_controller_test['summary'] if original_controller_test else None,
            original_controller_retained=eligible(after, original_controller_test) if original_controller_test else None,
            protected_actor=protected(PPO.load(folder / 'selected_policy.zip', device='cpu'), reference),
            simulation_only=True, automatic_velocity_training=False, **motor_options()))
        status('completed', steps=total, stage_passed=stage_passed, target_passed=target_passed, target_150N_passed=False)
    except BaseException as error:
        status('failed', error=repr(error))
        raise
    finally:
        assessor.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--stage', choices=list(STAGES), default='balance')
    parser.add_argument('--profile', choices=['balance_first18', 'sustained_balance18', *RATED_PROFILES], default='rated_balance18')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--wheel-kv', type=float, choices=[.3,.4,.5])
    args = parser.parse_args()
    setup(args.stage, args.profile, args.wheel_kv)
    folder = args.folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(folder)
    elif args.check:
        preflight(folder)
    else:
        run(folder)
