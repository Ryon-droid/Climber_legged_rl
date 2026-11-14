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
from push_recovery_rehearsal import ScaledReward
import train_spring208_push_ab as ab
import train_spring208_push150 as common
from train_spring208_push import CFG, evaluate_case, summarize, protected, write
from prepare_spring208_phase3 import sha

PARENT = ROOT / 'outputs/push_wheel_recovery18_20261113_2020/selected_policy.zip'
PARENT_SHA = 'cd6d6eb1674e4a0753c91ad710f8b679c368256cd9402007c1262ca29f39a144'
STEPS = 491520
INTERVAL = 98304
MAX_SECONDS = 2100
TRAIN_SEED = 1210000
STAGE = 'balance'
PROFILE = 'rated_balance18'
RECOVERY_WEIGHTS = dict(tilt_weight=8., tilt_rate_weight=2., velocity_weight=3., yaw_rate_weight=1.)


def motor_options():
    return dict(standing_leg_torque_limit_Nm=20.) if PROFILE == 'rated_balance18' else {}


def sustained_profile():
    return PROFILE in ('sustained_balance18', 'rated_balance18')


def setup(stage, profile='rated_balance18'):
    global STAGE, PROFILE, TRAIN_SEED
    STAGE = stage
    PROFILE = profile
    TRAIN_SEED = 1240000 if sustained_profile() else 1210000
    ab.configure_profile('wheel_recovery18')
    ab.PARENT = PARENT
    ab.PARENT_SHA = PARENT_SHA
    ab.ARMS = ('all_actor',)
    ab.NUM_ENVS = 8
    ab.SEED = TRAIN_SEED
    ab.LEARNING_RATE = 1e-5
    ab.ACTION_STD = ([.03] * 4 + [.20] * 2 if sustained_profile()
                     else [.015] * 4 + [.15] * 2)
    ab.setup()
    common.SELECT_SEED = 1250000 if sustained_profile() else 1220000
    common.TEST_SEED = 1260000 if sustained_profile() else 1230000


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
    specs = [dict(group='training_discovery', config=str(CFG), mode='standing',
        push=i >= 8, force_N=[18., 18.], seed=TRAIN_SEED + 1000 + i,
        balance_curriculum_stage=STAGE, **motor_options()) for i in range(48)]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        rows = list(pool.map(evaluate_case, [(str(PARENT), s, False) for s in specs]))
    cases = []
    for row in rows:
        if row['success']:
            continue
        events = row['push_metrics']['push_events']
        delayed = bool(events and all(e['recovered'] for e in events))
        cases.append(dict(seed=row['spec']['seed'], force_N=[18., 18.],
            push=row['spec']['push'], expected_push_events=events,
            source='fresh training discovery', failure_reason=row['reason'],
            delayed_failure_after_recovery=delayed))
    if not cases or any(r['warnings'] for r in rows):
        raise RuntimeError('No usable training failures or numerical warnings during discovery')
    write(folder / 'training_discovery.json', dict(training_seed_range=[TRAIN_SEED+1000, TRAIN_SEED+1047],
        holdouts_not_replayed=True, outcomes=rows, failed_cases=len(cases),
        delayed_failures=sum(c['delayed_failure_after_recovery'] for c in cases), **motor_options()))
    write(folder / 'hard_training_resets.json', dict(cases=cases, probability=.5,
        source='fresh training-only no-push and exact18 failures; no previous test seeds', **motor_options()))


def source_hashes():
    return {name: sha(ROOT / name) for name in (
        'scripts/train_spring208_push_balance.py', 'scripts/train_spring208_push.py',
        'scripts/train_spring208_push_ab.py', 'source/climber_legged_rl/push_balance_curriculum.py',
        'source/climber_legged_rl/push_reset_replay.py', 'source/climber_legged_rl/motor_authority.py')}


def factory(folder, seed, stage, monitor=None):
    cases = json.loads((folder / 'hard_training_resets.json').read_text())['cases']
    sustained = sustained_profile()
    limits = motor_options()
    def create():
        config = load_config(CFG)
        env = make_mujoco_env(ROOT / config['experiment']['output_directory'] / 'robot_experiment.xml', CFG)
        base = env.unwrapped
        apply_standing_motor_limit(env, **limits)
        base.getup['stand_probability'] = 1.
        base.push.config = dict(base.push.config, episode_probability=.75, force_N=[2., 18.],
            force_bands=[dict(force_N=[2., 10.], probability=.1),
                         dict(force_N=[10., 15.], probability=.2),
                         dict(force_N=[15., 18.], probability=.2 if sustained else .3),
                         dict(force_N=[18., 18.], probability=.5 if sustained else .4)])
        if sustained:
            base.balance_recovery.settings = dict(base.balance_recovery.settings, **RECOVERY_WEIGHTS)
        env = PushBalanceCurriculum(env, stage, sustained_reward=2. if sustained else 0.)
        env.reset(seed=seed)
        env = ExactPushResetReplay(env, cases, .5, seed + 1000003)
        return Monitor(ScaledReward(env, .01), filename=monitor,
                       info_keywords=('strict_success',))
    return create


class Assessor(common.Assessor):
    def evaluate(self, checkpoint, split='selection', physics=False):
        started = time.monotonic()
        specs = common.specifications(18., split)
        if split == 'selection':
            specs += [dict(group='exact18', config=str(common.NOMINAL), mode='standing', push=True,
                           force_N=[18., 18.], seed=common.SELECT_SEED + 200 + i) for i in range(12)]
        specs = [dict(s, balance_curriculum_stage=STAGE, **motor_options()) for s in specs]
        rows = list(self.pool.map(evaluate_case, [(str(checkpoint), s, physics) for s in specs]))
        strict = [dict(r, success=r['strict_success'], push_metrics=r['strict_push_metrics']) for r in rows]
        return dict(checkpoint=str(checkpoint), sha256=sha(checkpoint), level_N=18., split=split,
            balance_curriculum_stage=STAGE, **motor_options(), episodes=rows, summary=summarize(rows),
            strict_summary=summarize(strict), seconds=time.monotonic() - started,
            physics_step_spring_measurement=physics)


def eligible(result, baseline):
    # Protect standing/getup under the current curriculum. Strict position loss
    # is recorded, but deliberately does not veto stage-1 balance improvements.
    return (not any(e['warnings'] for e in result['episodes'])
            and all(result['summary'][g]['success'] >= baseline['summary'][g]['success']
                    for g in ('standing', 'nominal_getup', 'difficult_getup')))


def score(result):
    s = result['summary']
    pushed = [e for e in result['episodes'] if e['spec']['group'] in ('exact18', 'level')]
    return [s['exact18']['success'], s['level']['success'], s['accepted15']['success'],
            sum(e['push_metrics']['recovered_push_count'] for e in pushed),
            sum(e['seconds'] for e in pushed),
            -sum(e['final_errors']['speed_m_s'] for e in pushed),
            -sum(e['final_errors']['yaw_rate_rad_s'] for e in pushed)]


def ready(result):
    s = result['summary']
    return (not any(e['warnings'] for e in result['episodes'])
            and all(s[g]['success'] / s[g]['count'] >= .9 for g in ('exact18', 'level', 'accepted15'))
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
    spec = dict(group='exact18', config=str(common.NOMINAL), mode='standing', push=True,
                force_N=[18., 18.], seed=630000, **motor_options())
    native = evaluate_case((str(PARENT), spec, True))
    stage = evaluate_case((str(PARENT), dict(spec, balance_curriculum_stage=STAGE), True))
    for key in ('seconds', 'maximum', 'final_errors', 'spring_upper_mm', 'spring_lower_mm', 'warnings'):
        if native[key] != stage[key]:
            raise RuntimeError('Wrapper changed physical rollout: ' + key)
    if native['success'] != stage['strict_success'] or native['push_metrics'] != stage['strict_push_metrics']:
        raise RuntimeError('Native strict accounting changed')
    forces = [np.linalg.norm(e['force_world_N']) for e in stage['push_metrics']['push_events']]
    if not forces or not np.allclose(forces, 18.) or stage['warnings']:
        raise RuntimeError('18 N physical smoke check failed')
    training_env = factory(folder, TRAIN_SEED, STAGE)()
    try:
        training_env.reset(seed=TRAIN_SEED, options={'reset_mode': 'standing'})
        wrapper = training_env
        while not isinstance(wrapper, PushBalanceCurriculum):
            wrapper = wrapper.env
        expected_limit = motor_options().get('standing_leg_torque_limit_Nm', 10.)
        np.testing.assert_array_equal(training_env.unwrapped.torque_limits, [expected_limit] * 4 + [2.] * 2)
        if sustained_profile():
            assert wrapper.sustained_reward == 2.
            assert all(training_env.unwrapped.balance_recovery.settings[k] == v for k,v in RECOVERY_WEIGHTS.items())
    finally:
        training_env.close()
    write(folder / 'preflight.json', dict(passed=True, source_hashes=source_hashes(), stage=STAGE, profile=PROFILE,
        parent_sha256=PARENT_SHA, discarded_steps=1024, num_envs=8,
        all_six_output_rows_updated=rows, contract=contract,
        physical_rollout_identical=True, native_strict_accounting_identical=True,
        force_N=forces, settings_learning_rate=settings['learning_rate'],
        action_std=ab.ACTION_STD, rehearsal_updates_per_4096=rehearsal_settings(folder)['updates_per_rollout'],
        reward_overrides=RECOVERY_WEIGHTS if sustained_profile() else {},
        sustained_reward=2. if sustained_profile() else 0., **motor_options()))


class Selection(BaseCallback):
    def __init__(self, folder, assessor, reference, baseline, status, start):
        super().__init__()
        self.folder = folder
        self.assessor = assessor
        self.reference = reference
        self.baseline = self.best = baseline
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
        checkpoint = self.folder / f'level18_attempt0_step{self.num_timesteps}.zip'
        protected(self.model, self.reference)
        self.model.save(checkpoint)
        self.status('selection', steps=self.num_timesteps)
        result = self.assessor.evaluate(checkpoint)
        write(checkpoint.with_suffix('.json'), result)
        accept = eligible(result, self.baseline) and score(result) > score(self.best)
        if accept:
            self.best = result
            shutil.copy2(checkpoint, self.folder / 'selected_policy.zip')
            write(self.folder / 'selected_selection.json', result)
        self.ready_streak = self.ready_streak + 1 if eligible(result, self.baseline) and ready(result) else 0
        self.curve.append(dict(steps=self.num_timesteps, attempt=0, arm='all_actor',
            accepted=accept, checkpoint=str(checkpoint), summary=result['summary'],
            strict_summary=result['strict_summary'], balance_curriculum_stage=STAGE,
            consecutive_ready_checks=self.ready_streak))
        write(self.folder / 'curve_18_0.json', self.curve)
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
            or checked.get('profile', 'balance_first18') != PROFILE):
        raise RuntimeError('Preflight is stale')
    start = time.monotonic()
    state = dict(pid=os.getpid(), started_unix=time.time(), profile=PROFILE,
        balance_curriculum_stage=STAGE, target_force_N=18., long_term_target_N=150., level_N=18.,
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
        model_and_motor_limits_unchanged=not bool(motor_options()),
        model_geometry_unchanged=True, motor_limit_overrides=motor_options(),
        **motor_options(),
        parent_baseline_uses_same_motor_limits=True,
        historical_rehearsal_torque_limit_Nm=10., training_only_replay=True,
        promotion_requires_two_validation_checks_and_independent_test=True,
        automatic_stage_promotion=False, automatic_velocity_training=False,
        expected_minutes=[20, 30], conservative_minutes=45, source_hashes=source_hashes()))
    for name in source_hashes():
        shutil.copy2(ROOT / name, folder / Path(name).name)
    shutil.copy2(PARENT, folder / 'selected_policy.zip')
    reference = PPO.load(PARENT, device='cpu')
    assessor = Assessor()
    try:
        status('level_baseline', steps=0)
        baseline = assessor.evaluate(PARENT)
        write(folder / 'parent_level18.json', baseline)
        write(folder / 'selected_selection.json', baseline)
        vector = SubprocVecEnv([factory(folder, TRAIN_SEED + i, STAGE,
            str(folder / f'monitor_18_0_{i}.csv')) for i in range(8)], start_method='spawn')
        try:
            model, settings = learner(folder, vector)
            write(folder / 'settings_18_0.json', dict(settings, rehearsal=rehearsal_settings(folder)))
            model.set_logger(configure(str(folder / 'logs_18_0'), ['csv']))
            callback = Selection(folder, assessor, reference, baseline, status, start)
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
        physics = not any(e['warnings'] or max(e['spring_upper_mm'] + e['spring_lower_mm']) > 1.
                          for e in after['episodes'])
        stage_passed = ready(after) and eligible(after, before) and physics and callback.ready_streak >= 2
        write(folder / 'assessment.json', dict(target_force_N=18., target_passed=False,
            target_150N_passed=False, balance_curriculum_stage=STAGE, stage_passed=stage_passed,
            promotion_ready=stage_passed, automatic_stage_promotion=False,
            steps=total, seconds=time.monotonic() - start, parent_summary=before['summary'],
            selected_summary=after['summary'], parent_strict_summary=before['strict_summary'],
            selected_strict_summary=after['strict_summary'], physical_checks_passed=physics,
            selected_checkpoint=str(folder / 'selected_policy.zip'), selected_sha256=sha(folder / 'selected_policy.zip'),
            parent_sha256=PARENT_SHA, selected_identical_to_parent=identical,
            protected_actor=protected(PPO.load(folder / 'selected_policy.zip', device='cpu'), reference),
            simulation_only=True, automatic_velocity_training=False, **motor_options()))
        status('completed', steps=total, stage_passed=stage_passed, target_passed=False, target_150N_passed=False)
    except BaseException as error:
        status('failed', error=repr(error))
        raise
    finally:
        assessor.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--stage', choices=list(STAGES), default='balance')
    parser.add_argument('--profile', choices=['balance_first18', 'sustained_balance18', 'rated_balance18'], default='rated_balance18')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    setup(args.stage, args.profile)
    folder = args.folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(folder)
    elif args.check:
        preflight(folder)
    else:
        run(folder)
