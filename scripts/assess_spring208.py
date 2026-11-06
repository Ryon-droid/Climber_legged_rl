"""Paired old/new-stop evaluation of the unchanged round-14 actor; no training."""
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
import numpy as np
import torch
from stable_baselines3 import PPO
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_training import validate_policy_transfer
from climber_legged_rl.spring_geometry import eye_center_distance

PARENT = ROOT/'outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip'
CONFIGS = {'old220': ROOT/'configs/robot/cad_balance_estimator_round15.yaml',
           'new208': ROOT/'configs/robot/cad_balance_spring208.yaml'}


def episode(job):
    label, mode, push, seed = job
    torch.set_num_threads(1)
    config_path = CONFIGS[label]
    config = load_config(config_path)
    env = make_mujoco_env(ROOT/config['experiment']['output_directory']/'robot_experiment.xml', config_path)
    policy = PPO.load(PARENT, device='cpu')
    env.push.config = dict(env.push.config, episode_probability=float(push))
    try:
        obs, reset = env.reset(seed=seed, options={'reset_mode': mode})
        low = np.full(2, np.inf)
        high = np.full(2, -np.inf)
        max_upper_penetration = np.zeros(2)
        max_lower_penetration = np.zeros(2)
        max_closure = max_tilt = max_heading = max_displacement = 0.
        warning_start = np.array(env.data.warning.number).copy()

        def record_spring():
            # Called once per 1 ms physics step via the existing torque observer,
            # plus reset and final states; this includes inter-policy-step peaks.
            lengths = np.array([eye_center_distance(env.model, env.data, s) for s in ('left', 'right')])
            np.minimum(low, lengths, out=low)
            np.maximum(high, lengths, out=high)
            bounds = np.array([config['spring']['eye_center_limits_m'][s] for s in ('left', 'right')])
            np.maximum(max_upper_penetration, lengths-bounds[:, 1], out=max_upper_penetration)
            np.maximum(max_lower_penetration, bounds[:, 0]-lengths, out=max_lower_penetration)

        original_observer = env.unwrapped._observe_motor_torque
        physics_samples = 0

        def observe(torque):
            nonlocal physics_samples
            original_observer(torque)
            record_spring()
            physics_samples += 1

        env.unwrapped._observe_motor_torque = observe
        record_spring()
        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            record_spring()
            max_closure = max(max_closure, info['closure_error_m'])
            max_tilt = max(max_tilt, abs(info['tilt_rad']))
            max_heading = max(max_heading, abs(info['heading_error_rad']))
            max_displacement = max(max_displacement, info['planar_displacement_m'])
            if done or truncated:
                if physics_samples != env.steps*config['control']['decimation']:
                    raise RuntimeError('Spring measurement hook did not cover every physics step')
                return dict(model=label, mode=mode, push=bool(push), seed=seed,
                    reset=reset, success=bool(info['combined_success']), acquired=bool(info['getup_acquired']),
                    getup_seconds=info['getup_seconds'], seconds=env.steps*env.dt,
                    reason=info['termination_reason'] or 'timeout',
                    spring_min_m=low.tolist(), spring_max_m=high.tolist(),
                    upper_stop_penetration_mm=(max_upper_penetration*1000).tolist(),
                    lower_stop_penetration_mm=(max_lower_penetration*1000).tolist(),
                    physics_samples=physics_samples,
                    max_closure_m=max_closure, max_tilt_deg=float(np.degrees(max_tilt)),
                    max_heading_deg=float(np.degrees(max_heading)), max_displacement_m=max_displacement,
                    warnings=(np.array(env.data.warning.number)-warning_start).tolist(),
                    peak_torque_Nm=info['peak_torque_Nm'], over_rated_seconds=info['over_rated_seconds'])
    finally:
        env.close()


def main():
    torch.set_num_threads(1)
    config = load_config(CONFIGS['new208'])
    folder = ROOT/config['experiment']['output_directory']
    validate_artifacts(CONFIGS['new208'])
    validate_policy_transfer(PARENT, folder/'robot_experiment.xml', config, warm_start=True)
    cases = [(mode, push, seed) for mode in ('standing', 'folded')
             for push in (False, True) for seed in range(315000, 315008)]
    cases += [('standing', False, 261409), ('standing', False, 262219),
              ('standing', True, 261106), ('standing', True, 261704)]
    jobs = [(label, *case) for label in CONFIGS for case in cases]
    started = time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(episode, job): job for job in jobs}
        for future in as_completed(pending):
            result = future.result()
            records.append(result)
            if len(records) % 8 == 0:
                print(json.dumps(dict(completed=len(records), total=len(jobs),
                                     elapsed_seconds=time.monotonic()-started)), flush=True)
    records.sort(key=lambda r: (r['model'], r['mode'], r['push'], r['seed']))
    groups = {}
    for label in CONFIGS:
        for mode in ('standing', 'folded'):
            for push in (False, True):
                rows = [r for r in records if (r['model'], r['mode'], r['push']) == (label, mode, push)
                        and r['seed'] >= 315000]
                key = f'{label}_{mode}_push{int(push)}'
                groups[key] = dict(count=len(rows), successes=sum(r['success'] for r in rows),
                    acquired=sum(r['acquired'] for r in rows),
                    max_spring_mm=1000*max(max(r['spring_max_m']) for r in rows),
                    max_upper_stop_penetration_mm=max(max(r['upper_stop_penetration_mm']) for r in rows),
                    max_lower_stop_penetration_mm=max(max(r['lower_stop_penetration_mm']) for r in rows),
                    max_closure_m=max(r['max_closure_m'] for r in rows),
                    warning_count=sum(sum(r['warnings']) for r in rows),
                    failures=[dict(seed=r['seed'], reason=r['reason']) for r in rows if not r['success']])
    report = dict(simulation_only=True, training_started=False, checkpoint=str(PARENT),
        checkpoint_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started, configs={k:str(v) for k,v in CONFIGS.items()},
        model_sha256={label:hashlib.sha256((ROOT/load_config(path)['experiment']['output_directory']/
                      'robot_experiment.xml').read_bytes()).hexdigest() for label,path in CONFIGS.items()},
        groups=groups, hard_cases=[r for r in records if r['seed'] < 315000], episodes=records,
        note='Paired small-sample diagnostics. Hard seeds are separate from fresh-seed success counts. '
             'MuJoCo stop penetration is reported, not silently interpreted as a perfectly rigid hardware stop.')
    (folder/'spring208_assessment.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(groups=groups, elapsed_seconds=report['elapsed_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    main()
