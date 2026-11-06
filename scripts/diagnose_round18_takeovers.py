"""Preserve state/history while switching baseline to recovery actor at fixed times."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys
import json
import hashlib
import time
ROOT = Path('/home/climber/Climber_legged_rl')
sys.path.insert(0, str(ROOT/'source'))
import numpy as np
import torch
from stable_baselines3 import PPO
from climber_legged_rl.mujoco_env import make_mujoco_env

BASELINE = ROOT/'outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip'
TEACHER = ROOT/'outputs/mujoco_runs/cad_balance_estimator_round10_20261101_204316/checkpoints/climber_120000_steps.zip'
CONFIG = ROOT/'configs/robot/cad_balance_estimator_round15.yaml'
XML = ROOT/'outputs/cad_balance_estimator_round15/robot_experiment.xml'
FOLDER = ROOT/'outputs/round18_takeovers_20261105'


def initialize():
    global policies
    torch.set_num_threads(1)
    policies = [PPO.load(path, device='cpu') for path in (BASELINE, TEACHER)]


def rollout(spec):
    seed, pushes, takeover = spec
    env = make_mujoco_env(XML, CONFIG)
    env.unwrapped.push.config = dict(env.unwrapped.push.config, episode_probability=float(pushes))
    rows, actions = [], []
    trace = []
    try:
        obs, _ = env.reset(seed=seed, options={'reset_mode':'standing'})
        while True:
            raw = env.unwrapped
            switched = raw.steps*raw.dt+1e-9 >= takeover
            action = policies[int(switched)].predict(obs, deterministic=True)[0]
            if switched and obs[179] > .95:
                rows.append(obs[:180].copy()); actions.append(np.clip(action,-1,1))
            obs, _, done, truncated, info = env.step(action)
            if raw.steps % 50 == 0 or done or truncated:
                trace.append(dict(seconds=raw.steps*raw.dt, heading_deg=float(np.degrees(info['heading_error_rad'])),
                    displacement_mm=1000*info['planar_displacement_m'],tilt_deg=float(np.degrees(info['tilt_rad']))))
            if done or truncated:
                break
        result = dict(seed=seed, pushes=pushes, takeover_seconds=takeover,
            combined_success=bool(info['combined_success']), seconds=raw.steps*raw.dt,
            reason=info['episode_metrics']['reason'], push_metrics=info['push_metrics'], trace=trace,
            retained_samples=len(rows) if info['combined_success'] else 0)
        if info['combined_success'] and takeover > 0 and rows:
            path = FOLDER/f'accepted_{seed}_{takeover:g}.npz'
            np.savez_compressed(path, observations=np.asarray(rows,dtype=np.float32),
                                actions=np.asarray(actions,dtype=np.float32))
            result['dataset'] = str(path)
            result['dataset_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result
    finally:
        env.close()


def main():
    FOLDER.mkdir(exist_ok=False)
    specs = [(seed,push,t) for seed,push in ((261409,False),(261106,True),(261704,True),
             (262219,False),(275005,False),(295004,False),(275013,True),(295009,True))
             for t in (0.,.1,.5,1.,2.,3.,12.5,13.5)]
    started = time.monotonic()
    results=[]
    with ProcessPoolExecutor(4,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for result in pool.map(rollout,specs):
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('trace','push_metrics')}),flush=True)
    report=dict(simulation_only=True, schema='validated_policy_takeovers_v1',
        full_episode_success_required=True, state_reset_at_takeover=False,history_reset_at_takeover=False,
        physics_assistance=False, runtime_controller='baseline_then_recovery_teacher',
        config=str(CONFIG), config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        model=str(XML), model_sha256=hashlib.sha256(XML.read_bytes()).hexdigest(),
        checkpoints={role:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                     for role,path in [('baseline',BASELINE),('teacher',TEACHER)]},
        seconds=time.monotonic()-started, outcomes=results)
    (FOLDER/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TAKEOVER_ASSESSMENT_COMPLETE',flush=True)


if __name__=='__main__':
    main()
