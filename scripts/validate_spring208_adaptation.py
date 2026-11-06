"""Independent paired assessment of a new-stop student; never use teacher actions."""
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
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'scripts')]
import numpy as np
import torch
from stable_baselines3 import PPO
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_training import validate_policy_transfer
from climber_legged_rl.mujoco_model import load_config

CFG = ROOT/'configs/robot/cad_balance_spring208.yaml'
XML = ROOT/'outputs/cad_balance_spring208/robot_experiment.xml'
OUT = ROOT/'outputs/spring208_adaptation'
PARENT = ROOT/'outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_case(job):
    import assess_spring208
    label, cp, mode, push, seed = job
    assess_spring208.PARENT = Path(cp)
    assess_spring208.CONFIGS = {label: CFG}
    return assess_spring208.episode((label, mode, push, seed))


def main():
    torch.set_num_threads(1)
    adaptation = json.loads((OUT/'distillation_latest.json').read_text())
    candidate = Path(adaptation['checkpoint'])
    if adaptation['checkpoint_sha256'] != digest(candidate) or adaptation['model_sha256'] != digest(XML):
        raise RuntimeError('Candidate or model changed')
    validate_artifacts(CFG)
    for cp in (PARENT, candidate):
        validate_policy_transfer(cp, XML, load_config(CFG), warm_start=True)
    parent = PPO.load(PARENT, device='cpu')
    student = PPO.load(candidate, device='cpu')
    protected = []
    for name, tensor in parent.policy.state_dict().items():
        if not name.startswith(('mlp_extractor.policy_net.rising_net.', 'mlp_extractor.policy_net.rising_head.')):
            if not torch.equal(tensor, student.policy.state_dict()[name]):
                raise RuntimeError('Protected tensor changed: '+name)
            protected.append(name)
    cps = {'baseline': PARENT, 'candidate': candidate}
    fresh_cases = [(mode, push, seed) for mode in ('standing', 'folded')
                   for push in (False, True) for seed in range(355000, 355020)]
    hard = [('standing', False, 261409), ('standing', False, 262219),
            ('standing', True, 261106), ('standing', True, 261704)]
    jobs = [(label, str(cp), *case) for label, cp in cps.items() for case in fresh_cases+hard]
    started = time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(evaluate_case, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 16 == 0:
                print(json.dumps(dict(completed=len(records), total=len(jobs), seconds=time.monotonic()-started)), flush=True)
    groups = {}
    for label in cps:
        for mode in ('standing', 'folded'):
            for push in (False, True):
                rows = [r for r in records if (r['model'], r['mode'], r['push']) == (label, mode, push)
                        and r['seed'] >= 355000]
                groups[f'{label}_{mode}_push{int(push)}'] = dict(count=len(rows),
                    successes=sum(r['success'] for r in rows), acquired=sum(r['acquired'] for r in rows),
                    max_upper_stop_penetration_mm=max(max(r['upper_stop_penetration_mm']) for r in rows),
                    max_lower_stop_penetration_mm=max(max(r['lower_stop_penetration_mm']) for r in rows),
                    max_closure_m=max(r['max_closure_m'] for r in rows),
                    max_heading_deg=max(r['max_heading_deg'] for r in rows),
                    max_displacement_m=max(r['max_displacement_m'] for r in rows),
                    getup_seconds=sorted({r['getup_seconds'] for r in rows if r['getup_seconds'] is not None}),
                    warning_count=sum(sum(r['warnings']) for r in rows),
                    failures=[dict(seed=r['seed'], reason=r['reason']) for r in rows if not r['success']])
    by_key = {(r['model'], r['mode'], r['push'], r['seed']): r for r in records}
    standing_regressions = []
    for mode, push, seed in fresh_cases+hard:
        if mode != 'standing':
            continue
        old = by_key[('baseline', mode, push, seed)]
        new = by_key[('candidate', mode, push, seed)]
        if (old['success'] != new['success'] or old['reason'] != new['reason']
                or old['seconds'] != new['seconds']
                or abs(old['max_displacement_m']-new['max_displacement_m']) > 1e-12
                or abs(old['max_heading_deg']-new['max_heading_deg']) > 1e-12):
            standing_regressions.append(dict(seed=seed, push=push))
    passed = (groups['candidate_folded_push0']['successes'] == 20 and not standing_regressions
              and all(not sum(r['warnings']) for r in records if r['model'] == 'candidate'))
    report = dict(simulation_only=True, passed_nominal_adaptation=passed,
        independent_seed_range=[355000, 355019], runtime_teacher_assistance=False,
        model_sha256=digest(XML), candidate=str(candidate), candidate_sha256=digest(candidate),
        parent=str(PARENT), parent_sha256=digest(PARENT),
        protected_tensor_count=len(protected), protected_tensors_identical=True,
        standing_regressions=standing_regressions, groups=groups,
        hard_cases=[r for r in records if r['seed'] < 355000], episodes=records,
        seconds=time.monotonic()-started,
        note='Complete 20 s task criteria unchanged. Folded reset is one deterministic nominal pose, '
             'so seed count is not a count of independent folded geometries. Existing hard standing '
             'failures are retained and reported; no claim of real-hardware readiness.')
    (OUT/'independent_assessment.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('episodes', 'hard_cases')}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
