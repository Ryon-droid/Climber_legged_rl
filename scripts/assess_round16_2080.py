"""Paired regression and fresh-seed assessment; never auto-promote a checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import torch
from climber_legged_rl.evaluation_runtime import ParallelEvaluation


def summarize(result):
    episodes = result['episodes']
    return dict(episodes=len(episodes), successes=sum(bool(e['combined_success']) for e in episodes),
        max_heading_deg=max(e['max_heading_error_deg'] for e in episodes),
        max_displacement_mm=1000*max(e['max_planar_displacement_m'] for e in episodes),
        warning_count=sum(sum(e['simulator_warning_counts']) for e in episodes),
        failures=[dict(seed=result['seed']+i, reason=e['reason'], seconds=e['seconds'],
                       heading_deg=e['max_heading_error_deg'], displacement_mm=1000*e['max_planar_displacement_m'])
                  for i, e in enumerate(episodes) if not e['combined_success']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stamp', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = ROOT/f'outputs/round16_assessment_{args.stamp}'
    output.mkdir(exist_ok=False)
    baseline = ROOT/'outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip'
    models = dict(baseline=baseline)
    for trial in ('a', 'b'):
        run = ROOT/f'outputs/mujoco_runs/cad_balance_estimator_round16{trial}_2080_{args.stamp}'
        models[trial] = run/'final_policy.zip'
    # Evaluate every model under the identical baseline task and success criteria.
    config = ROOT/'configs/robot/cad_balance_estimator_round15.yaml'
    xml = ROOT/'outputs/cad_balance_estimator_round15/robot_experiment.xml'
    jobs = {}
    for label, cp in models.items():
        for split, seed in [('regression', 275000), ('fresh', 285000)]:
            for mode in ('standing', 'folded'):
                for push in (False, True):
                    key = f'{label}_{split}_{mode}' + ('_push' if push else '')
                    jobs[key] = dict(checkpoint=cp, config_path=config, model_path=xml,
                        episodes=20, seed=seed, reset_mode=mode, push_episode_probability=float(push))
        for seed, push in ((261409, False), (262219, False), (261106, True), (261704, True)):
            jobs[f'{label}_hard_{seed}'] = dict(checkpoint=cp, config_path=config, model_path=xml,
                episodes=1, seed=seed, reset_mode='standing', push_episode_probability=float(push))
    started = time.monotonic()
    evaluator = ParallelEvaluation(4)
    try:
        results = evaluator.run(jobs)
    finally:
        evaluator.close()
    summary = {}
    for key, result in results.items():
        (output/(key+'.json')).write_text(json.dumps(result, indent=2)+'\n')
        summary[key] = summarize(result)
    report = dict(simulation_only=True, seconds=time.monotonic()-started,
        checkpoints={k:str(v) for k,v in models.items()}, summaries=summary,
        auto_promotion=False, note='Compare complete episodes and failure severity before continuing training.')
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
