"""Paired gain screening on fresh diagnostic seeds, separate from holdouts."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'scripts')]
from train_spring208_push import evaluate_case, summarize, write
from prepare_spring208_phase3 import sha

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Existing diagnostic will not be overwritten')
    started = time.monotonic()
    cfg = ROOT/'configs/robot/cad_spring208_phase3_nominal.yaml'
    specs = []
    for group, count, force, offset in (
            ('standing', 16, None, 0), ('exact24', 24, [24., 24.], 100),
            ('mixed24', 24, [22., 24.], 200), ('small15', 12, [10., 15.], 300),
            ('exact18', 12, [18., 18.], 400), ('getup', 4, None, 500)):
        specs += [dict(group=group, config=str(cfg), mode='folded' if group=='getup' else 'standing',
                       push=force is not None, force_N=force, seed=1610000+offset+i,
                       balance_curriculum_stage='balance', standing_leg_torque_limit_Nm=20.)
                  for i in range(count)]
    jobs = [(str(args.checkpoint), dict(spec, standing_wheel_kv=gain), False)
            for gain in (.3, .4, .5) for spec in specs]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        rows = list(pool.map(evaluate_case, jobs))
    arms = {str(gain): rows[i*len(specs):(i+1)*len(specs)] for i,gain in enumerate((.3,.4,.5))}
    summaries = {gain: summarize(episodes) for gain, episodes in arms.items()}
    baseline = summaries['0.4']
    eligible = [gain for gain, summary in summaries.items()
                if all(summary[g]['success'] >= baseline[g]['success'] for g in baseline)
                and not any(r['warnings'] for r in arms[gain])]
    score = lambda gain: sum(summaries[gain][g]['success'] for g in ('exact24','mixed24'))
    chosen = max(eligible, key=lambda gain:(score(gain), gain=='0.4')) if eligible else '0.4'
    report = dict(checkpoint_sha256=sha(args.checkpoint), diagnostic_only=True,
                  diagnostic_seed_base=1610000, holdout_seeds_used=False,
                  summaries=summaries, chosen_wheel_kv=float(chosen),
                  warnings=sum(r['warnings'] for r in rows), seconds=time.monotonic()-started,
                  episodes=arms, script_sha256=sha(Path(__file__)))
    write(args.output, report)
    print({k:v for k,v in report.items() if k!='episodes'}, flush=True)

if __name__=='__main__':
    main()
