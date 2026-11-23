"""Paired controller diagnosis on training discovery plus fresh diagnostic cases.

No optimizer, saved policy changes, or future final-test claims. Both controllers
use the same checkpoint, reset seeds and applied force laws.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'scripts')]
from train_spring208_push import evaluate_case, summarize, write
from prepare_spring208_phase3 import sha
from climber_legged_rl.mujoco_model import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite diagnostic')
    start = time.monotonic()
    source = args.source.resolve()
    checkpoint = source/'selected_policy.zip'
    protocol = json.loads((source/'protocol.json').read_text())
    if sha(checkpoint) != protocol['parent_sha256']:
        raise RuntimeError('This comparison reuses discovery from the unchanged parent only')
    original = json.loads((source/'training_discovery.json').read_text())['outcomes']
    diagnostic = ROOT/'configs/robot/cad_spring208_phase3_nominal.yaml'
    cfg = ROOT/'configs/robot/cad_spring208_phase3_train.yaml'
    fresh = [dict(group='standing', config=str(diagnostic), mode='standing', push=False,
                  seed=1420000+i) for i in range(16)]
    for group, force, offset in [('exact18', [18.,18.], 100), ('level', [15.,18.], 200),
                                 ('accepted15', [10.,15.], 300)]:
        fresh += [dict(group=group, config=str(diagnostic), mode='standing', push=True,
                       force_N=force, seed=1420000+offset+i) for i in range(12)]
    fresh += [dict(group='nominal_getup', config=str(diagnostic), mode='folded', push=False,
                   seed=1420400+i) for i in range(4)]
    offsets = load_config(cfg)['getup']['rearward_offsets_m']
    for offset in [offsets[0], offsets[3], offsets[-1]]:
        fresh += [dict(group='difficult_getup', config=str(cfg), mode='folded', push=False,
                       offset_m=offset, seed=1420500+i) for i in range(4)]
    candidate_specs = [dict(row['spec'], standing_wheel_kv=.4) for row in original]
    fresh_specs = [dict(spec, balance_curriculum_stage='balance',
                        standing_leg_torque_limit_Nm=20., standing_wheel_kv=gain)
                   for gain in (.2,.4) for spec in fresh]
    jobs = [(str(checkpoint), spec, True) for spec in candidate_specs+fresh_specs]
    print(json.dumps(dict(phase='running', episodes=len(jobs), discovery_reused=len(original))), flush=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        rows = list(pool.map(evaluate_case, jobs))
    discovery = rows[:len(original)]
    paired_fresh = rows[len(original):]
    baseline = paired_fresh[:len(fresh)]
    candidate = paired_fresh[len(fresh):]
    def changes(before, after):
        return dict(wins=sum(not a['success'] and b['success'] for a,b in zip(before,after)),
                    regressions=sum(a['success'] and not b['success'] for a,b in zip(before,after)))
    report = dict(diagnostic_only=True, optimizer_steps=0, source=str(source),
        checkpoint_sha256=sha(checkpoint), script_sha256=sha(Path(__file__)),
        gains=[.2,.4], fresh_diagnostic_seed=1420000,
        fresh_cases_are_not_final_test=True, seconds=time.monotonic()-start,
        discovery_baseline_summary=summarize(original), discovery_candidate_summary=summarize(discovery),
        fresh_baseline_summary=summarize(baseline), fresh_candidate_summary=summarize(candidate),
        discovery_changes=changes(original,discovery), fresh_changes=changes(baseline,candidate),
        warnings=sum(r['warnings'] for r in rows),
        discovery_candidate_episodes=discovery, fresh_baseline_episodes=baseline,
        fresh_candidate_episodes=candidate)
    write(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_episodes')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
