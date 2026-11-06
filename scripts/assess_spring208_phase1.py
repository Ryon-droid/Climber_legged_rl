"""Held-out geometry and nominal/standing regression after the bounded PPO pilot."""
import argparse
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
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_training import validate_policy_transfer

PARENT = ROOT/'outputs/mujoco_runs/spring208_rising_distill_20261106_031028/final_policy.zip'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case(job):
    import assess_spring208
    from climber_legged_rl.mujoco_env import make_mujoco_env
    label, checkpoint, split, offset, mode, push, seed = job
    config = ROOT/f'configs/robot/cad_spring208_phase1_{split}.yaml'
    assess_spring208.PARENT = Path(checkpoint)
    assess_spring208.CONFIGS = {label: config}

    def factory(*args, **kwargs):
        env = make_mujoco_env(*args, **kwargs)
        if offset is not None:
            env.pose_bank = [p for p in env.pose_bank if abs(p['rearward_offset_m']-offset) < 1e-10]
            if not env.pose_bank:
                raise RuntimeError('Requested assessment pose missing')
        return env

    assess_spring208.make_mujoco_env = factory
    result = assess_spring208.episode((label, mode, push, seed))
    result.update(split=split, offset_m=offset)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = args.run_dir.resolve()
    if not (folder/'final_policy.zip').is_file():
        raise RuntimeError('Training has not saved a final checkpoint')
    # Choose with the in-training selection set only, never the held-out poses.
    selected = folder/'best/best_model.zip'
    if not selected.exists():
        raise RuntimeError('Missing in-training selection checkpoint')
    cfg = ROOT/'configs/robot/cad_spring208_phase1_train.yaml'
    config = load_config(cfg)
    xml = ROOT/config['experiment']['output_directory']/'robot_experiment.xml'
    for split in ('train', 'test', 'nominal'):
        validate_artifacts(ROOT/f'configs/robot/cad_spring208_phase1_{split}.yaml')
    validate_policy_transfer(selected, xml, config, warm_start=True)
    parent = PPO.load(PARENT, device='cpu')
    final = PPO.load(folder/'final_policy.zip', device='cpu')
    candidate = PPO.load(selected, device='cpu')
    changes = []
    for name, tensor in parent.policy.state_dict().items():
        if name.startswith(('mlp_extractor.policy_net.standing_', 'mlp_extractor.policy_net.velocity_estimator.')):
            for model in (candidate, final):
                if not torch.equal(tensor, model.policy.state_dict()[name]):
                    raise RuntimeError('Protected actor tensor changed: '+name)
        if name.startswith(('mlp_extractor.policy_net.rising_net.', 'mlp_extractor.policy_net.rising_head.')):
            if not torch.equal(tensor, final.policy.state_dict()[name]):
                changes.append(name)
    if not changes:
        raise RuntimeError('No rising weights updated in the PPO final checkpoint')
    if not torch.equal(candidate.policy.log_std, final.policy.log_std):
        raise RuntimeError('Exploration std changed after initial reset')
    specs = []
    heldout = load_config(ROOT/'configs/robot/cad_spring208_phase1_test.yaml')['getup']['rearward_offsets_m']
    for offset in heldout:
        specs += [('test', offset, 'folded', False, seed) for seed in range(385000, 385008)]
    specs += [('nominal', None, 'folded', False, seed) for seed in range(386000, 386008)]
    specs += [('nominal', None, 'standing', push, seed)
              for push in (False, True) for seed in range(387000, 387008)]
    specs += [('nominal', None, 'standing', False, 261409), ('nominal', None, 'standing', False, 262219),
              ('nominal', None, 'standing', True, 261106), ('nominal', None, 'standing', True, 261704)]
    jobs = [(label, str(cp), *spec) for label, cp in [('parent', PARENT), ('selected', selected)] for spec in specs]
    rows = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(case, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 12 == 0:
                print(json.dumps(dict(assessment_completed=len(rows), total=len(jobs), seconds=time.monotonic()-start)), flush=True)
    summary = {}
    for label in ('parent', 'selected'):
        groups = {'heldout_folded': [r for r in rows if r['model'] == label and r['split'] == 'test'],
                  'nominal_folded': [r for r in rows if r['model'] == label and r['split'] == 'nominal' and r['mode'] == 'folded'],
                  'ordinary_standing': [r for r in rows if r['model'] == label and r['mode'] == 'standing' and r['seed'] >= 387000],
                  'hard_standing': [r for r in rows if r['model'] == label and r['mode'] == 'standing' and r['seed'] < 387000]}
        for group, items in groups.items():
            summary[label+'_'+group] = dict(count=len(items), success=sum(r['success'] for r in items),
                acquired=sum(r['acquired'] for r in items), warnings=sum(sum(r['warnings']) for r in items),
                max_upper_penetration_mm=max(max(r['upper_stop_penetration_mm']) for r in items),
                max_closure_m=max(r['max_closure_m'] for r in items),
                failures=[dict(seed=r['seed'], offset=r['offset_m'], reason=r['reason']) for r in items if not r['success']])
    lookup = {(r['model'], r['split'], r['offset_m'], r['mode'], r['push'], r['seed']): r for r in rows}
    regressions = []
    for spec in specs:
        if spec[2] != 'standing':
            continue
        a, b = lookup[('parent', *spec)], lookup[('selected', *spec)]
        if any(a[k] != b[k] for k in ('success', 'reason', 'seconds', 'max_heading_deg', 'max_displacement_m')):
            regressions.append(dict(seed=spec[-1], push=spec[-2]))
    held = summary['selected_heldout_folded']
    nominal = summary['selected_nominal_folded']
    passed = (held['success']/held['count'] >= .95 and nominal['success'] == nominal['count']
              and not regressions and not sum(sum(r['warnings']) for r in rows if r['model'] == 'selected'))
    result = dict(simulation_only=True, requested_training_steps=40000, final_training_steps=final.num_timesteps,
        selected_checkpoint=str(selected), selected_checkpoint_sha256=sha(selected),
        final_checkpoint_sha256=sha(folder/'final_policy.zip'), parent_sha256=sha(PARENT), model_sha256=sha(xml),
        protected_standing_and_estimator_identical=True, changed_final_rising_tensors=changes,
        standing_regressions=regressions, summaries=summary, episodes=rows,
        passed_phase1_gate=passed, auto_promoted=False, auto_extended=False,
        seconds=time.monotonic()-start,
        next_step='review_before_longer_training' if passed else 'retain_parent_and_review_failed_resets',
        note='Held-out geometries excluded from training and checkpoint selection; no automatic replacement of the baseline.')
    (folder/'phase1_assessment.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'episodes'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
