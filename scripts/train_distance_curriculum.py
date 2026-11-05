"""Train 20 -> 10 -> 5 cm hold limits; advance only after held-out checks."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
STAGES = ('loose', 'medium', 'tight')


def advancement_check(groups):
    reasons = []
    for name, result in groups.items():
        episodes = result['episodes']
        rate = result['combined_success_fraction']
        minimum = .8 if name.startswith('push_') else .9
        if not episodes or rate < minimum:
            reasons.append(f'{name}: success {rate:.3f} below {minimum:.2f}')
        if name.startswith('push_') and any(e.get('push_metrics', {}).get('push_count', 0) < 1 for e in episodes):
            reasons.append(name + ': not all episodes actually received a push')
        if name.endswith('folded') and any(not e['getup_metrics']['acquired'] for e in episodes):
            reasons.append(name + ': get-up was lost')
        if any(sum(e.get('simulator_warning_counts', [])) for e in episodes):
            reasons.append(name + ': simulator warnings')
    if set(groups) != {'baseline_folded', 'baseline_standing', 'push_folded', 'push_standing'}:
        reasons.append('Missing required evaluation group')
    return dict(passed=not reasons, reasons=reasons)


def run_logged(command, log):
    print('RUN ' + ' '.join(map(str, command)), flush=True)
    with log.open('w') as output:
        process = subprocess.Popen(list(map(str, command)), cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                output.write(line)
                output.flush()
            if process.wait():
                raise RuntimeError('Command failed; see ' + str(log))
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise


def find_parent():
    runs = sorted((ROOT / 'outputs/mujoco_runs').glob('cad_getup_stand_round19_recovery_*'))
    if not runs:
        raise ValueError('No round19 run found; provide --parent explicitly.')
    run = runs[-1]
    if json.loads((run / 'status.json').read_text()).get('state') != 'completed':
        raise ValueError('Latest round19 has not completed; run this after it finishes.')
    return run / 'best/best_model.zip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, help='Optional explicit reviewed parent checkpoint')
    parser.add_argument('--start-stage', choices=STAGES, default='loose')
    parser.add_argument('--stop-after-stage', choices=STAGES, default='tight')
    parser.add_argument('--episodes', type=int, default=20, help='Held-out episodes per group')
    args = parser.parse_args()
    if args.episodes < 10 or STAGES.index(args.start_stage) > STAGES.index(args.stop_after_stage):
        parser.error('Use at least 10 episodes and ordered stages')
    # Do not compete with or alter an existing training run.
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            parts = entry.read_bytes().split(bytes([0]))
        except OSError:
            continue
        if any(Path(p.decode(errors='replace')).name == 'train.py' for p in parts if p):
            raise SystemExit('A train.py process is still running. Start the curriculum after it finishes.')
    parent = (args.parent or find_parent()).resolve()
    if not parent.is_file():
        parser.error('Parent checkpoint does not exist: ' + str(parent))
    from climber_legged_rl.evaluation_runtime import ParallelEvaluation
    from climber_legged_rl.mujoco_model import load_config
    evaluator = ParallelEvaluation(4)
    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = ROOT / 'outputs/cad_distance_curriculum' / session
    folder.mkdir(parents=True)
    manifest = dict(session=session, simulation_only=True, stages=[], state='training',
                    advancement=dict(no_push_success=.9, forced_push_success=.8, episodes=args.episodes))
    manifest_path = folder / 'progress.json'
    strict_config = ROOT / 'configs/robot/cad_getup_stand_round20_tight.yaml'
    selected_stages = STAGES[STAGES.index(args.start_stage):STAGES.index(args.stop_after_stage) + 1]
    try:
        # Prepare all three upfront so each stage can also report the original 5 cm criterion.
        for stage in STAGES:
            config = ROOT / f'configs/robot/cad_getup_stand_round20_{stage}.yaml'
            run_logged([sys.executable, 'scripts/prepare_getup.py', '--config', config], folder / f'{stage}_prepare.log')
            run_logged([sys.executable, 'scripts/smoke_mujoco.py', '--config', config], folder / f'{stage}_smoke.log')
        for index, stage in enumerate(selected_stages):
            config = ROOT / f'configs/robot/cad_getup_stand_round20_{stage}.yaml'
            settings = load_config(config)
            name = f'cad_getup_stand_round20_{stage}_{session}'
            manifest.update(state='training', current_stage=stage)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            run_logged([sys.executable, '-u', 'scripts/train.py', '--backend', 'cad-dynamics',
                        '--config', config, '--warm-start', parent, '--device', 'cpu', '--run-name', name],
                       folder / f'{stage}_training.log')
            run = ROOT / 'outputs/mujoco_runs' / name
            candidate = run / 'best/best_model.zip'
            manifest['state'] = 'evaluating'
            manifest_path.write_text(json.dumps(manifest, indent=2))
            jobs = {}
            # Per-stage seeds are separate from selection seeds; curriculum check seeds are not final test data.
            seed = 102000 + 100 * STAGES.index(stage)
            for label, path in [('stage', config), ('strict', strict_config)]:
                model = ROOT / load_config(path)['experiment']['output_directory'] / 'robot_experiment.xml'
                for pushed in (False, True):
                    for mode in ('folded', 'standing'):
                        group = ('push_' if pushed else 'baseline_') + mode
                        jobs[label + '_' + group] = dict(checkpoint=candidate, config_path=path, model_path=model,
                            episodes=args.episodes, seed=seed, reset_mode=mode, push_episode_probability=float(pushed))
            groups = evaluator.run(jobs)
            local = {k[len('stage_'):]: v for k, v in groups.items() if k.startswith('stage_')}
            check = advancement_check(local)
            report = dict(stage=stage, candidate=str(candidate),
                checkpoint_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                displacement_limit_m=settings['task']['stationary_max_displacement_m'],
                strict_displacement_limit_m=.05, check=check, groups=groups)
            (folder / f'{stage}_assessment.json').write_text(json.dumps(report, indent=2))
            manifest['stages'].append({k: v for k, v in report.items() if k != 'groups'})
            print('DISTANCE_STAGE ' + json.dumps(manifest['stages'][-1]), flush=True)
            if not check['passed']:
                manifest['state'] = 'needs_improvement'
                print('Stage did not pass. Keep its distance limit; do not advance.', flush=True)
                break
            parent = candidate
        else:
            manifest['state'] = 'completed'
        manifest_path.write_text(json.dumps(manifest, indent=2))
    except BaseException as exc:
        manifest.update(state='failed', error=repr(exc))
        manifest_path.write_text(json.dumps(manifest, indent=2))
        raise
    finally:
        evaluator.close()
    print('Curriculum report: ' + str(manifest_path), flush=True)


if __name__ == '__main__':
    main()
