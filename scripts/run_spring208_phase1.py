"""One bounded 40k PPO pilot followed by independent assessment, with a process lock."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name:
        raise ValueError('Invalid run name')
    out = ROOT/'outputs/spring208_phase1_launch'
    out.mkdir(exist_ok=True)
    lock = (out/'workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    folder = ROOT/'outputs/mujoco_runs'/args.run_name
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    status = dict(run_name=args.run_name, pid=os.getpid(), state='starting', requested_steps=40000,
        run_directory=str(folder), started_utc=datetime.now(timezone.utc).isoformat(), auto_extend=False)
    status_path = out/(args.run_name+'.json')

    def save(state, **extra):
        status.update(state=state, **extra)
        status_path.write_text(json.dumps(status, indent=2)+'\n')

    started = time.monotonic()
    try:
        save('training')
        command = [sys.executable, '-u', str(ROOT/'scripts/train.py'), '--backend', 'cad-dynamics',
            '--config', 'configs/robot/cad_spring208_phase1_train.yaml', '--device', 'cpu',
            '--num-envs', '4', '--total-timesteps', '40000', '--warm-start',
            'outputs/mujoco_runs/spring208_rising_distill_20261106_031028/final_policy.zip',
            '--run-name', args.run_name]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        save('independent_assessment', training_elapsed_seconds=time.monotonic()-started)
        subprocess.run([sys.executable, '-u', str(ROOT/'scripts/assess_spring208_phase1.py'),
            '--run-dir', str(folder)], cwd=ROOT, env=env, check=True)
        report = json.loads((folder/'phase1_assessment.json').read_text())
        save('completed', elapsed_seconds=time.monotonic()-started,
            gate_passed=report['passed_phase1_gate'], report=str(folder/'phase1_assessment.json'),
            auto_promoted=False)
    except BaseException as error:
        save('failed', error=repr(error), elapsed_seconds=time.monotonic()-started)
        raise


if __name__ == '__main__':
    main()
