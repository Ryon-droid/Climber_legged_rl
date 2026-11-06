"""Bounded corrective learning and native-policy independent assessment."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-name',required=True)
    args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:raise ValueError('Invalid run name')
    out=ROOT/'outputs/spring208_coordination_fix';out.mkdir(exist_ok=True)
    lock=(ROOT/'outputs/spring208_phase1_launch/workflow.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    folder=ROOT/'outputs/mujoco_runs'/args.run_name
    if folder.exists():raise RuntimeError('Run exists')
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    status=dict(run_name=args.run_name,pid=os.getpid(),run_directory=str(folder),
        started_utc=datetime.now(timezone.utc).isoformat(),auto_extend=False)
    started=time.monotonic()
    def save(state,**extra):
        status.update(state=state,**extra)
        (out/(args.run_name+'.json')).write_text(json.dumps(status,indent=2)+'\n')
    try:
        save('corrective_training')
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/improve_spring208_coordination.py'),
            'distill','--run-name',args.run_name],cwd=ROOT,env=env,check=True)
        save('native_policy_assessment',training_seconds=time.monotonic()-started)
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/assess_spring208_coordination.py'),
            '--run-dir',str(folder)],cwd=ROOT,env=env,check=True)
        result=json.loads((folder/'coordination_assessment.json').read_text())
        save('completed',seconds=time.monotonic()-started,
            improved=result['improved_on_fresh_cases'],passed_95_percent_gate=result['passed_95_percent_gate'])
    except BaseException as error:
        save('failed',error=repr(error),seconds=time.monotonic()-started);raise


if __name__=='__main__':main()
