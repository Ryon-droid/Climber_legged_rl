"""One bounded, sequential three-arm pilot; native held-out test runs only at end."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
CPU='/home/climber/.venvs/climber_rl_20261104/bin/python'
GPU='/home/climber/.venvs/bpx_train/bin/python'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'source')]
from supermimic_pilot import PARENT,collect,evaluate,import_actor,write_json,sha


def run(folder):
    folder=Path(folder).resolve()
    lock=(ROOT/'outputs/spring208_phase1_launch/workflow.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();commands=[]
    state=dict(pid=os.getpid(),state='starting',started_unix=time.time(),auto_extend=False,auto_promote=False)
    def status(phase,**extra):
        state.update(state=phase,seconds=time.monotonic()-start,**extra)
        write_json(folder/'workflow.json',state)
        print(json.dumps(state),flush=True)
    def call(python,script,*arguments):
        if (folder/'STOP').exists():raise RuntimeError('Explicit stop file present')
        before=time.monotonic()
        subprocess.run([python,'-u',str(ROOT/'scripts'/script),*map(str,arguments)],cwd=ROOT,check=True)
        row=dict(script=script,arguments=list(map(str,arguments)),seconds=time.monotonic()-before)
        commands.append(row);write_json(folder/'command_times.json',commands)
        return row['seconds']
    try:
        if not json.loads((folder/'numerical_checks.json').read_text())['passed']:
            raise RuntimeError('Numerical checks missing')
        original_sha=sha(PARENT)
        status('initial_native_selection')
        initial_start=time.monotonic()
        initial=evaluate(PARENT);write_json(folder/'initial_selection.json',initial)
        initial_seconds=time.monotonic()-initial_start
        arm=folder/'world_model';arm.mkdir(exist_ok=False)
        shutil.copy2(PARENT,arm/'selected_policy.zip')
        dataset=json.loads((folder/'dataset.json').read_text())
        fit=json.loads((folder/'world_fit_0.json').read_text())
        native_steps=dataset['transitions'];best=initial['key'];curve=[]
        shared=dataset['seconds']+fit['seconds']+initial_seconds;arm_started=time.monotonic()
        for cycle in range(4):
            if cycle:
                # A whole online collection can take at most 14 x 2000 steps.
                if native_steps+28000>200704:break
                status('online_collection',cycle=cycle,native_steps=native_steps)
                path=collect(folder,arm/f'cycle_{cycle-1}.zip',cycle-1)
                native_steps+=json.loads(path.read_text())['transitions']
                status('world_model_refresh',cycle=cycle,native_steps=native_steps)
                call(GPU,'supermimic_world_model.py','fit','--folder',folder,'--cycle',cycle,'--fit-steps',800)
                call(CPU,'supermimic_pilot.py','geometry','--folder',folder,'--cycle',cycle)
            validation=json.loads((folder/f'world_validation_{cycle}.json').read_text())
            geometry=json.loads((folder/f'world_geometry_{cycle}.json').read_text())
            if not (validation['prediction_gate_passed'] and geometry['passed']):
                status('world_model_gate_failed',cycle=cycle,validation=validation,geometry=geometry)
                break
            status('world_model_policy_training',cycle=cycle,native_steps=native_steps)
            call(GPU,'supermimic_world_model.py','optimize','--folder',folder,'--cycle',cycle,'--updates',80)
            candidate=arm/f'cycle_{cycle}.zip'
            import_actor(folder,folder/f'actor_cycle_{cycle}.npz',candidate)
            status('native_policy_selection',cycle=cycle)
            selected=evaluate(candidate)
            if selected['key']>best:
                best=selected['key'];shutil.copy2(candidate,arm/'selected_policy.zip')
            record=dict(cycle=cycle,native_steps=native_steps,
                seconds=shared+time.monotonic()-arm_started,selection=selected,best_key=best)
            curve.append(record);write_json(arm/'curve.json',curve)
        write_json(arm/'training.json',dict(seed=208345,total_native_transitions=native_steps,
            inclusive_seconds=shared+time.monotonic()-arm_started,
            shared_data_seconds=dataset['seconds'],initial_world_fit_seconds=fit['seconds'],
            initial_selection_seconds=initial_seconds,
            best_key=best,cycles=len(curve),preliminary=True,
            first_time_engineering_and_unstratified_diagnostic_fit_excluded=True))
        for mode,extra in [('ppo_current',[]),('ppo_reference',['--reference'])]:
            status(mode+'_training')
            call(CPU,'supermimic_pilot.py','ppo','--folder',folder,*extra)
        status('final_heldout_assessment')
        results={}
        for name,cp in [('parent',PARENT),*[(name,folder/name/'selected_policy.zip')
                                         for name in ('world_model','ppo_current','ppo_reference')]]:
            result=evaluate(cp,'test',name)
            results[name]=result;write_json(folder/f'test_{name}.json',result)
        timing={name:json.loads((folder/name/'training.json').read_text()) for name in ('world_model','ppo_current','ppo_reference')}
        summaries={name:r['summary'] for name,r in results.items()}
        def passed(s):
            return (s['folded']['success']/s['folded']['count']>=.95
                and s['nominal']['success']==s['nominal']['count']
                and s['standing']['success']==summaries['parent']['standing']['success']
                and not sum(v['warnings'] for v in s.values()))
        report=dict(simulation_only=True,preliminary_single_training_seed=True,
            method='Small SuperMimic-inspired adaptation; not official reproduction',
            seed=208345,source_policy_sha256=original_sha,summaries=summaries,timing=timing,
            passed_95_percent_gate={name:passed(s) for name,s in summaries.items()},
            task_criteria_unchanged=True,runtime_world_model_or_teacher=False,
            limitations=['One training seed; confirmation requires at least 3 independent seeds.',
                        'CPU MuJoCo data collection; GPU small neural model, not paper full MuJoCo Warp stack.',
                        'Existing known held-out geometries with entirely new seeds; not unseen robot geometries.',
                        'No hardware claim; no automatic promotion.',
                        'Model validation, collection, optimization and selection included in primary timing; implementation effort excluded.'])
        write_json(folder/'comparison.json',report)
        if sha(PARENT)!=original_sha:raise RuntimeError('Parent checkpoint changed')
        status('completed',comparison=str(folder/'comparison.json'))
    except BaseException as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True)
    run(p.parse_args().folder)
