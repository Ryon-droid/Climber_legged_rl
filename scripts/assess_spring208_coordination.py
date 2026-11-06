"""Native-policy paired assessment; corrective teachers are never invoked here."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'scripts')]
import torch
from stable_baselines3 import PPO
from assess_spring208_phase3 import case
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_model import load_config
from improve_spring208_coordination import PARENT,PARENT_SHA,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    folder=args.run_dir.resolve()
    torch.set_num_threads(1)
    cp=folder/'selected_policy.zip'
    if sha(PARENT)!=PARENT_SHA:raise RuntimeError('Unexpected parent')
    parent=PPO.load(PARENT,device='cpu');model=PPO.load(cp,device='cpu')
    changed=[]
    for key,value in parent.policy.state_dict().items():
        other=model.policy.state_dict()[key]
        if key.startswith(('mlp_extractor.policy_net.rising_net.','mlp_extractor.policy_net.rising_head.')):
            if not torch.equal(value,other):changed.append(key)
        elif not torch.equal(value,other):raise RuntimeError('Protected tensor changed: '+key)
    test=ROOT/'configs/robot/cad_spring208_phase3_test.yaml'
    for split in ('train','test','nominal'):validate_artifacts(ROOT/f'configs/robot/cad_spring208_phase3_{split}.yaml')
    config=load_config(test)
    offsets=config['getup']['rearward_offsets_m']
    specs=[('test',offset,'folded',False,seed) for offset in offsets for seed in range(475000,475016)]
    specs += [('nominal',None,'folded',False,seed) for seed in range(476000,476008)]
    specs += [('nominal',None,'standing',push,seed) for push in (False,True) for seed in range(477000,477008)]
    specs += [('nominal',None,'standing',False,261409),('nominal',None,'standing',False,262219),
        ('nominal',None,'standing',True,261106),('nominal',None,'standing',True,261704)]
    # Previously inspected failures are explicitly marked regression cases, not fresh evidence.
    specs += [('test',min(offsets),'folded',False,seed) for seed in (445000,445001,445002)]
    jobs=[(label,str(checkpoint),*spec) for label,checkpoint in [('parent',PARENT),('selected',cp)] for spec in specs]
    rows=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending=[pool.submit(case,job) for job in jobs]
        for future in as_completed(pending):
            rows.append(future.result())
            if len(rows)%16==0:print(json.dumps(dict(completed=len(rows),total=len(jobs),seconds=time.monotonic()-started)),flush=True)
    summaries={}
    for label in ('parent','selected'):
        subset=[r for r in rows if r['model']==label]
        groups=dict(fresh_folded=[r for r in subset if r['split']=='test' and r['seed']>=475000],
            nominal_folded=[r for r in subset if r['split']=='nominal' and r['mode']=='folded'],
            ordinary_standing=[r for r in subset if r['mode']=='standing' and r['seed']>=477000],
            hard_standing=[r for r in subset if r['mode']=='standing' and r['seed']<477000],
            inspected_regressions=[r for r in subset if r['split']=='test' and r['seed']<475000])
        for name,group in groups.items():
            summaries[label+'_'+name]=dict(count=len(group),success=sum(r['success'] for r in group),
                warnings=sum(sum(r['warnings']) for r in group),
                max_upper_penetration_mm=max(max(r['upper_stop_penetration_mm']) for r in group),
                max_closure_m=max(r['max_closure_m'] for r in group),
                failures=[dict(seed=r['seed'],offset=r['offset_m'],reason=r['reason']) for r in group if not r['success']])
    regression=[]
    lookup={(r['model'],r['split'],r['offset_m'],r['mode'],r['push'],r['seed']):r for r in rows}
    for spec in specs:
        if spec[2]!='standing':continue
        a,b=lookup[('parent',*spec)],lookup[('selected',*spec)]
        if any(a[k]!=b[k] for k in ('success','reason','seconds','max_heading_deg','max_displacement_m')):
            regression.append(dict(seed=spec[-1],push=spec[-2]))
    fresh=summaries['selected_fresh_folded'];prior=summaries['parent_fresh_folded']
    nominal=summaries['selected_nominal_folded']
    improved=bool(changed and fresh['success']>prior['success'] and nominal['success']==nominal['count']
        and not regression and fresh['warnings']==0)
    passed=bool(improved and fresh['success']/fresh['count']>=.95)
    report=dict(simulation_only=True,runtime_teacher_assistance=False,teacher_used_in_assessment=False,
        method='corrective_demonstrations_and_dagger',candidate=str(cp),candidate_sha256=sha(cp),
        parent_sha256=sha(PARENT),model_sha256=sha(ROOT/config['experiment']['output_directory']/'robot_experiment.xml'),
        changed_rising_tensors=changed,protected_tensors_identical=True,standing_regressions=regression,
        summaries=summaries,episodes=rows,improved_on_fresh_cases=improved,passed_95_percent_gate=passed,
        seconds=time.monotonic()-started,auto_promoted=False,
        note='48 fresh seeds on existing held-out geometries; three inspected cases are regression only. Hardware unvalidated.')
    (folder/'coordination_assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2),flush=True)


if __name__=='__main__':main()
