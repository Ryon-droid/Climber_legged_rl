"""Allow short PPO only for warm-ups preserving ordinary task success and severity."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import sys
import argparse
import json
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'source'))
from climber_legged_rl.evaluation_runtime import ParallelEvaluation


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stamp',required=True); args=parser.parse_args()
    cfg=ROOT/'configs/robot/cad_balance_estimator_round15.yaml'
    xml=ROOT/'outputs/cad_balance_estimator_round15/robot_experiment.xml'
    models=dict(baseline=ROOT/'outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip')
    for t in ('a','b'): models[t]=ROOT/f'outputs/mujoco_runs/cad_balance_estimator_round18{t}_warmup_{args.stamp}/final_policy.zip'
    jobs={}
    for label,cp in models.items():
        specs=[('standing','standing',275000,20,False),('folded','folded',275000,1,False)]
        specs += [(str(s),'standing',s,1,push) for s,push in ((261409,False),(262219,False),
            (261106,True),(261704,True),(275013,True),(295009,True),(295004,False))]
        for key,mode,seed,n,push in specs:
            jobs[label+'_'+key]=dict(checkpoint=cp,config_path=cfg,model_path=xml,episodes=n,seed=seed,
                reset_mode=mode,push_episode_probability=float(push))
    evaluator=ParallelEvaluation(4)
    try: results=evaluator.run(jobs)
    finally: evaluator.close()
    output=ROOT/f'outputs/round18_warmup_{args.stamp}'
    eligible=[]; decisions={}
    for key,result in results.items(): (output/(key+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    for label in ('a','b'):
        violations=[]; improved=[]
        for key,baseline in results.items():
            if not key.startswith('baseline_'): continue
            task=key[len('baseline_'):]; candidate=results[label+'_'+task]
            for index,(old,new) in enumerate(zip(baseline['episodes'],candidate['episodes'])):
                if old['combined_success'] and not new['combined_success']: violations.append(f'{task}:{index}:lost_success')
                if (new['max_heading_error_deg']>old['max_heading_error_deg']+3
                    or new['max_planar_displacement_m']>old['max_planar_displacement_m']+.03
                    or (new['reason']=='base_contact' and new['seconds']<old['seconds']-.5)):
                    violations.append(f'{task}:{index}:severity_regression')
                if not old['combined_success'] and new['combined_success']: improved.append(f'{task}:{index}')
        # This gate permits bounded 40k investigation, never a baseline promotion.
        decisions[label]=dict(eligible=not violations,violations=violations,improved_cases=improved)
        if not violations: eligible.append(label)
    report=dict(simulation_only=True,eligible_trials=eligible,decisions=decisions,
        next_step='bounded_40k_ppo' if eligible else 'stop_before_ppo',auto_promotion=False)
    (output/'gate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
