"""Benchmark unchanged seeded evaluation workloads with serial and parallel workers."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
from climber_legged_rl.evaluation_runtime import ParallelEvaluation
from climber_legged_rl.mujoco_model import ROOT,load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=8)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.episodes<1:parser.error('Positive episode count required')
    torch.set_num_threads(1)
    c=load_config(args.config);physical=ROOT/c['experiment']['output_directory']/'robot_experiment.xml'
    common=dict(checkpoint=args.checkpoint,config_path=args.config,model_path=physical,episodes=args.episodes)
    jobs={label+'_'+mode:dict(common,seed=100700,reset_mode=mode,push_episode_probability=probability)
        for label,probability in [('no_push',0.),('forced_push',1.)] for mode in ['standing','folded']}
    timings={};results={}
    serial=ParallelEvaluation(1);parallel=ParallelEvaluation(args.workers)
    try:
        for label,engine in [('serial',serial),('parallel_cold',parallel),('parallel_warm',parallel)]:
            started=time.monotonic();results[label]=engine.run(jobs);timings[label]=time.monotonic()-started
            print(label,round(timings[label],3),flush=True)
        equal={label:results['serial']==results[label] for label in ['parallel_cold','parallel_warm']}
        report=dict(simulation_only=True,config=str(args.config),checkpoint=str(args.checkpoint),
            episodes_per_group=args.episodes,workers=args.workers,seconds=timings,exact_results_equal=equal,
            cold_speedup=timings['serial']/timings['parallel_cold'],
            warm_speedup=timings['serial']/timings['parallel_warm'])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        if not all(equal.values()):raise RuntimeError('Parallel evaluation differs from serial results')
    finally:serial.close();parallel.close()

if __name__=='__main__':main()
