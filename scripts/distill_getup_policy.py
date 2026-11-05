"""Learn successful assisted trajectories, then test an unassisted student actor."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import numpy as np
import torch
from stable_baselines3 import PPO
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_model import load_config,ROOT
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_training import validate_policy_transfer
from climber_legged_rl.getup_imitation import demonstrations,collect_student_states,fit_actor


def short_score(env, policy, horizon=6., standing_episodes=0):
    poses = env.pose_bank
    previous_max_steps = env.max_steps
    results = []
    cases = [('folded', index, pose) for index, pose in enumerate(poses)]
    cases += [('standing', index, None) for index in range(standing_episodes)]
    try:
        for mode, index, pose in cases:
            env.pose_bank = [pose] if pose is not None else poses
            obs, _ = env.reset(seed=99500+index,options={'reset_mode':mode})
            env.max_steps = round(horizon/env.dt)
            while True:
                action, _ = policy.predict(obs,deterministic=True)
                obs, _, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            results.append(dict(mode=mode,offset_m=pose['rearward_offset_m'] if pose else None, success=bool(info['combined_success']),
                acquired=bool(info['getup_acquired']),getup_seconds=info['getup_seconds'],
                position_m=info['planar_displacement_m'],reason=info['termination_reason'],
                leg_error_m=float(np.mean(abs(np.asarray(info['virtual_leg_lengths_m'])-.135))),
                tilt_rad=info['tilt_rad']))
    finally:
        env.pose_bank = poses
        env.max_steps = previous_max_steps
    return results


def selection_score(results):
    groups = {}
    for record in results:
        groups.setdefault(record['mode'], []).append(record)
    rates = [sum(r['success'] for r in group)/len(group) for group in groups.values()]
    return (min(rates), sum(rates)/len(rates), sum(r['acquired'] for r in results),
            -sum(r['leg_error_m']+.1*r['position_m']+.1*r['tilt_rad'] for r in results))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/robot/cad_getup_stand_round9_imitation.yaml'))
    parser.add_argument('--run-name')
    parser.add_argument('--updates',type=int)
    parser.add_argument('--dagger-rounds',type=int)
    parser.add_argument('--initial-checkpoint',type=Path,help='Continue imitation from a saved full-policy actor')
    parser.add_argument('--focus-offset-m',type=float,help='Increase replay weight for a prepared prone pose')
    parser.add_argument('--focus-weight',type=float,default=2.)
    parser.add_argument('--learning-rate',type=float,help='Override supervised learning rate')
    parser.add_argument('--dagger-updates',type=int)
    parser.add_argument('--selection-horizon',type=float,default=6.)
    parser.add_argument('--selection-standing-episodes',type=int,default=0)
    parser.add_argument('--no-publish-latest',action='store_true',help='Validate a run without changing the playback candidate')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(109);np.random.seed(109)
    config=load_config(args.config);settings=config['imitation']
    updates=settings['updates'] if args.updates is None else args.updates
    rounds=settings['dagger_rounds'] if args.dagger_rounds is None else args.dagger_rounds
    rate=settings['learning_rate'] if args.learning_rate is None else args.learning_rate
    dagger_updates=settings['dagger_updates'] if args.dagger_updates is None else args.dagger_updates
    if updates<1 or rounds<0:parser.error('Require positive updates and nonnegative rounds')
    if (dagger_updates<1 or not np.isfinite(rate) or rate<=0 or
            not 0<args.selection_horizon<=120 or args.selection_standing_episodes<0):
        parser.error('Invalid optimization or selection settings')
    if not np.isfinite(args.focus_weight) or args.focus_weight<=0:parser.error('Require a positive focus weight')
    validate_artifacts(args.config)
    source=ROOT/config['experiment']['output_directory']
    teacher_config=load_config(settings['teacher_config'])
    teacher_model=ROOT/teacher_config['experiment']['output_directory']/'robot_experiment.xml'
    if teacher_model.read_bytes()!=(source/'robot_experiment.xml').read_bytes():
        raise ValueError('Teacher and student must share the identical physical XML model')
    initial_checkpoint=args.initial_checkpoint or Path(settings['initial_checkpoint'])
    validate_policy_transfer(initial_checkpoint,source/'robot_experiment.xml',config,warm_start=True)
    student=make_mujoco_env(source/'robot_experiment.xml',args.config)
    if args.focus_offset_m is not None and not any(abs(p['rearward_offset_m']-args.focus_offset_m)<1e-8 for p in student.pose_bank):
        student.close()
        parser.error('Focus offset must match a prepared prone pose')
    teacher=make_mujoco_env(teacher_model,settings['teacher_config'])
    policy=PPO.load(initial_checkpoint,device='cpu')
    expert=PPO.load(settings['teacher_checkpoint'],device='cpu')
    name=args.run_name or 'cad_getup_stand_round9_imitation_seed_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    if Path(name).name!=name:parser.error('Run name must be a single directory name')
    folder=ROOT/'outputs/mujoco_runs'/name
    folder.mkdir(parents=True,exist_ok=False)
    for name in ('robot_experiment.xml','model_report.json','getup_poses.json'):
        (folder/name).write_bytes((source/name).read_bytes())
    (folder/'config.yaml').write_bytes(args.config.read_bytes())
    print('IMITATION_DIRECTORY='+str(folder),flush=True)
    report=dict(simulation_only=True, runtime_reference_assistance=False,teacher=settings['teacher_checkpoint'],
        teacher_sha256=hashlib.sha256(Path(settings['teacher_checkpoint']).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256((source/'robot_experiment.xml').read_bytes()).hexdigest(),
        initial_checkpoint=str(initial_checkpoint),
        focus_offset_m=args.focus_offset_m, focus_weight=args.focus_weight,
        optimization=dict(updates=updates,dagger_rounds=rounds,dagger_updates=dagger_updates,learning_rate=rate),
        selection=dict(horizon_seconds=args.selection_horizon,standing_episodes=args.selection_standing_episodes),
        initial_checkpoint_sha256=hashlib.sha256(initial_checkpoint.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'scripts/distill_getup_policy.py',ROOT/'source/climber_legged_rl/getup_imitation.py',
            ROOT/'source/climber_legged_rl/policy_getup.py',ROOT/'source/climber_legged_rl/supported_getup.py')},stages=[])
    best_score=None
    try:
        baseline=short_score(student,policy,args.selection_horizon,args.selection_standing_episodes)
        best_score=selection_score(baseline)
        policy.save(folder/'initial_policy.zip')
        report['baseline_selection_evaluation']=baseline
        report['best_stage']='parent'
        x,y,w,outcomes=demonstrations(student,teacher,expert,settings['trajectory_seconds'],settings['standing_episodes'],
                                     settings.get('augmentation_samples',0),args.focus_offset_m,args.focus_weight)
        report['expert_outcomes']=outcomes
        print('EXPERT_DEMONSTRATIONS '+json.dumps(outcomes),flush=True)
        for stage in range(rounds+1):
            if stage:
                mix=.75*(1.-stage/rounds)
                dx,dy,dw,outcomes=collect_student_states(student,teacher,expert,policy,settings['trajectory_seconds'],
                                                        teacher_mix=mix,focus_offset=args.focus_offset_m,focus_weight=args.focus_weight)
                x.extend(dx);y.extend(dy);w.extend(dw)
                print('STUDENT_STATES teacher_mix='+str(mix)+' '+json.dumps(outcomes),flush=True)
            loss=fit_actor(policy,x,y,w,updates if stage==0 else dagger_updates,
                           settings['batch_size'],rate,109+stage)
            policy.save(folder/f'stage_{stage}.zip')
            results=short_score(student,policy,args.selection_horizon,args.selection_standing_episodes)
            score=selection_score(results)
            if best_score is None or score>best_score:
                best_score=score
                policy.save(folder/'initial_policy.zip')
                report['best_stage']=stage
            report['stages'].append(dict(stage=stage,samples=len(x),loss=loss,selection_evaluation=results))
            (folder/'imitation_report.json').write_text(json.dumps(report,indent=2))
            print('STUDENT_EVALUATION '+json.dumps(results),flush=True)
        np.savez_compressed(folder/'demonstrations.npz',observations=np.asarray(x),actions=np.asarray(y),weights=np.asarray(w))
        report['checkpoint']=str((folder/'initial_policy.zip').relative_to(ROOT))
        report['checkpoint_sha256']=hashlib.sha256((folder/'initial_policy.zip').read_bytes()).hexdigest()
        report['best_selection_score']=best_score
        report['state']='completed'
        report['note']='Selection covers the configured horizon and initial states; independent assessment is required.'
        (folder/'imitation_report.json').write_text(json.dumps(report,indent=2))
        if not args.no_publish_latest:
            (source/'imitation_latest.json').write_text(json.dumps(report,indent=2))
        print('INITIAL_POLICY='+report['checkpoint'],flush=True)
    finally:
        student.close();teacher.close()


if __name__=='__main__':main()
