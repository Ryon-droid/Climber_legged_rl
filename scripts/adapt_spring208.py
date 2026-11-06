"""Learn the new measured stop using real dynamics and unchanged motor limits."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
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
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.velocity_estimation import VelocityEstimationPPO
from climber_legged_rl.mujoco_training import validate_policy_transfer

CONFIG = ROOT/'configs/robot/cad_balance_spring208.yaml'
PARENT = ROOT/'outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip'
OUT = ROOT/'outputs/spring208_adaptation'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def teacher_action(policy, env, obs, recipe, delta):
    if env.acquired:
        return policy.predict(obs, deterministic=True)[0]
    x = obs.copy()
    blend = max(0., 1.-env.steps*env.dt/recipe['seconds'])
    x[:180].reshape(5, 36)[:, 8:12] += delta*recipe['bias']*blend
    action = policy.predict(x, deterministic=True)[0]
    if recipe['symmetric']:
        action[:4] = np.tile((action[:2]+action[2:4])/2., 2)
        wheel = (action[4]-action[5])/2.
        action[4:] = [wheel, -wheel]
    return np.clip(action, -1., 1.)


def rollout(recipe, seed=325000, student=None, mix=1., collect=False):
    torch.set_num_threads(1)
    config = load_config(CONFIG)
    env = make_mujoco_env(ROOT/config['experiment']['output_directory']/'robot_experiment.xml', CONFIG)
    env.unwrapped.push.config = dict(env.push.config, episode_probability=0.)
    expert = PPO.load(PARENT, device='cpu')
    policy = PPO.load(student, device='cpu') if student else None
    old = json.loads((ROOT/'outputs/cad_balance_estimator_round15/getup_poses.json').read_text())['poses'][0]
    delta = np.array(old['leg_reference_rad'])-np.array(env.pose_bank[0]['leg_reference_rad'])
    xs, ys = [], []
    try:
        obs, reset = env.reset(seed=seed, options={'reset_mode': 'folded'})
        while True:
            action = teacher_action(expert, env, obs, recipe, delta)
            if collect and not env.acquired:
                xs.append(obs.copy())
                ys.append(action.copy())
            if policy is not None:
                action = mix*action+(1.-mix)*policy.predict(obs, deterministic=True)[0]
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                result = dict(recipe=recipe, seed=seed, success=bool(info['combined_success']),
                    acquired=bool(info['getup_acquired']), seconds=env.steps*env.dt,
                    getup_seconds=info['getup_seconds'], reason=info['termination_reason'] or 'timeout',
                    final_tilt_rad=info['tilt_rad'], final_position_m=info['planar_displacement_m'],
                    leg_error_m=float(np.mean(abs(np.array(info['virtual_leg_lengths_m'])-.135))),
                    warning_count=int(np.sum(env.data.warning.number)),
                    peak_torque_Nm=info['peak_torque_Nm'])
                return result, xs, ys
    finally:
        env.close()


def probe():
    recipes = [dict(bias=bias, seconds=seconds, symmetric=symmetric)
               for bias in (0., .5, 1., 1.5) for seconds in (.4, 1., 2.) for symmetric in (False, True)
               if bias or seconds == .4]
    rows = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(rollout, recipe) for recipe in recipes]
        for future in as_completed(jobs):
            row = future.result()[0]
            rows.append(row)
            print(json.dumps(row), flush=True)
    rows.sort(key=lambda r: (r['success'], r['acquired'], -r['leg_error_m']-.1*r['final_tilt_rad']), reverse=True)
    report = dict(simulation_only=True, actual_dynamics=True, model_sha256=digest(
        ROOT/'outputs/cad_balance_spring208/robot_experiment.xml'), parent_sha256=digest(PARENT),
        rows=rows, successful=[r for r in rows if r['success']], note='Teacher probes are not autonomous student performance.')
    (OUT/'probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PROBE_SUCCESS_COUNT='+str(len(report['successful'])), flush=True)


def save_student(model, folder):
    # Persist the gradient setting as well as the tensors so a future loader
    # cannot silently refreeze the newly trained rising branch.
    model.policy.freeze_rising_actor = False
    model.policy_kwargs['freeze_rising_actor'] = False
    model.save(folder/'final_policy.zip')


def distill():
    started = time.monotonic()
    torch.manual_seed(208)
    np.random.seed(208)
    probe_report = json.loads((OUT/'probe.json').read_text())
    xml = ROOT/'outputs/cad_balance_spring208/robot_experiment.xml'
    if probe_report['parent_sha256'] != digest(PARENT) or probe_report['model_sha256'] != digest(xml):
        raise RuntimeError('Teacher probe provenance changed')
    if not probe_report['successful']:
        raise RuntimeError('No verified successful teacher recipe; use PPO exploration instead')
    recipe = probe_report['successful'][0]['recipe']
    outcome, xs, ys = rollout(recipe, collect=True)
    if not outcome['success']:
        raise RuntimeError('Selected teacher no longer succeeds under actual dynamics')
    folder = ROOT/'outputs/mujoco_runs'/('spring208_rising_distill_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(exist_ok=False)
    config = load_config(CONFIG)
    config['training']['freeze_rising_actor'] = False
    config['training']['freeze_standing_actor'] = True
    import yaml
    (folder/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    for name in ('robot_experiment.xml', 'model_report.json', 'getup_poses.json'):
        (folder/name).write_bytes((xml.parent/name).read_bytes())
    model = VelocityEstimationPPO.load(PARENT, device='cpu')
    original = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    model.policy.requires_grad_(False)
    actor = model.policy.mlp_extractor.policy_net
    trainable = list(actor.rising_net.parameters())+list(actor.rising_head.parameters())
    for p in trainable:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(trainable, lr=3e-5)
    stages = []
    save_student(model, folder)
    initial_path = folder/'initial_policy.zip'
    model.save(initial_path)
    best_key = (-1, -1, -float('inf'))
    best = None
    for stage, mix in enumerate((1., .75, .5, .25, 0., 0., 0., 0.)):
        if stage:
            teacher_outcome, more_x, more_y = rollout(recipe, student=folder/'final_policy.zip',
                mix=mix, collect=True)
            # DAgger labels are explicitly identified, including failed rollout
            # states. They never count as successful autonomous demonstrations.
            xs.extend(more_x)
            ys.extend(more_y)
        else:
            teacher_outcome = outcome
        tx = torch.as_tensor(np.asarray(xs), dtype=torch.float32)
        ty = torch.as_tensor(np.asarray(ys), dtype=torch.float32)
        losses = []
        for update in range(2000 if stage == 0 else 1000):
            ids = torch.randint(len(tx), (256,))
            prediction = actor(tx[ids]).clamp(-1, 1)
            loss = ((prediction-ty[ids])/torch.tensor([.1]*4+[.25]*2)).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        save_student(model, folder)
        score, _, _ = rollout(recipe, seed=335000, student=folder/'final_policy.zip', mix=0.)
        key = (int(score['success']), int(score['acquired']), -score['leg_error_m']-.1*score['final_tilt_rad'])
        if key > best_key:
            best_key, best = key, copy.deepcopy(score)
            model.save(folder/'best_policy.zip')
        row = dict(stage=stage, teacher_mix=mix, labelled_trajectory=teacher_outcome,
            samples=len(xs), first_loss=float(np.mean(losses[:50])), last_loss=float(np.mean(losses[-50:])),
            autonomous_evaluation=score)
        stages.append(row)
        print(json.dumps(row), flush=True)
        if score['success'] and stage >= 1:
            break
    # Select only by a complete autonomous rollout; keep the last candidate
    # separate when the best score belonged to an earlier update.
    best_model = VelocityEstimationPPO.load(folder/'best_policy.zip', device='cpu')
    changed = []
    for name, value in best_model.policy.state_dict().items():
        if name.startswith(('mlp_extractor.policy_net.rising_net.', 'mlp_extractor.policy_net.rising_head.')):
            if not torch.equal(value, original[name]):
                changed.append(name)
        elif not torch.equal(value, original[name]):
            raise RuntimeError('Protected standing/estimator/critic weight changed: '+name)
    if not changed:
        raise RuntimeError('No rising tensors updated')
    best_model.save(folder/'final_policy.zip')
    np.savez_compressed(folder/'teacher_labels.npz', observations=np.asarray(xs), actions=np.asarray(ys))
    report = dict(simulation_only=True, method='successful_teacher_distillation_and_dagger',
        runtime_teacher_assistance=False, model_sha256=digest(xml), parent=str(PARENT), parent_sha256=digest(PARENT),
        checkpoint=str(folder/'final_policy.zip'), checkpoint_sha256=digest(folder/'final_policy.zip'),
        teacher_recipe=recipe, teacher_verified_outcome=outcome, stages=stages, selection=best,
        changed_tensors=changed, standing_estimator_critic_tensors_identical=True,
        elapsed_seconds=time.monotonic()-started, dataset_sha256=digest(folder/'teacher_labels.npz'),
        dataset_kind='verified_teacher_trajectory_plus_teacher_labelled_student_rollouts',
        selection_is_not_independent_assessment=True)
    (folder/'adaptation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    (OUT/'distillation_latest.json').write_text(json.dumps(report, indent=2)+'\n')
    print('ADAPTATION_REPORT='+str(folder/'adaptation_report.json'), flush=True)
    print(json.dumps(dict(checkpoint=report['checkpoint'], selection=best, seconds=report['elapsed_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['probe', 'distill'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    validate_artifacts(CONFIG)
    config = load_config(CONFIG)
    validate_policy_transfer(PARENT, ROOT/config['experiment']['output_directory']/'robot_experiment.xml', config, warm_start=True)
    OUT.mkdir(exist_ok=True)
    if args.mode == 'probe':
        probe()
    elif args.mode == 'distill':
        distill()


if __name__ == '__main__':
    main()
