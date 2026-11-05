"""Collect successful stable and recovery teachers with the actor history only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'


def recovery_window(step, dt, active, pushed, initial_seconds=1.5, push_tail_seconds=0.):
    """Optionally retain the settling tail after a push has met recovery gates."""
    import math
    if (not math.isfinite(initial_seconds) or initial_seconds <= 0
            or not math.isfinite(push_tail_seconds) or push_tail_seconds < 0):
        raise ValueError('Rehearsal window durations must be finite and valid')
    if not pushed:
        return step * dt < initial_seconds
    return active is not None and (not active['recovered'] or
        (push_tail_seconds > 0 and step * dt < active['end_step'] * dt + push_tail_seconds))


def collect(config_path, stable, recovery, output, initial_seconds=1.5, push_tail_seconds=0.):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from climber_legged_rl.mujoco_env import make_mujoco_env
    from climber_legged_rl.mujoco_model import load_config
    from climber_legged_rl.mujoco_training import validate_policy_transfer
    torch.set_num_threads(1)
    cfg = load_config(config_path)
    xml = ROOT/cfg['experiment']['output_directory']/'robot_experiment.xml'
    if output.exists():
        raise ValueError('Use a new rehearsal output directory')
    policies = {}
    for name, checkpoint in [('stable', stable), ('recovery', recovery)]:
        validate_policy_transfer(checkpoint, xml, cfg, warm_start=True)
        policies[name] = PPO.load(checkpoint, device='cpu')
    specs = [('stable', 'standing', s, 0.) for s in
             [99905, 99911, 99914, 104014, 262216, 262217]]
    specs += [('stable', 'folded', 41000, 0.), ('recovery', 'standing', 261409, 0.),
              ('recovery', 'standing', 261106, 1.), ('recovery', 'standing', 261704, 1.)]
    accepted, outcomes = [], []
    env = make_mujoco_env(xml, config_path)
    try:
        for role, mode, seed, probability in specs:
            env.unwrapped.push.config = dict(env.unwrapped.push.config, episode_probability=probability)
            obs, _ = env.reset(seed=seed, options={'reset_mode': mode})
            rows = []
            while True:
                action, _ = policies[role].predict(obs, deterministic=True)
                # Retain a bounded recovery and settling window. The stable
                # teacher supplies the remaining steady-state behavior.
                active = env.unwrapped.push.active
                transient = recovery_window(env.unwrapped.steps, env.unwrapped.dt, active,
                                             bool(probability), initial_seconds, push_tail_seconds)
                if obs[179] > .95 and (role == 'stable' or transient):
                    rows.append((obs[:180].copy(), np.clip(action, -1., 1.)))
                obs, _, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            success = bool(info['combined_success'])
            kept = len(rows) if success else 0
            outcomes.append(dict(role=role, mode=mode, seed=seed, push_probability=probability,
                success=success, retained_samples=kept, push_count=info['push_metrics']['push_count']))
            print(json.dumps(outcomes[-1]), flush=True)
            if kept:
                accepted.append((role, rows))
    finally:
        env.close()
    counts = {role: sum(r == role for r, _ in accepted) for role in policies}
    if not all(counts.values()):
        raise RuntimeError('Both stable and recovery teachers need successful trajectories')
    xs, ys, weights = [], [], []
    for role, rows in accepted:
        mass = .9 if role == 'stable' else .1
        for x, y in rows:
            xs.append(x); ys.append(y); weights.append(mass/counts[role]/len(rows))
    output.mkdir(parents=True)
    dataset = output/'standing.npz'
    np.savez_compressed(dataset, observations=np.asarray(xs, dtype=np.float32),
        actions=np.asarray(ys, dtype=np.float32), weights=np.asarray(weights, dtype=np.float64))
    (output/'config.yaml').write_bytes(config_path.read_bytes())
    report = dict(simulation_only=True, successful_data_only=True,
        runtime_reference_assistance=False, actor_dimensions=180, privileged_labels_saved=False,
        model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(), outcomes=outcomes,
        category_mass=dict(stable=.9, recovery=.1),
        recovery_filter=dict(initial_seconds=initial_seconds, push_tail_seconds=push_tail_seconds,
                             include_unrecovered_push=True),
        checkpoints={name:dict(path=str(cp), sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
                     for name, cp in [('stable', stable), ('recovery', recovery)]})
    dataset.with_suffix('.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stable', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-recovery-seconds', type=float, default=1.5)
    parser.add_argument('--push-tail-seconds', type=float, default=0.)
    args = parser.parse_args()
    collect(args.config, args.stable, args.recovery, args.output,
            args.initial_recovery_seconds, args.push_tail_seconds)
