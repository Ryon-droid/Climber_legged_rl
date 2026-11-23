"""Training-only matched recovery traces; no optimizer or test seeds."""
import argparse
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'source'), str(ROOT / 'scripts')]
import numpy as np
import torch
from stable_baselines3 import PPO
from climber_legged_rl.motor_authority import apply_standing_motor_limit
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.push_balance_curriculum import PushBalanceCurriculum
from train_spring208_push import CFG, write
from prepare_spring208_phase3 import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/push_reactive_balance18_20261115')
    parser.add_argument('--forces', nargs='+', type=float, default=[12., 18.])
    parser.add_argument('--leg-kp', type=float)
    parser.add_argument('--wheel-kv', type=float)
    parser.add_argument('--parent-only', action='store_true')
    parser.add_argument('--failed-count', type=int, default=4)
    parser.add_argument('--discovered-forces', action='store_true',
                        help='Replay each training discovery force range, including mixed forces')
    parser.add_argument('--oracle-velocity', action='store_true',
                        help='Diagnostic only: replace velocity estimate with simulator truth')
    parser.add_argument('--skip-first-push', action='store_true',
                        help='Diagnostic only: zero the first pulse without changing pulse RNG')
    parser.add_argument('--zero-hold-observations', action='store_true',
                        help='Diagnostic only: mask position/heading history at policy input')
    args = parser.parse_args()
    script_digest = sha(Path(__file__))
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite diagnostic')
    torch.set_num_threads(1)
    source = args.source.resolve()
    discovery = json.loads((source / 'training_discovery.json').read_text())
    failed = [r for r in discovery['outcomes'] if not r['success'] and r['spec']['push']]
    selected = failed[:args.failed_count]
    selected += [r for r in discovery['outcomes'] if r['success'] and r['spec']['push']][:1]
    selected += [r for r in discovery['outcomes'] if not r['spec']['push']][:1]
    policies = dict(parent=source / 'selected_policy.zip',
                    rejected=source / 'level18_attempt0_step491520.zip')
    if args.parent_only:
        policies.pop('rejected')
    rows = []
    for name, checkpoint in policies.items():
        policy = PPO.load(checkpoint, device='cpu')
        actor = policy.policy.mlp_extractor.policy_net
        if args.oracle_velocity:
            actor.estimate_velocity = lambda observations: observations[:, -3:]
        for case in selected:
            ranges = ([case['spec']['force_N']] if args.discovered_forces
                      else [[f, f] for f in args.forces]) if case['spec']['push'] else [[0., 0.]]
            for limits in ranges:
                force = limits[1]
                config = load_config(CFG)
                env = make_mujoco_env(ROOT / config['experiment']['output_directory'] / 'robot_experiment.xml', CFG)
                base = env.unwrapped
                if args.wheel_kv is not None:
                    if not np.isfinite(args.wheel_kv) or args.wheel_kv <= 0:
                        raise ValueError('Invalid diagnostic wheel kv')
                    base.control = dict(base.control, wheel_kv=args.wheel_kv)
                if args.skip_first_push:
                    native_force = base._push_force
                    def diagnostic_force():
                        force = native_force()
                        return np.zeros(3) if len(base.push.events) == 1 else force
                    base._push_force = diagnostic_force
                apply_standing_motor_limit(env, 20.)
                if args.leg_kp is not None:
                    if not np.isfinite(args.leg_kp) or args.leg_kp <= 0:
                        raise ValueError('Invalid diagnostic leg kp')
                    base.envelope = dict(base.envelope, standing_leg_kp=args.leg_kp)
                    base._set_motor_envelope()
                base.push.config = dict(base.push.config, episode_probability=float(force > 0),
                                        force_N=list(limits) if force else [2., 18.])
                base.push.config.pop('force_bands', None)
                # Match the training reward terms, without adding training noise.
                base.balance_recovery.settings = dict(base.balance_recovery.settings,
                    tilt_weight=8., tilt_rate_weight=2., velocity_weight=3., yaw_rate_weight=1.)
                env = PushBalanceCurriculum(env, 'balance', sustained_reward=2.)
                trace = []
                try:
                    obs, reset_info = env.reset(seed=case['spec']['seed'], options={'reset_mode': 'standing'})
                    while True:
                        with torch.no_grad():
                            tensor = torch.as_tensor(obs[None], dtype=torch.float32)
                            if args.zero_hold_observations:
                                tensor[:, :-3].reshape(1, -1, 36)[:, :, 24:28] = 0.
                            raw = policy.policy.get_distribution(tensor).distribution.mean.cpu().numpy()[0]
                            estimated_velocity = actor.estimate_velocity(tensor).cpu().numpy()[0]
                        velocity_before = base._kinematics()[1].copy()
                        action = np.clip(raw, -1., 1.)
                        targets = base._leg_targets(action).copy()
                        before = base.data.qpos[base.q_indices[:4]].copy()
                        obs, reward, done, truncated, info = env.step(action)
                        angular = base._kinematics()[0]
                        applied = base.action.copy()
                        applied_targets = base._leg_targets(applied)
                        rotation = base._rotation()
                        trace.append(dict(seconds=base.steps * base.dt,
                            tilt_deg=float(np.degrees(info['tilt_rad'])),
                            angular_velocity_rad_s=angular.tolist(),
                            linear_velocity_body_m_s=base._kinematics()[1].tolist(),
                            projected_gravity=base._kinematics()[2].tolist(),
                            raw_action=raw.tolist(), action=action.tolist(),
                            estimated_velocity_before_m_s=estimated_velocity.tolist(),
                            true_velocity_before_m_s=velocity_before.tolist(),
                            applied_action=applied.tolist(),
                            applied_leg_target_rad=applied_targets.tolist(),
                            applied_leg_error_rad=(applied_targets-before).tolist(),
                            leg_target_rad=targets.tolist(), leg_error_rad=(targets-before).tolist(),
                            leg_angle_rad=base.data.qpos[base.q_indices[:4]].tolist(),
                            wheel_speed_rad_s=base.data.qvel[base.v_indices[4:]].tolist(),
                            wheel_target_rad_s=(action[4:]*base.control['wheel_action_scale_rad_s']*base.wheel_action_sign).tolist(),
                            applied_wheel_target_rad_s=(applied[4:]*base.control['wheel_action_scale_rad_s']*base.wheel_action_sign).tolist(),
                            heading_rad=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                            wheel_contacts=info['wheel_contacts'],
                            torque_Nm=info['motor_torque_Nm'], saturation=info['motor_saturation_fraction'],
                            speed_m_s=info['planar_speed_m_s'], reward_terms=info['reward_terms'],
                            force_N=info['push_force_world_N']))
                        if done or truncated:
                            break
                    post = [t for t in trace if t['seconds'] >= base.push.config['first_push_seconds']]
                    samples = post or trace
                    rewards = {k:float(np.mean([s['reward_terms'].get(k, 0.) for s in samples]))
                               for k in set().union(*(s['reward_terms'] for s in samples))}
                    rows.append(dict(policy=name, seed=case['spec']['seed'], force_N=force,
                        force_range_N=limits, reset_info=reset_info,
                        success=bool(info['combined_success']), reason=info['termination_reason'],
                        seconds=base.steps*base.dt, mean_post_push_rewards=rewards,
                        max_raw_action=np.max(np.abs([t['raw_action'] for t in samples]), axis=0).tolist(),
                        clipped_fraction=np.mean(np.abs([t['raw_action'] for t in samples])>1., axis=0).tolist(),
                        peak_torque_Nm=info['peak_torque_Nm'],
                        push_events=info['push_metrics']['push_events'], trace=trace))
                    print(json.dumps(dict(policy=name, seed=case['spec']['seed'], force_N=limits,
                        success=rows[-1]['success'], reason=rows[-1]['reason']), ensure_ascii=False), flush=True)
                finally:
                    env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, dict(training_only=True, source=str(source),
        uses_discovered_force_ranges=args.discovered_forces,
        diagnostic_oracle_velocity=args.oracle_velocity,
        diagnostic_skip_first_push=args.skip_first_push,
        diagnostic_zero_hold_observations=args.zero_hold_observations,
        diagnostic_wheel_kv=args.wheel_kv,
        diagnostic_script_sha256=script_digest,
        diagnostic_standing_leg_kp=args.leg_kp,
        checkpoints={k:dict(path=str(p), sha256=sha(p)) for k,p in policies.items()}, cases=rows))
    print(json.dumps(dict(output=str(args.output), episodes=len(rows),
                         successful=sum(row['success'] for row in rows)), ensure_ascii=False))


if __name__ == '__main__':
    main()
