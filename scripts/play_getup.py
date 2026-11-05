"""Preview real get-up dynamics with a parent actor and the documented PD reference."""
import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/robot/cad_getup_stand_push.yaml'))
    parser.add_argument('--seed', type=int, default=97000)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--reset-mode', choices=['folded', 'standing', 'mixed', 'curriculum'], default='folded')
    parser.add_argument('--pose-only', action='store_true', help='Freeze a reset pose without policy or dynamics')
    parser.add_argument('--folded-length-m', type=float, help='Select an exact length from the prepared reset bank')
    parser.add_argument('--rearward-offset-m', type=float, help='Select the wheel rearward displacement from the prepared bank')
    parser.add_argument('--push-episode-probability',type=float,
                        help='Override push sampling: 0 for no-push playback, 1 for forced-push episodes')
    args = parser.parse_args()
    if not args.pose_only and args.checkpoint is None:
        parser.error('--checkpoint is required unless --pose-only is selected')
    if args.pose_only and args.reset_mode == 'mixed':
        parser.error('--pose-only requires folded or standing reset-mode')
    os.environ.setdefault('MUJOCO_GL', 'glfw')
    import mujoco.viewer
    from stable_baselines3 import PPO
    from climber_legged_rl.mujoco_env import make_mujoco_env
    from climber_legged_rl.mujoco_model import ROOT, load_config
    config = load_config(args.config)
    env = make_mujoco_env(ROOT / config['experiment']['output_directory'] / 'robot_experiment.xml', args.config)
    if args.push_episode_probability is not None:
        if not env.push.enabled or not 0 <= args.push_episode_probability <= 1:
            env.close()
            parser.error('--push-episode-probability requires enabled pushes and probability in [0,1]')
        env.push.config=dict(env.push.config,episode_probability=args.push_episode_probability)
    if args.rearward_offset_m is not None:
        poses = [p for p in env.pose_bank if abs(p['rearward_offset_m'] - args.rearward_offset_m) < 1e-8]
        if not poses:
            env.close()
            parser.error('--rearward-offset-m must match an existing prepared rearward offset')
        env.pose_bank = poses
    if args.folded_length_m is not None:
        poses = [p for p in env.pose_bank if abs(p['length_m'] - args.folded_length_m) < 1e-8]
        if not poses:
            env.close()
            parser.error('--folded-length-m must match an existing reset length')
        env.pose_bank = poses
    if args.pose_only:
        _, info = env.reset(seed=args.seed, options={'reset_mode': args.reset_mode})
        print('STATIC POSE ONLY: no policy and no physics stepping.', info, flush=True)
        try:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                viewer.cam.lookat[:] = env.data.xpos[env.base_id]
                viewer.cam.distance = 1.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -10
                viewer.opt.geomgroup[1] = 1
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(.02)
        finally:
            env.close()
        return
    policy = PPO.load(args.checkpoint, device='cpu')
    dimensions = policy.observation_space.shape[0]
    expected_dimensions = env.observation_space.shape[0]
    if dimensions != expected_dimensions and not (dimensions == 30 and expected_dimensions == 36):
        raise ValueError('Checkpoint observations do not match this configuration')
    if config.get('getup',{}).get('policy_control',{}).get('enabled'):
        if dimensions != expected_dimensions:
            raise ValueError('Full policy get-up requires the complete configured observation')
        print('FULL POLICY GET-UP: one actor controls all six motors from reset. No reference assistance.',flush=True)
    elif config.get('getup',{}).get('support_control',{}).get('enabled'):
        if dimensions != 36:
            raise ValueError('Reference-assisted handoff requires the validated 36-input combined actor')
        print('REFERENCE-ASSISTED GET-UP: bounded motor-driven reference, then learned standing. No state teleportation.',flush=True)
    elif config.get('getup',{}).get('dynamic_control',{}).get('enabled'):
        print('DYNAMIC GET-UP: learned incremental PD targets, then standing residual; check success metrics.',flush=True)
    else:
        print('GUIDED GET-UP: smooth PD reference + actor; this is not proof of a trained combined policy.', flush=True)
    try:
        options = None if args.reset_mode == 'mixed' else {'reset_mode': args.reset_mode}
        obs, reset_info = env.reset(seed=args.seed, options=options)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.distance = 1.2
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
            viewer.opt.geomgroup[1] = 1
            episode = 0
            while viewer.is_running():
                if episode:
                    with viewer.lock():
                        obs, reset_info = env.reset(seed=args.seed + episode, options=options)
                print('RESET', reset_info, flush=True)
                with viewer.lock():
                    viewer.cam.lookat[:] = env.data.xpos[env.base_id]
                deadline = time.monotonic()
                while viewer.is_running():
                    action, _ = policy.predict(obs[:dimensions], deterministic=True)
                    with viewer.lock():
                        obs, _, done, truncated, info = env.step(action)
                    # Playback owns physics; keep GUI edits from restoring pre-reset state.
                    viewer.sync(state_only=True)
                    deadline += env.dt
                    time.sleep(max(0., deadline - time.monotonic()))
                    if done or truncated:
                        print('RESULT', info['getup_metrics'], info['termination_reason'], flush=True)
                        break
                episode += 1
                if not args.loop:
                    break
    finally:
        env.close()


if __name__ == '__main__':
    main()
