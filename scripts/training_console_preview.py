"""Record the actual current 208 mm policy; never use a generic robot demo."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

os.environ['MUJOCO_GL'] = 'egl'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'scripts')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['standing', 'folded'], default='standing')
    parser.add_argument('--force', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=840000)
    parser.add_argument('--seconds', type=float, default=20.)
    args = parser.parse_args()
    folder = (ROOT/'outputs'/args.run).resolve()
    if folder.parent != (ROOT/'outputs').resolve() or not (folder/'workflow.json').is_file():
        raise ValueError('Unknown run')
    if not 0 <= args.force <= 150 or not 1 <= args.seconds <= 20:
        raise ValueError('Preview parameters out of range')
    output = args.output.resolve()
    if output.parent != (ROOT/'outputs/training_console_previews').resolve():
        raise ValueError('Invalid preview output')
    output.parent.mkdir(exist_ok=True)
    source = folder/'selected_policy.zip'
    import imageio_ffmpeg
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from stable_baselines3 import PPO
    from climber_legged_rl.cad_dynamics import validate_artifacts
    from climber_legged_rl.mujoco_env import make_mujoco_env
    from climber_legged_rl.mujoco_model import load_config
    from climber_legged_rl.motor_authority import apply_standing_motor_limit
    torch.set_num_threads(1)
    cfg = ROOT/'configs/robot/cad_spring208_phase3_nominal.yaml'
    validate_artifacts(cfg)
    config = load_config(cfg)
    xml = ROOT/config['experiment']['output_directory']/'robot_experiment.xml'
    with tempfile.TemporaryDirectory(prefix='climber_preview_') as temp:
        snapshot = Path(temp)/'policy.zip'
        for attempt in range(5):
            try:
                before = digest(source)
                shutil.copy2(source, snapshot)
                if digest(snapshot) != before or digest(source) != before:
                    raise RuntimeError('Checkpoint changed during copy')
                with zipfile.ZipFile(snapshot) as archive:
                    if archive.testzip() is not None:
                        raise RuntimeError('Incomplete checkpoint')
                break
            except (OSError, RuntimeError, zipfile.BadZipFile):
                if attempt == 4:
                    raise
                time.sleep(.25)
        policy = PPO.load(snapshot, device='cpu')
        env = make_mujoco_env(xml, cfg)
        protocol_path = folder/'protocol.json'
        protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
        motor_envelope = apply_standing_motor_limit(env, protocol.get('standing_leg_torque_limit_Nm'),
                                                   protocol.get('standing_wheel_kv'))
        env.unwrapped.push.config = dict(env.unwrapped.push.config, episode_probability=float(args.force > 0),
                                        force_N=[args.force, args.force] if args.force else [2., 15.])
        env.unwrapped.push.config.pop('force_bands', None)
        renderer = None
        encoder = None
        frames = 0
        info = {}
        ended = False
        trace = []
        try:
            obs, _ = env.reset(seed=args.seed, options={'reset_mode': args.mode})
            renderer = mujoco.Renderer(env.model, height=360, width=640)
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 1.25, 125, -18
            option = mujoco.MjvOption()
            option.geomgroup[1] = 1
            encoder = imageio_ffmpeg.write_frames(str(output), (640, 360), fps=25, codec='libx264',
                        pix_fmt_in='rgb24', pix_fmt_out='yuv420p', quality=7,
                        output_params=['-movflags', '+faststart'])
            encoder.send(None)
            while True:
                if env.steps % 4 == 0 or ended:
                    camera.lookat[:] = env.data.xpos[env.base_id]
                    renderer.update_scene(env.data, camera=camera, scene_option=option)
                    frame = Image.fromarray(renderer.render().copy())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 640, 59), fill='#101b29')
                    force = float(np.linalg.norm(info.get('push_force_world_N', [0., 0., 0.])))
                    draw.text((12, 6), f'CLIMBER | actual policy | {env.steps*env.dt:.2f}s | {args.mode} | {args.force:g}N test', fill='white')
                    draw.text((12, 24), f'Applied now: {force:.1f}N | 208mm stop | seed {args.seed}', fill='#aee8cd')
                    displacement = float(info.get('planar_displacement_m', 0.))
                    tilt = float(np.degrees(info.get('tilt_rad', 0.)))
                    gain = motor_envelope['standing_wheel_kv']
                    draw.text((12, 42), f'Distance: {displacement:.3f}m | Tilt: {tilt:.1f}deg | standing wheel kv: {gain:g}', fill='#ffdf91')
                    encoder.send(np.asarray(frame))
                    if frames == 0:
                        frame.save(output.with_suffix('.jpg'))
                    frames += 1
                if ended or env.steps*env.dt >= args.seconds:
                    break
                obs, _, done, truncated, info = env.step(policy.predict(obs, deterministic=True)[0])
                ended = done or truncated
                if env.steps % 10 == 0 or ended:
                    trace.append(dict(seconds=env.steps*env.dt,
                                      tilt_deg=float(np.degrees(info['tilt_rad'])),
                                      position_m=info['planar_displacement_m'],
                                      force_N=np.asarray(info['push_force_world_N']).tolist()))
            report = dict(run=args.run, checkpoint=str(source), checkpoint_sha256=before,
                          model_sha256=digest(xml), config=str(cfg), seed=args.seed, reset_mode=args.mode,
                          force_N=args.force, requested_seconds=args.seconds, seconds=env.steps*env.dt,
                          episode_complete=bool(ended), success=bool(info.get('combined_success')) if ended else None,
                          reason=info.get('termination_reason') or ('timeout' if ended else 'preview_limit'),
                          motor_envelope=motor_envelope,
                          peak_torque_Nm=info.get('peak_torque_Nm'), push_metrics=info.get('push_metrics'),
                          numerical_warnings=int(sum(env.data.warning.number)), simulation_only=True,
                          created_unix=time.time(), trace=trace, frames=frames)
            output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps({key: value for key, value in report.items() if key != 'trace'}), flush=True)
        finally:
            if encoder:
                encoder.close()
            if renderer:
                renderer.close()
            env.close()


if __name__ == '__main__':
    main()
