"""Preview or record the independently checked 208 mm policy using native observations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.record:
        # MuJoCo selects its OpenGL backend when imported by artifact checking.
        os.environ['MUJOCO_GL'] = 'egl'
    result_path = ROOT/'outputs/spring208_adaptation/independent_assessment.json'
    report = json.loads(result_path.read_text())
    cp = Path(report['candidate'])
    cfg = ROOT/'configs/robot/cad_balance_spring208.yaml'
    xml = ROOT/'outputs/cad_balance_spring208/robot_experiment.xml'
    if not report['passed_nominal_adaptation'] or digest(cp) != report['candidate_sha256'] or digest(xml) != report['model_sha256']:
        raise RuntimeError('No matching independently verified candidate')
    from climber_legged_rl.cad_dynamics import validate_artifacts
    validate_artifacts(cfg)
    if args.check_only:
        print(json.dumps(dict(verified=True, checkpoint=str(cp), assessment=str(result_path))))
        return
    if not args.record:
        subprocess.run([sys.executable, str(ROOT/'scripts/play_getup.py'), '--config', str(cfg),
            '--checkpoint', str(cp), '--reset-mode', 'folded', '--push-episode-probability', '0', '--loop'],
            cwd=ROOT, check=True)
        return
    import imageio_ffmpeg
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from stable_baselines3 import PPO
    from climber_legged_rl.mujoco_env import make_mujoco_env
    torch.set_num_threads(1)
    env = make_mujoco_env(xml, cfg)
    env.unwrapped.push.config = dict(env.push.config, episode_probability=0.)
    policy = PPO.load(cp, device='cpu')
    output = ROOT/'outputs/spring208_adaptation/Climber_208mm_getup.mp4'
    renderer = mujoco.Renderer(env.model, height=360, width=640)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.15, 125, -15
    camera.lookat[:] = [-.10, 0, .16]
    option = mujoco.MjvOption()
    option.geomgroup[1] = 1
    encoder = imageio_ffmpeg.write_frames(str(output), (640, 360), fps=25,
        codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', quality=7,
        output_params=['-movflags', '+faststart'])
    encoder.send(None)
    panels = []
    captures = [0., .4, .8, 1.2, 3.64, 6.]
    obs, _ = env.reset(seed=355000, options={'reset_mode': 'folded'})
    try:
        while True:
            if env.steps % 4 == 0:
                renderer.update_scene(env.data, camera=camera, scene_option=option)
                frame = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 25), fill='white')
                draw.text((10, 7), f'208 mm stop | {env.steps*env.dt:.2f} s | '
                    + ('Standing' if env.acquired else 'Getting up'), fill='black')
                encoder.send(np.asarray(frame))
                if len(panels) < len(captures) and env.steps*env.dt+1e-9 >= captures[len(panels)]:
                    panels.append(frame.copy())
            action = policy.predict(obs, deterministic=True)[0]
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                if not info['combined_success']:
                    raise RuntimeError('Video rollout failed the complete task criteria')
                break
    finally:
        encoder.close()
        renderer.close()
        env.close()
    sheet = Image.new('RGB', (1280, 1080), 'white')
    for i, frame in enumerate(panels):
        sheet.paste(frame, ((i % 2)*640, (i//2)*360))
    sheet.save(output.with_suffix('.jpg'))
    video_report = dict(simulation_only=True, actual_policy=True, runtime_teacher_assistance=False,
        checkpoint=str(cp), checkpoint_sha256=digest(cp), model_sha256=digest(xml),
        seed=355000, push_episode_probability=0., seconds=env.steps*env.dt,
        success=bool(info['combined_success']), getup_seconds=info['getup_seconds'],
        peak_torque_Nm=info['peak_torque_Nm'], over_rated_seconds=info['over_rated_seconds'])
    output.with_suffix('.json').write_text(json.dumps(video_report, indent=2)+'\n')
    print(json.dumps(video_report), flush=True)


if __name__ == '__main__':
    main()
