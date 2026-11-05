"""Render an actual difficult standing reset and the learned-policy failure trajectory."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from stable_baselines3 import PPO
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=99905)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.config)
    env = make_mujoco_env(ROOT / config['experiment']['output_directory'] / 'robot_experiment.xml', args.config)
    policy = PPO.load(args.checkpoint, device='cpu')
    renderer = mujoco.Renderer(env.model, height=360, width=640)
    camera = mujoco.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -16
    camera.distance = .95
    camera.lookat[:] = [-.06, 0., .15]
    options = mujoco.MjvOption()
    options.geomgroup[1] = 1
    frames = []

    def capture(title, note):
        camera.lookat[:] = env.data.xpos[env.base_id]
        renderer.update_scene(env.data, camera=camera, scene_option=options)
        frames.append((title, note, Image.fromarray(renderer.render().copy())))

    try:
        mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
        mujoco.mj_forward(env.model, env.data)
        capture('标准站姿参考', '135 mm轮腿；本图为静态参考')
        env.push.enabled = False
        obs, _ = env.reset(seed=args.seed, options={'reset_mode': 'standing'})
        pitch, roll = env._attitude()
        velocity = env.data.qvel[:6].copy()
        capture('困难初态：0.00秒', 'seed=%d；横滚%.3f°，俯仰%.3f°' % (args.seed, np.degrees(roll), np.degrees(pitch)))
        trace = []
        captured_midway = False
        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            seconds = env.steps * env.dt
            if env.steps % 10 == 0 or done or truncated:
                trace.append({k: info[k] for k in ('planar_displacement_m', 'heading_error_rad', 'tilt_rad', 'wheel_contacts')} | {'seconds': seconds})
            if not captured_midway and seconds >= 2.:
                capture('同一回合：2.00秒', '偏移%.1f mm；朝向误差%.1f°' % (1000*info['planar_displacement_m'], np.degrees(info['heading_error_rad'])))
                captured_midway = True
            if done or truncated:
                capture('同一回合：%.2f秒' % seconds,
                    '%s；偏移%.0f mm，朝向误差%.1f°' % (info['termination_reason'] or 'timeout', 1000*info['planar_displacement_m'], np.degrees(info['heading_error_rad'])))
                break
        canvas = Image.new('RGB', (1280, 860), '#f4f6f8')
        draw = ImageDraw.Draw(canvas)
        font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        title_font = ImageFont.truetype(font_path, 23)
        note_font = ImageFont.truetype(font_path, 17)
        for index, (title, note, frame) in enumerate(frames[:4]):
            x, y = index % 2 * 640, index // 2 * 430
            draw.text((x+12, y+8), title, font=title_font, fill='#182237')
            canvas.paste(frame, (x, y+40))
            draw.text((x+12, y+403), note, font=note_font, fill='#27354a')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(args.output)
        report = dict(simulation_only=True, config=str(args.config), checkpoint=str(args.checkpoint),
            seed=args.seed, pushes_enabled=False, initial_pitch_deg=float(np.degrees(pitch)),
            initial_roll_deg=float(np.degrees(roll)), initial_free_joint_qvel=velocity.tolist(),
            termination=info['termination_reason'], seconds=seconds, trace=trace)
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != 'trace'}))
        print(args.output)
    finally:
        renderer.close()
        env.close()


if __name__ == '__main__':
    main()
