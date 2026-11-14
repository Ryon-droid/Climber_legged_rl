"""Verified replay of held-out failures; rendering never changes training or physics."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['MUJOCO_GL'] = 'egl'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'source'), str(ROOT / 'scripts')]
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from stable_baselines3 import PPO
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.push_balance_curriculum import PushBalanceCurriculum
from climber_legged_rl.motor_authority import apply_standing_motor_limit

CASES = [
    (1230204, '第一次推扰后触地', 'first_push'),
    (1230208, '第二次推扰后触地', 'second_push'),
    (1230205, '短暂恢复后再次失稳', 'late_fall'),
]
W, H, FPS = 1280, 720, 25
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoder(path):
    stream = imageio_ffmpeg.write_frames(str(path), (W, H), fps=FPS, codec='libx264',
        pix_fmt_in='rgb24', pix_fmt_out='yuv420p', quality=7,
        output_params=['-movflags', '+faststart'])
    stream.send(None)
    return stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    folder = args.folder.resolve()
    report = json.loads((folder / 'selected_test.json').read_text())
    checkpoint = Path(report['checkpoint'])
    if digest(checkpoint) != report['sha256']:
        raise RuntimeError('Checkpoint differs from tested policy')
    out = folder / 'failure_replays'
    out.mkdir(exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('Refusing to overwrite verified replays')
    torch.set_num_threads(1)
    policy = PPO.load(checkpoint, device='cpu')
    fonts = [ImageFont.truetype(FONT, s) for s in (27, 20, 17)]
    combined = encoder(out / 'failures_3_cases.mp4')
    results, contacts = [], []
    try:
        for index, (seed, title, name) in enumerate(CASES):
            row = next(e for e in report['episodes']
                       if e['spec']['group'] == 'exact18' and e['spec']['seed'] == seed)
            if row['success'] or row['reason'] != 'base_contact':
                raise RuntimeError('Requested case is not a tested contact failure')
            spec = row['spec']
            cfg = Path(spec['config'])
            validate_artifacts(cfg)
            config = load_config(cfg)
            model_path = ROOT / config['experiment']['output_directory'] / 'robot_experiment.xml'
            env = make_mujoco_env(model_path, cfg)
            base = env.unwrapped
            apply_standing_motor_limit(env, spec.get('standing_leg_torque_limit_Nm'))
            base.push.config = dict(base.push.config, episode_probability=float(spec['push']),
                                    force_N=spec['force_N'])
            base.push.config.pop('force_bands', None)
            env = PushBalanceCurriculum(env, spec['balance_curriculum_stage'])
            render = mujoco.Renderer(base.model, height=360, width=640)
            option = mujoco.MjvOption()
            option.geomgroup[1] = 1
            cameras = []
            for azimuth in (90, 0):
                camera = mujoco.MjvCamera()
                camera.distance, camera.azimuth, camera.elevation = 1.25, azimuth, -12
                cameras.append(camera)
            stream = encoder(out / f'{index + 1}_{name}_{seed}.mp4')
            frames, trace, info = 0, [], {}
            final_image = None
            max_position = max_heading = max_tilt = 0.

            def frame(mode):
                canvas = Image.new('RGB', (W, H), '#101b29')
                draw = ImageDraw.Draw(canvas)
                now = base.steps * base.dt
                draw.text((16, 8), f'失败回放 {index + 1}/3：{title}', font=fonts[0], fill='white')
                draw.text((16, 47), f'实际测试模型 | 18 N | seed {seed} | 仿真 {now:.2f} s | {mode}',
                          font=fonts[1], fill='#9de0bf')
                force = np.array(info.get('push_force_world_N', [0., 0., 0.]))
                draw.text((16, 76), '当前施力 ' + f'{np.linalg.norm(force):.1f} N' +
                          ('  ← 正在推扰' if np.linalg.norm(force) else ''), font=fonts[2],
                          fill='#ff9c7b' if np.linalg.norm(force) else '#b9c7d7')
                if base.push.events:
                    event = base.push.events[-1]
                    f = np.asarray(event['force_world_N'])
                    draw.text((620, 10), f'最近推力 XYZ = ({f[0]:+.1f}, {f[1]:+.1f}, {f[2]:+.1f}) N',
                              font=fonts[1], fill='#ffbd9b')
                    draw.text((620, 43), f"第 {len(base.push.events)} 次推扰 | {event['start_step']*base.dt:.2f} s 开始 | 持续 {event['duration_seconds']:.2f} s",
                              font=fonts[2], fill='#b9c7d7')
                    cx, cy, radius = 1215, 55, 24
                    draw.line((cx-radius, cy, cx+radius, cy), fill='#72869a')
                    draw.line((cx, cy-radius, cx, cy+radius), fill='#72869a')
                    u = f[:2] / max(np.linalg.norm(f[:2]), 1e-9)
                    end = (cx + radius*u[0], cy - radius*u[1])
                    draw.line((cx, cy, *end), fill='#ff8c76', width=4)
                    draw.ellipse((end[0]-3, end[1]-3, end[0]+3, end[1]+3), fill='#ff8c76')
                    draw.text((1170, 83), '+X→ +Y↑', font=fonts[2], fill='#b9c7d7')
                for view, camera in enumerate(cameras):
                    camera.lookat[:] = base.data.xpos[base.base_id]
                    render.update_scene(base.data, camera=camera, scene_option=option)
                    canvas.paste(Image.fromarray(render.render().copy()), (view*640, 110))
                    draw.text((view*640+14, 115), ('侧视（沿 Y 轴）' if view == 0 else '正视（沿 X 轴）'),
                              font=fonts[1], fill='#f3f7fc', stroke_width=1, stroke_fill='#101b29')
                tilt = np.degrees(info.get('tilt_rad', 0.))
                speed = info.get('planar_speed_m_s', 0.)
                displacement = 100 * info.get('planar_displacement_m', 0.)
                draw.text((16, 478), f'倾角 {tilt:.1f}°     速度 {speed:.2f} m/s     位移 {displacement:.1f} cm     双轮接触数 {info.get("wheel_contacts", "—")}',
                          font=fonts[1], fill='#f1d18c')
                torques = info.get('motor_torque_Nm', [0.]*6)
                labels = [(0, '左前腿 8009P'), (1, '左后腿 8009P'), (4, '左轮 3508'),
                          (2, '右前腿 8009P'), (3, '右后腿 8009P'), (5, '右轮 3508')]
                for n, (joint, label) in enumerate(labels):
                    x, y = 16 + n//3*640, 518 + n%3*51
                    value = float(torques[joint])
                    if joint < 4:
                        measure = f'模型关节角 {np.degrees(base.data.qpos[base.q_indices[joint]]):+.1f}°'
                    else:
                        measure = f'轮速 {base.data.qvel[base.v_indices[joint]]:+.1f} rad/s'
                    draw.text((x, y), f'{label}   {value:+.2f} N·m   {measure}', font=fonts[2], fill='#c9d7e6')
                    bx, by = x+2, y+32
                    draw.line((bx, by, bx+520, by), fill='#34475b', width=5)
                    draw.line((bx+260, by-5, bx+260, by+5), fill='#71899f')
                    end = bx+260+250*np.clip(value/base.torque_limits[joint], -1, 1)
                    draw.line((bx+260, by, end, by), fill='#77dcb2' if joint < 4 else '#79b5fb', width=6)
                text = '底盘触地：此处终止回合；未继续模拟倒地。' if info.get('termination_reason') else '最后 2 秒按 0.5 倍速播放；画面时间始终为真实仿真时间。'
                draw.text((16, 683), text, font=fonts[1], fill='#ff9b8b' if info.get('termination_reason') else '#9eb2c7')
                return np.asarray(canvas)

            try:
                obs, _ = env.reset(seed=seed, options={'reset_mode': spec['mode']})
                first = frame('片头定格')
                for _ in range(FPS):
                    stream.send(first); combined.send(first); frames += 1
                while True:
                    action = policy.predict(obs, deterministic=True)[0]
                    obs, _, done, truncated, info = env.step(action)
                    now = base.steps * base.dt
                    max_position = max(max_position, info['planar_displacement_m'])
                    max_heading = max(max_heading, float(np.degrees(abs(info['heading_error_rad']))))
                    max_tilt = max(max_tilt, float(np.degrees(info['tilt_rad'])))
                    if base.steps % 4 == 0 or done or truncated:
                        slow = now >= row['seconds'] - 2.
                        final_image = frame('终止画面' if done or truncated else '0.5 倍速' if slow else '正常速度')
                        for _ in range(2 if slow else 1):
                            stream.send(final_image); combined.send(final_image); frames += 1
                        trace.append(dict(seconds=now, action=np.asarray(action).tolist(),
                            joint_position_rad=base.data.qpos[base.q_indices].tolist(),
                            joint_velocity_rad_s=base.data.qvel[base.v_indices].tolist(),
                            torque_Nm=list(info['motor_torque_Nm']), tilt_deg=float(np.degrees(info['tilt_rad'])),
                            speed_m_s=info['planar_speed_m_s'], wheel_contacts=info['wheel_contacts'],
                            force_world_N=list(info['push_force_world_N'])))
                    if done or truncated:
                        break
                # Rendering is accepted only if it reproduces the evaluated case.
                actual_max = [max_position, max_heading, max_tilt]
                expected_max = [row['maximum'][k] for k in ('displacement_m', 'heading_deg', 'tilt_deg')]
                if not (np.isclose(now, row['seconds'], atol=1e-9)
                        and info['termination_reason'] == row['reason']
                        and np.allclose(actual_max, expected_max, rtol=1e-9, atol=1e-9)
                        and not info['combined_success']
                        and info['push_metrics'] == row['push_metrics']):
                    raise RuntimeError('Rendered rollout does not match saved independent test')
                for _ in range(round(1.5*FPS)):
                    stream.send(final_image); combined.send(final_image); frames += 1
                Image.fromarray(final_image).save(out / f'{index+1}_{name}_contact.jpg')
                contacts.append(Image.fromarray(final_image).resize((960, 540)))
                metadata = dict(seed=seed, title=title, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                    config=str(cfg), model_sha256=digest(model_path), reason=info['termination_reason'],
                    seconds=now, maximum=copy.deepcopy(row['maximum']), push_metrics=info['push_metrics'],
                    strict_success=info['strict_success'], success=info['combined_success'], frames=frames,
                    exact_test_reproduction=True, numerical_warnings=int(sum(base.data.warning.number)),
                    video=f'{index+1}_{name}_{seed}.mp4', trace=trace)
                (out / f'{index+1}_{name}_{seed}.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
                results.append({k:v for k,v in metadata.items() if k != 'trace'})
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
            finally:
                stream.close(); render.close(); env.close()
    finally:
        combined.close()
    contact_sheet = Image.new('RGB', (960, 1620))
    for i, image in enumerate(contacts):
        contact_sheet.paste(image, (0, i*540))
    contact_sheet.save(out / 'failure_contact_sheet.jpg')
    (out / 'manifest.json').write_text(json.dumps(dict(run=str(folder),
        combined_video='failures_3_cases.mp4', cases=results), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
