"""Render the actual prepared standing/prone reset geometries without stepping physics."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import mujoco
from PIL import Image, ImageDraw, ImageFont
from climber_legged_rl.getup import validate_pose_bank
from climber_legged_rl.mujoco_model import ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/robot/cad_getup_stand_push.yaml'))
    args = parser.parse_args()
    config = load_config(args.config)
    folder = ROOT / config['experiment']['output_directory']
    bank = validate_pose_bank(args.config)['poses']
    model = mujoco.MjModel.from_xml_path(str(folder / 'robot_experiment.xml'))
    data = mujoco.MjData(model)
    pose = max(bank, key=lambda p: p['rearward_offset_m'])
    poses = [('站姿：135 mm', model.key_qpos[0], '原先确认的站立参考'),
             ('趴姿：腹部朝下、双腿向后伸展', pose['qpos'],
              '后展 %.0f mm；气弹簧 %.1f mm（允许155–220）' %
              (pose['rearward_offset_m']*1000, pose['spring_eye_distances_m']['left']*1000))]
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 16)
    canvas = Image.new('RGB', (1200, 910), '#f4f6f8')
    draw = ImageDraw.Draw(canvas)
    options = mujoco.MjvOption(); options.geomgroup[1] = 1
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.055, 0, .13]; camera.distance = 1.03; camera.elevation = -8
    renderer = mujoco.Renderer(model, height=360, width=600)
    try:
        for row, (label, qpos, note) in enumerate(poses):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            for column, azimuth in enumerate((90, 135)):
                x, y = column*600, row*440
                draw.text((x+12, y+8), label, font=font, fill='#17213a')
                camera.azimuth = azimuth
                renderer.update_scene(data, camera=camera, scene_option=options)
                canvas.paste(Image.fromarray(renderer.render()), (x, y+40))
                draw.text((x+12, y+405), note, font=small, fill='#27354a')
        draw.text((12, 882), '侧视：图左为模型后方。后展长度和关节范围为仿真配置，尚非实物角度标定；本图为静态初态。', font=small, fill='#27354a')
        path = folder / 'rear_extended_prone_comparison.png'
        canvas.save(path)
        print(path)
    finally:
        renderer.close()


if __name__ == '__main__':
    main()
