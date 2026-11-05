"""Record an actual motor-driven get-up rollout, its metrics and a frame sequence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from stable_baselines3 import PPO
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/robot/cad_getup_stand_round6_support.yaml'))
    parser.add_argument('--checkpoint',type=Path,default=Path('outputs/mujoco_runs/cad_getup_stand_round5_dynamic_20261029/best/best_model.zip'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rearward-offset-m',type=float,default=.37)
    parser.add_argument('--seconds',type=float,default=5.)
    parser.add_argument('--seed',type=int,default=95000)
    args=parser.parse_args()
    if not 0<args.seconds<=20:
        parser.error('Require 0 < seconds <= 20')
    torch.set_num_threads(1)
    config=load_config(args.config)
    model_path=Path(config['experiment']['output_directory'])/'robot_experiment.xml'
    env=make_mujoco_env(model_path,args.config)
    poses=[p for p in env.pose_bank if abs(p['rearward_offset_m']-args.rearward_offset_m)<1e-8]
    if not poses:
        parser.error('Requested offset must match the prepared pose bank')
    env.pose_bank=poses
    policy=PPO.load(args.checkpoint,device='cpu')
    renderer=mujoco.Renderer(env.model,height=360,width=640)
    camera=mujoco.MjvCamera();camera.distance=1.;camera.azimuth=90;camera.elevation=-8
    camera.lookat[:]=[-.06,0,.17]
    option=mujoco.MjvOption();option.geomgroup[1]=1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-pix_fmt','rgb24','-s','640x360','-r','50','-i','-','-an','-c:v','libx264',
        '-preset','fast','-crf','20','-pix_fmt','yuv420p',str(args.output)],stdin=subprocess.PIPE)
    trace=[];frames=[];capture_times=[0.,.2,.4,.6,1.02,2.]
    try:
        observation,reset=env.reset(seed=args.seed,options={'reset_mode':'folded'})
        for step in range(round(args.seconds/env.dt)+1):
            if step%2==0:
                renderer.update_scene(env.data,camera=camera,scene_option=option)
                frame=renderer.render().copy();encoder.stdin.write(frame.tobytes())
                if len(frames)<len(capture_times) and env.steps*env.dt+1e-9>=capture_times[len(frames)]:
                    frames.append((env.steps*env.dt,frame))
            if step*env.dt>=args.seconds:
                break
            action,_=policy.predict(observation,deterministic=True)
            observation,_,done,truncated,info=env.step(action)
            if step%10==0:
                trace.append(dict(seconds=env.steps*env.dt,**{k:info[k] for k in
                    ('tilt_rad','virtual_leg_lengths_m','planar_displacement_m','heading_error_rad',
                     'wheel_contacts','getup_acquired','controller_phase')}))
            if done or truncated:
                break
        encoder.stdin.close()
        if encoder.wait()!=0:
            raise RuntimeError('Video encoder failed')
        canvas=Image.new('RGB',(1280,3*390),'white');draw=ImageDraw.Draw(canvas)
        for n,(seconds,frame) in enumerate(frames):
            x=(n%2)*640;y=(n//2)*390
            draw.text((x+10,y+8),f'{seconds:.2f} s | actual MuJoCo dynamics',fill='black')
            canvas.paste(Image.fromarray(frame),(x,y+30))
        canvas.save(args.output.with_suffix('.jpg'))
        report=dict(simulation_only=True,reset=reset,config=str(args.config),checkpoint=str(args.checkpoint),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
            getup_acquired=env.acquired,getup_seconds=info['getup_seconds'],
            over_rated_seconds=info['over_rated_seconds'],peak_torque_Nm=info['peak_torque_Nm'],trace=trace,
            note='A short visualization is not a complete episode assessment.')
        args.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:report[k] for k in ('getup_acquired','getup_seconds','over_rated_seconds','peak_torque_Nm')}))
    finally:
        if encoder.poll() is None:
            encoder.kill();encoder.wait()
        renderer.close();env.close()


if __name__=='__main__':
    main()
