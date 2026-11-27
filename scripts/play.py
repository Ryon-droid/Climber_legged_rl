"""Evaluate, record or interactively view a saved MuJoCo policy."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "source"))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument('--pose-only',action='store_true',help='Preview the nominal pose without policy or dynamics')
    parser.add_argument('--config',type=Path,help='Configuration for --pose-only')
    parser.add_argument("--episodes",type=int,default=10)
    parser.add_argument("--seed",type=int,default=20000)
    parser.add_argument("--horizon-seconds",type=float)
    parser.add_argument("--video",type=Path)
    parser.add_argument("--report",type=Path)
    parser.add_argument('--viewer',action='store_true',help='Open the interactive MuJoCo viewer')
    parser.add_argument('--loop',action='store_true',help='Repeat episodes until the viewer is closed')
    parser.add_argument('--command-forward',type=float,help='Fixed forward command for velocity-task viewer, m/s')
    parser.add_argument('--command-yaw',type=float,help='Fixed yaw command for velocity-task viewer, rad/s')
    parser.add_argument('--stop-after-seconds',type=float,help='Return a fixed movement command to zero at this time')
    args=parser.parse_args()
    if args.pose_only:
        if not args.config or args.checkpoint or args.video or args.report:
            parser.error('--pose-only requires --config and cannot use --checkpoint, --video or --report')
        os.environ.setdefault('MUJOCO_GL','glfw')
        import mujoco.viewer
        from climber_legged_rl.mujoco_env import make_mujoco_env as ClimberMujocoEnv
        from climber_legged_rl.mujoco_model import ROOT,load_config
        config=load_config(args.config)
        env=ClimberMujocoEnv(ROOT/config['experiment']['output_directory']/'robot_experiment.xml',args.config)
        try:
            mujoco.mj_resetDataKeyframe(env.model,env.data,0)
            mujoco.mj_forward(env.model,env.data)
            print('POSE PREVIEW ONLY: no policy, no dynamics. Close window to exit.',flush=True)
            with mujoco.viewer.launch_passive(env.model,env.data) as viewer:
                viewer.cam.lookat[:]=env.data.xpos[env.base_id]
                viewer.cam.distance=1.
                viewer.cam.azimuth=90
                viewer.cam.elevation=-10
                viewer.opt.geomgroup[1]=int(env.cad_dynamics)
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(.02)
        finally:
            env.close()
        return
    if args.checkpoint is None:
        parser.error('--checkpoint is required for policy playback')
    if args.config:
        parser.error('--config is only for --pose-only; policy playback uses its saved config')
    if args.episodes<1:
        parser.error("--episodes must be positive")
    if args.horizon_seconds is not None and not 0 < args.horizon_seconds <= 120:
        parser.error('--horizon-seconds must be in (0,120]')
    if args.loop and not args.viewer:
        parser.error('--loop requires --viewer')
    if args.viewer and args.video:
        parser.error('Choose --viewer or --video')
    fixed_command=args.command_forward is not None or args.command_yaw is not None
    if (fixed_command or args.stop_after_seconds is not None) and not (args.viewer or args.video):
        parser.error('Command overrides require --viewer or --video')
    if args.stop_after_seconds is not None and not fixed_command:
        parser.error('--stop-after-seconds requires a fixed movement command')
    os.environ.setdefault("MUJOCO_GL",'glfw' if args.viewer else 'egl')
    from climber_legged_rl.mujoco_training import evaluate
    directory=args.checkpoint.resolve().parent
    if directory.name in ("checkpoints","best"):
        directory=directory.parent
    config_path=directory/"config.yaml"
    model_path=directory/"robot_experiment.xml"
    if args.viewer:
        import mujoco.viewer
        import numpy as np
        from stable_baselines3 import PPO
        from climber_legged_rl.mujoco_env import make_mujoco_env as ClimberMujocoEnv
        env=ClimberMujocoEnv(model_path,config_path)
        velocity_task=env.config['task'].get('velocity_tracking',False)
        if fixed_command and not velocity_task:
            env.close()
            parser.error('Fixed movement commands require a velocity-task checkpoint')
        reset_options={'velocity_command':[args.command_forward or 0.,args.command_yaw or 0.]} if fixed_command else None
        if args.stop_after_seconds is not None:
            reset_options['stop_after_seconds']=args.stop_after_seconds
        if args.horizon_seconds is not None:
            env.max_steps=int(np.ceil(args.horizon_seconds/env.dt))
        policy=PPO.load(str(args.checkpoint),device='cpu')
        print('VIEWER_CHECKPOINT='+str(args.checkpoint.resolve()),flush=True)
        print('Close the MuJoCo window to stop playback.',flush=True)
        try:
            with mujoco.viewer.launch_passive(env.model,env.data) as viewer:
                viewer.cam.lookat[:]=env.data.xpos[env.base_id]
                viewer.cam.distance=1.25
                viewer.cam.azimuth=135
                viewer.cam.elevation=-15
                viewer.opt.geomgroup[1]=int(env.cad_dynamics)
                episode=0
                while viewer.is_running() and (args.loop or episode<args.episodes):
                    obs,_=env.reset(seed=args.seed+episode,options=reset_options)
                    print('PLAYBACK_EPISODE='+str(episode+1),flush=True)
                    deadline=time.monotonic()
                    last_command=None
                    while viewer.is_running():
                        if velocity_task:
                            command=tuple(np.round(env.commands,2))
                            if command!=last_command and env.steps%50==0:
                                print('COMMAND forward=%.2f m/s yaw=%.2f rad/s'%command,flush=True)
                                last_command=command
                        action,_=policy.predict(obs,deterministic=True)
                        obs,_,done,truncated,_=env.step(action)
                        if velocity_task:
                            viewer.cam.lookat[:]=env.data.xpos[env.base_id]
                        viewer.sync()
                        deadline+=env.dt
                        time.sleep(max(0.,deadline-time.monotonic()))
                        if done or truncated:
                            break
                    episode+=1
        finally:
            env.close()
        return
    report=evaluate(args.checkpoint,config_path,model_path,args.episodes,args.seed,horizon_seconds=args.horizon_seconds)
    print(json.dumps(report,indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2),encoding="utf-8")
    if args.video:
        import imageio.v2 as imageio
        from stable_baselines3 import PPO
        from climber_legged_rl.mujoco_env import make_mujoco_env as ClimberMujocoEnv
        env=ClimberMujocoEnv(model_path,config_path,render_mode="rgb_array")
        if args.horizon_seconds is not None:
            import numpy as np
            env.max_steps=int(np.ceil(args.horizon_seconds/env.dt))
        policy=PPO.load(str(args.checkpoint),device="cpu")
        reset_options={'velocity_command':[args.command_forward or 0.,args.command_yaw or 0.]} if fixed_command else None
        if args.stop_after_seconds is not None:
            reset_options['stop_after_seconds']=args.stop_after_seconds
        args.video.parent.mkdir(parents=True,exist_ok=True)
        try:
            with imageio.get_writer(str(args.video),fps=50) as writer:
                for episode in range(min(args.episodes,3)):
                    obs,_=env.reset(seed=args.seed+episode,options=reset_options)
                    writer.append_data(env.render())
                    while True:
                        action,_=policy.predict(obs,deterministic=True)
                        obs,_,done,truncated,_=env.step(action)
                        if env.steps%2==0:
                            writer.append_data(env.render())
                        if done or truncated:
                            break
        finally:
            env.close()


if __name__ == "__main__":
    main()
