"""Evaluate or render an approximate MuJoCo policy without hardware control."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "source"))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--episodes",type=int,default=10)
    parser.add_argument("--seed",type=int,default=20000)
    parser.add_argument("--horizon-seconds",type=float)
    parser.add_argument("--video",type=Path)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    if args.episodes<1:
        parser.error("--episodes must be positive")
    os.environ.setdefault("MUJOCO_GL","egl")
    from climber_legged_rl.mujoco_training import evaluate
    directory=args.checkpoint.resolve().parent
    if directory.name in ("checkpoints","best"):
        directory=directory.parent
    config_path=directory/"config.yaml"
    model_path=directory/"robot_experiment.xml"
    report=evaluate(args.checkpoint,config_path,model_path,args.episodes,args.seed,horizon_seconds=args.horizon_seconds)
    print(json.dumps(report,indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2),encoding="utf-8")
    if args.video:
        import imageio.v2 as imageio
        from stable_baselines3 import PPO
        from climber_legged_rl.mujoco_env import ClimberMujocoEnv
        env=ClimberMujocoEnv(model_path,config_path,render_mode="rgb_array")
        policy=PPO.load(str(args.checkpoint),device="cpu")
        args.video.parent.mkdir(parents=True,exist_ok=True)
        try:
            with imageio.get_writer(str(args.video),fps=50) as writer:
                for episode in range(min(args.episodes,3)):
                    obs,_=env.reset(seed=args.seed+episode)
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
