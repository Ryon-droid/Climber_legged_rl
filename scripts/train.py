"""Default to actual CAD model audit; legacy experiments require explicit selection."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "source"))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",choices=["cad","cad-repaired","cad-dynamics","mujoco","isaaclab"],default="cad")
    parser.add_argument("--allow-approximate-model",action="store_true")
    parser.add_argument("--config",type=Path,help="Defaults to the selected model route configuration")
    parser.add_argument("--total-timesteps",type=int)
    parser.add_argument("--num-envs",type=int)
    parser.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    parser.add_argument("--run-name")
    parser.add_argument("--resume",type=Path)
    parser.add_argument('--warm-start',type=Path,help='Initialize from a policy with identical physics/control; allow a new task/reward and reset step counter')
    args=parser.parse_args()
    if args.backend=='cad-dynamics':
        from climber_legged_rl.cad_dynamics import DEFAULT_DYNAMICS_CONFIG
        from climber_legged_rl.mujoco_training import train
        train(args.config or DEFAULT_DYNAMICS_CONFIG,args.total_timesteps,args.num_envs,args.device,args.run_name,args.resume,args.warm_start)
        return
    if args.backend=="cad-repaired":
        from climber_legged_rl.cad_repair import build_repaired, DEFAULT_REPAIR_CONFIG
        _, report = build_repaired(args.config or DEFAULT_REPAIR_CONFIG)
        print(__import__("json").dumps(report, ensure_ascii=False, indent=2))
        parser.exit(2, "CAD kinematics repaired; physical mass, spring and actuator inputs remain missing. See docs/CAD修正结果.md.\n")
    if args.backend=="cad":
        from climber_legged_rl.cad_model import audit_model
        from climber_legged_rl.cad_model import DEFAULT_CAD_CONFIG
        report = audit_model(args.config or DEFAULT_CAD_CONFIG)
        print(__import__("json").dumps(report, ensure_ascii=False, indent=2))
        parser.exit(2, "CAD model is not released for training. See outputs/cad_model/audit_report.json and docs/真实模型训练.md.\n")
    if args.backend=="isaaclab":
        parser.error("Isaac Lab task is not implemented; only the explicit MuJoCo experiment is available.")
    if not args.allow_approximate_model:
        parser.error("Read configs/robot/mujoco_experiment.yaml and pass --allow-approximate-model for SIMULATION ONLY.")
    from climber_legged_rl.mujoco_training import train
    config = args.config or Path(__file__).resolve().parents[1]/"configs/robot/mujoco_experiment.yaml"
    train(config,args.total_timesteps,args.num_envs,args.device,args.run_name,args.resume,args.warm_start)


if __name__ == "__main__":
    main()
