"""Export a simulation-only deterministic actor and verify numerical parity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "source"))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from climber_legged_rl.mujoco_env import ClimberMujocoEnv
    from climber_legged_rl.mujoco_model import load_config
    directory=args.checkpoint.resolve().parent
    if directory.name in ("checkpoints","best"):
        directory=directory.parent
    config_path=directory/"config.yaml"
    model_path=directory/"robot_experiment.xml"
    config=load_config(config_path)
    policy=PPO.load(str(args.checkpoint),device="cpu").policy
    policy.set_training_mode(False)

    class Actor(torch.nn.Module):
        def __init__(self,policy):
            super().__init__()
            self.layers=policy.mlp_extractor.policy_net
            self.output=policy.action_net

        def forward(self,obs):
            return torch.clamp(self.output(self.layers(obs)),-1.0,1.0)

    actor=Actor(policy).eval()
    traced=torch.jit.trace(actor,torch.zeros((1,24),dtype=torch.float32))
    rng=np.random.default_rng(2026)
    observations=rng.uniform(-10,10,(128,24)).astype(np.float32)
    expected,_=policy.predict(observations,deterministic=True)
    with torch.no_grad():
        actual=traced(torch.from_numpy(observations)).numpy()
    max_error=float(np.abs(expected-actual).max())
    if max_error>1e-5:
        raise RuntimeError("Export parity failed: "+str(max_error))
    env=ClimberMujocoEnv(model_path,config_path)
    try:
        obs,_=env.reset(seed=15000)
        trajectory_error=0.0
        for _ in range(100):
            expected,_=policy.predict(obs,deterministic=True)
            with torch.no_grad():
                actual=traced(torch.from_numpy(obs).reshape(1,24)).numpy()[0]
            trajectory_error=max(trajectory_error,float(np.abs(expected-actual).max()))
            obs,_,done,truncated,_=env.step(actual)
            if done or truncated:
                obs,_=env.reset(seed=15000)
        if trajectory_error>1e-5:
            raise RuntimeError("Trajectory export parity failed")
        contract={
            "simulation_only":True,"hardware_deployment_permitted":False,
            "format":"TorchScript","dtype":"float32","input_shape":[1,24],"output_shape":[1,6],
            "observation_clip":[-10,10],
            "observations_in_order":[
                {"name":"base_ang_vel","size":3,"scale":0.25,"unit":"rad/s"},
                {"name":"projected_gravity","size":3,"scale":1.0},
                {"name":"commands_forward_yaw","size":2,"scales":[2.0,0.25]},
                {"name":"leg_q_minus_numerical_reset","size":4,"scale":1.0},
                {"name":"active_joint_velocities","size":6,"scale":0.05},
                {"name":"previous_clipped_policy_action","size":6,"scale":1.0}],
            "action_order":config["control"]["leg_joints"]+config["control"]["wheel_joints"],
            "output_clipped":[-1,1],"nominal_leg_q_rad":env.nominal_legs.tolist(),
            "leg_action_scale_rad":config["control"]["leg_action_scale_rad"],
            "wheel_action_scale_rad_s":config["control"]["wheel_action_scale_rad_s"],
            "control_frequency_hz":1.0/env.dt,
            "model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "random_observation_max_abs_error":max_error,
            "trajectory_max_abs_error":trajectory_error,
            "warning":"Not ONNX/CubeAI conversion. Hardware axes/encoders, spring, limits and inertias are uncalibrated."}
    finally:
        env.close()
    args.output.mkdir(parents=True,exist_ok=True)
    traced.save(str(args.output/"actor_simulation_only.ts"))
    (args.output/"interface.json").write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(contract,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
