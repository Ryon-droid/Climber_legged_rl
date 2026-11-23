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
    from climber_legged_rl.mujoco_env import make_mujoco_env as ClimberMujocoEnv
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
    from climber_legged_rl.velocity_estimation import VelocityEstimatedPolicy
    estimates_velocity = isinstance(policy, VelocityEstimatedPolicy)
    training_observation_dim=policy.observation_space.shape[0]
    observation_dim=(policy.mlp_extractor.policy_net.actor_dimensions if estimates_velocity
                     else training_observation_dim)
    traced=torch.jit.trace(actor,torch.zeros((1,observation_dim),dtype=torch.float32))
    rng=np.random.default_rng(2026)
    observations=rng.uniform(-10,10,(128,training_observation_dim)).astype(np.float32)
    expected,_=policy.predict(observations,deterministic=True)
    with torch.no_grad():
        actual=traced(torch.from_numpy(observations[:, :observation_dim].copy())).numpy()
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
                actual=traced(torch.from_numpy(obs[:observation_dim].copy()).reshape(1,observation_dim)).numpy()[0]
            trajectory_error=max(trajectory_error,float(np.abs(expected-actual).max()))
            obs,_,done,truncated,_=env.step(actual)
            if done or truncated:
                obs,_=env.reset(seed=15000)
        if trajectory_error>1e-5:
            raise RuntimeError("Trajectory export parity failed")
        contract={
            "simulation_only":True,"hardware_deployment_permitted":False,
            "format":"TorchScript","dtype":"float32","input_shape":[1,observation_dim],"output_shape":[1,6],
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
            "wheel_action_sign":list(config["control"].get("wheel_action_sign",[1.0,-1.0])),
            "control_frequency_hz":1.0/env.dt,
            "model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "random_observation_max_abs_error":max_error,
            "trajectory_max_abs_error":trajectory_error,
            "warning":"Not ONNX/CubeAI conversion. Hardware axes/encoders, spring, limits and inertias are uncalibrated."}
        if env.stationary_hold:
            contract['observations_in_order'].extend([
                {"name":"reset_position_error_in_body_xy","size":2,"scale":5.0,"unit":"m"},
                {"name":"heading_error_sin_cos_minus_one","size":2,"scale":1.0},
                {"name":"body_planar_velocity","size":2,"scale":2.0,"unit":"m/s"}])
        if config.get('getup', {}).get('enabled'):
            contract['observations_in_order'].extend([
                {"name":"scheduled_leg_reference_minus_standing","size":4,"scale":1.0,"unit":"rad"},
                {"name":"hip_height_error","size":1,"scale":5.0,"unit":"m"},
                {"name":"rise_phase","size":1,"scale":1.0}])
            contract['getup_reference_schedule']=config['getup']
            contract['action_reference']='scheduled folded-to-standing leg reference; reproduce smoothstep schedule'
            contract['folded_pose_bank']=env.pose_report
            if config['getup'].get('policy_control', {}).get('enabled'):
                contract['action_reference'] = 'learned incremental target while rising; nominal residual while standing'
        from climber_legged_rl.observation_history import history_frames
        contract['observation_history'] = dict(frames=history_frames(config),
            frame_dimensions=sum(block['size'] for block in contract['observations_in_order']),
            order='oldest_to_newest', reset_fill='repeat_first_frame',
            update='once per policy step; append after physics and phase update')
        contract['robustness'] = config.get('robustness', {})
        if estimates_velocity:
            contract['training_input_dimensions'] = training_observation_dim
            contract['velocity_estimator'] = dict(input_dimensions=observation_dim,
                output_dimensions=3, output_unit='m/s', output_frame='chassis_body',
                sensor_indices=policy.mlp_extractor.policy_net.sensor_indices.tolist(),
                actor_velocity_slots='Every frame 28:30 is ignored; estimated vx/vy inserted in latest frame',
                privileged_velocity_labels='Three appended labels exist only during training; excluded from export',
                remaining_state_requirements=['reset-relative planar position', 'heading', 'hip height', 'phase'])
            contract['observations_in_order'][8]['name'] = 'ignored_velocity_slots_fill_zero'
        if history_frames(config) > 1 or config.get('robustness'):
            contract['observations_in_order'][5]['name'] = 'previous_clipped_policy_command'
        if config['training'].get('history_standing_actor'):
            contract['phase_split'] = dict(threshold=.95, phase_index_in_latest_frame=35,
                rising_reads_latest_frame_only=True, standing_reads_all_frames=True)
    finally:
        env.close()
    args.output.mkdir(parents=True,exist_ok=True)
    traced.save(str(args.output/"actor_simulation_only.ts"))
    if estimates_velocity:
        class Estimator(torch.nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.register_buffer('indices', layers.sensor_indices.clone())
                self.network = layers.velocity_estimator

            def forward(self, obs):
                return self.network(obs.index_select(1, self.indices))

        estimator=Estimator(policy.mlp_extractor.policy_net).eval()
        traced_estimator=torch.jit.trace(estimator, torch.zeros(1, observation_dim))
        with torch.no_grad():
            sample=torch.from_numpy(observations[:, :observation_dim].copy())
            error=float((traced_estimator(sample)-policy.mlp_extractor.policy_net.estimate_velocity(sample)).abs().max())
        if error > 1e-5:
            raise RuntimeError('Velocity estimator export parity failed')
        contract['velocity_estimator']['export_max_abs_error_m_s'] = error
        traced_estimator.save(str(args.output/'velocity_estimator.ts'))
    (args.output/"interface.json").write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(contract,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
