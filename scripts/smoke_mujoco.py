"""Validate observations, actions, closure, contacts and resets before PPO."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "source"))

import numpy as np
from stable_baselines3.common.env_checker import check_env
from climber_legged_rl.mujoco_env import ClimberMujocoEnv
from climber_legged_rl.mujoco_model import DEFAULT_CONFIG, ROOT, load_config


def run(config_path=DEFAULT_CONFIG, steps=1000):
    config = load_config(config_path)
    folder = ROOT / config["experiment"]["output_directory"]
    env = ClimberMujocoEnv(folder / "robot_experiment.xml",config_path)
    check_env(env,warn=True)
    results = {}
    started = time.monotonic()
    rng = np.random.default_rng(42)
    for mode in ("zero", "random"):
        obs, _ = env.reset(seed=42)
        episodes = []
        maximum_closure = 0.0
        saw_contact = False
        for i in range(steps):
            action = np.zeros(6) if mode == "zero" else rng.uniform(-1,1,6)
            obs,reward,terminated,truncated,info=env.step(action)
            assert obs.shape == env.observation_space.shape and env.observation_space.contains(obs)
            assert np.isfinite(reward)
            maximum_closure = max(maximum_closure,info["closure_error_m"])
            saw_contact |= info["wheel_contacts"] > 0
            assert np.max(np.abs(env.data.ctrl)-env.torque_limits) < 1e-8
            if terminated or truncated:
                episodes.append(info["episode_metrics"])
                obs,_=env.reset(seed=42+len(episodes))
        results[mode]={"steps":steps,"max_closure_error_m":maximum_closure,
                       "saw_wheel_contact":saw_contact,"episodes":episodes,
                       "mean_episode_seconds":float(np.mean([e["seconds"] for e in episodes])) if episodes else None}
        if not saw_contact:
            raise RuntimeError("No wheel-ground contact seen; do not train a falling/floating model")
    env.close()
    report={"simulation_only":True,"passed":True,"elapsed_seconds":time.monotonic()-started,"results":results,
            "config_sha256":hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            "model_sha256":hashlib.sha256((folder/"robot_experiment.xml").read_bytes()).hexdigest(),
            "env_sha256":hashlib.sha256((ROOT/"source/climber_legged_rl/mujoco_env.py").read_bytes()).hexdigest()}
    if config['experiment'].get('model_route')=='cad_dynamics':
        from climber_legged_rl.cad_dynamics import input_hashes
        report['input_sha256']=input_hashes(config_path)
    (folder/"smoke_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return report

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(DEFAULT_CONFIG))
    parser.add_argument("--steps",type=int,default=1000)
    args=parser.parse_args()
    run(args.config,args.steps)
