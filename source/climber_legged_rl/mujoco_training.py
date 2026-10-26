"""Bounded PPO experiment runner, evaluation and portable artifacts."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecCheckNan

from .mujoco_env import ClimberMujocoEnv
from .mujoco_model import ROOT, load_config


def _factory(model_path,config_path,seed,monitor_path):
    def make():
        env=ClimberMujocoEnv(model_path,config_path)
        env.reset(seed=seed)
        return Monitor(env,filename=monitor_path)
    return make


def evaluate(checkpoint,config_path,model_path,episodes=10,seed=10000,random_actions=False,horizon_seconds=None):
    policy=PPO.load(str(checkpoint),device="cpu") if checkpoint else None
    env=ClimberMujocoEnv(model_path,config_path)
    if horizon_seconds is not None:
        if not 0 < horizon_seconds <= 120:
            raise ValueError("Evaluation horizon must be in (0,120] seconds")
        env.max_steps=int(np.ceil(horizon_seconds/env.dt))
    rng=np.random.default_rng(seed)
    records=[]
    for episode in range(episodes):
        obs,_=env.reset(seed=seed+episode)
        heights=[]; uprights=[]; contacts=[]; speeds=[]; max_closure=0.0
        initial_position=env.data.xpos[env.base_id,:2].copy()
        initial_warning_counts=np.array(env.data.warning.number).copy()
        while True:
            if policy is not None:
                action,_=policy.predict(obs,deterministic=True)
            else:
                action=rng.uniform(-1,1,6) if random_actions else np.zeros(6)
            obs,reward,done,truncated,info=env.step(action)
            heights.append(info["hip_height_m"]); uprights.append(info["upright"]); contacts.append(info["wheel_contacts"])
            speeds.append(info["forward_velocity_m_s"])
            max_closure=max(max_closure,info["closure_error_m"])
            if done or truncated:
                records.append({**info["episode_metrics"],"mean_hip_height_m":float(np.mean(heights)),
                                "mean_upright":float(np.mean(uprights)),"both_wheels_contact_fraction":float(np.mean(np.array(contacts)==2)),
                                "max_closure_error_m":max_closure,
                                "planar_displacement_m":float(np.linalg.norm(env.data.xpos[env.base_id,:2]-initial_position)),
                                "forward_velocity_rmse_m_s":float(np.sqrt(np.mean((np.array(speeds)-env.commands[0])**2))),
                                "simulator_warning_counts":(np.array(env.data.warning.number)-initial_warning_counts).tolist()})
                break
    env.close()
    return {"simulation_only":True,"episodes":records,"seed":seed,
            "evaluation_horizon_seconds":env.max_steps*env.dt,
            "checkpoint_sha256":hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
            "mean_reward":float(np.mean([r["reward"] for r in records])),
            "mean_episode_seconds":float(np.mean([r["seconds"] for r in records])),
            "timeout_fraction":float(np.mean([r["reason"]=="timeout" for r in records])),
            "warning":"Approximate no-spring standing experiment; not hardware or obstacle validation."}


class StatusCallback(BaseCallback):
    def __init__(self,run_directory,total_steps,start_steps=0):
        super().__init__()
        self.run_directory=Path(run_directory)
        self.total_steps=total_steps
        self.start_steps=start_steps
        self.started=time.monotonic()
        self.last_update=0

    def _on_step(self):
        if self.num_timesteps-self.last_update>=1024:
            self.last_update=self.num_timesteps
            status={"state":"training","simulation_only":True,"timesteps":self.num_timesteps,
                    "requested_additional_timesteps":self.total_steps,
                    "starting_timesteps":self.start_steps,
                    "additional_timesteps":self.num_timesteps-self.start_steps,
                    "target_total_timesteps":self.start_steps+self.total_steps,
                    "elapsed_seconds":time.monotonic()-self.started,
                    "pid":os.getpid()}
            (self.run_directory/"status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        return True


def train(config_path,total_steps=None,num_envs=None,device="cpu",run_name=None,resume=None):
    config_path=Path(config_path).resolve()
    config=load_config(config_path)
    settings=config["training"]
    total_steps=int(settings["total_timesteps"] if total_steps is None else total_steps)
    num_envs=int(settings["num_envs"] if num_envs is None else num_envs)
    if total_steps<=0 or num_envs<1 or num_envs>14:
        raise ValueError("Require positive steps and 1..14 environments")
    model_dir=ROOT/config["experiment"]["output_directory"]
    model_path=model_dir/"robot_experiment.xml"
    if not model_path.is_file() or not (model_dir/"smoke_report.json").is_file():
        raise RuntimeError("Run prepare_mujoco_experiment.py and smoke_mujoco.py first")
    smoke=json.loads((model_dir/"smoke_report.json").read_text())
    if not smoke.get("passed"):
        raise RuntimeError("Model has not passed smoke checks")
    for key,path in (("config_sha256",config_path),("model_sha256",model_path),
                     ("env_sha256",ROOT/"source/climber_legged_rl/mujoco_env.py")):
        if smoke.get(key)!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError("Smoke report is stale: "+key+". Rerun smoke_mujoco.py.")
    # Validate before spawning workers or creating a run directory.
    if resume:
        resume=Path(resume).resolve()
        previous=resume.parent.parent if resume.parent.name in ("checkpoints","best") else resume.parent
        previous_model=previous/"robot_experiment.xml"
        if not resume.is_file() or not previous_model.is_file() or previous_model.read_bytes()!=model_path.read_bytes():
            raise ValueError("Resume requires an existing checkpoint and the identical derived model.")
        previous_config=yaml.safe_load((previous/"config.yaml").read_text())
        for key in ("mass","physics","control","task","reward"):
            if previous_config[key]!=config[key]:
                raise ValueError("Resume configuration changed: "+key)
    run_name=run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    if Path(run_name).name!=run_name:
        raise ValueError("Run name must be a single directory name")
    run_dir=ROOT/"outputs"/"mujoco_runs"/run_name
    run_dir.mkdir(parents=True,exist_ok=False)
    (run_dir/"checkpoints").mkdir()
    (run_dir/"config.yaml").write_text(yaml.safe_dump(config,allow_unicode=True),encoding="utf-8")
    (run_dir/"model_report.json").write_bytes((model_dir/"model_report.json").read_bytes())
    (run_dir/"robot_experiment.xml").write_bytes(model_path.read_bytes())
    seed=settings["seed"]
    torch.set_num_threads(1)
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    try:
        git_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        git_sha="uncommitted_uploaded_working_tree"
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in (ROOT/"source",ROOT/"scripts") for p in folder.rglob("*.py")}
    metadata={"simulation_only":True,"backend":"MuJoCo_CPU_physics_SB3_PPO",
              "device":device,"cuda_available":torch.cuda.is_available(),"torch":torch.__version__,
              "python":platform.python_version(),"pid":os.getpid(),"git_sha":git_sha,
              "source_sha256":sources,"model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "requested_timesteps":total_steps,"num_envs":num_envs}
    (run_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("RUN_DIRECTORY="+str(run_dir),flush=True)
    print("SIMULATION ONLY: APPROXIMATE DYNAMICS, NO HARDWARE DEPLOYMENT",flush=True)
    factories=[_factory(model_path,config_path,seed+i,str(run_dir/f"env_{i}")) for i in range(num_envs)]
    vector=SubprocVecEnv(factories,start_method="spawn") if num_envs>1 else DummyVecEnv(factories)
    vector=VecCheckNan(vector,raise_exception=True)
    if resume:
        model=PPO.load(str(resume),env=vector,device=device)
        metadata["resume_checkpoint"]=str(resume)
        metadata["resume_timesteps"]=model.num_timesteps
        (run_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    else:
        model=PPO("MlpPolicy",vector,device=device,seed=seed,n_steps=settings["n_steps"],
                  batch_size=min(settings["batch_size"],settings["n_steps"]*num_envs),
                  n_epochs=settings["n_epochs"],learning_rate=settings["learning_rate"],gamma=settings["gamma"],
                  gae_lambda=settings["gae_lambda"],clip_range=settings["clip_range"],ent_coef=settings["ent_coef"],
                  max_grad_norm=settings["max_grad_norm"],verbose=1,
                  policy_kwargs=dict(net_arch=dict(pi=settings["actor_hidden_dims"],vf=settings["critic_hidden_dims"]),activation_fn=torch.nn.ReLU))
    model.set_logger(configure(str(run_dir),["stdout","csv","tensorboard"]))
    eval_env=VecCheckNan(DummyVecEnv([_factory(model_path,config_path,30000,None)]),raise_exception=True)
    callbacks=CallbackList([CheckpointCallback(save_freq=max(settings["checkpoint_steps"]//num_envs,1),
                    save_path=str(run_dir/"checkpoints"),name_prefix="climber"),StatusCallback(run_dir,total_steps,model.num_timesteps),
                    EvalCallback(eval_env,best_model_save_path=str(run_dir/"best"),log_path=str(run_dir/"eval"),
                                 eval_freq=max(50000//num_envs,1),n_eval_episodes=5,deterministic=True)])
    try:
        baseline=evaluate(None,config_path,model_path,episodes=5)
        (run_dir/"baseline_zero.json").write_text(json.dumps(baseline,indent=2),encoding="utf-8")
        model.learn(total_timesteps=total_steps,callback=callbacks,log_interval=1,reset_num_timesteps=not bool(resume))
        model.save(str(run_dir/"final_policy"))
        result=evaluate(run_dir/"final_policy.zip",config_path,model_path,episodes=10)
        (run_dir/"evaluation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        status={"state":"completed","simulation_only":True,"timesteps":model.num_timesteps,
                "evaluation_mean_seconds":result["mean_episode_seconds"],"timeout_fraction":result["timeout_fraction"]}
        (run_dir/"status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        print(json.dumps(status),flush=True)
    except BaseException as exc:
        model.save(str(run_dir/"interrupted_policy"))
        (run_dir/"status.json").write_text(json.dumps({"state":"failed","error":repr(exc),"timesteps":model.num_timesteps},indent=2),encoding="utf-8")
        raise
    finally:
        vector.close()
        eval_env.close()
    return run_dir
