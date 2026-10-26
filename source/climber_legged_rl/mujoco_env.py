"""Approximate Climber MJCF standing task; explicitly not a hardware model."""
from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .mujoco_model import DEFAULT_CONFIG, load_config, closure_error


class ClimberMujocoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, model_path, config_path=DEFAULT_CONFIG, render_mode=None):
        super().__init__()
        self.config = load_config(config_path)
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.renderer = None
        self.control = self.config["control"]
        names = self.control["leg_joints"] + self.control["wheel_joints"]
        if self.model.nu != 6 or any(self.model.actuator(i).trnid[0] != self.model.joint(n).id for i, n in enumerate(names)):
            raise ValueError("Model motor ordering must match four leg joints and two wheels")
        self.q_indices = np.array([int(self.model.joint(n).qposadr[0]) for n in names])
        self.v_indices = np.array([int(self.model.joint(n).dofadr[0]) for n in names])
        self.base_id = self.model.body("base").id
        self.hip_ids = [self.model.body(f"{s}_link1").id for s in ("left", "right")]
        self.wheel_ids = [self.model.body(f"{s}_link 3").id for s in ("left", "right")]
        self.wheel_geoms = [self.model.geom(f"{s}_wheel_collision").id for s in ("left", "right")]
        self.ground_id = self.model.geom("ground").id
        self.base_geom_id = self.model.geom("base_collision").id
        self.dt = self.model.opt.timestep * self.control["decimation"]
        self.max_steps = math.ceil(self.config["task"]["episode_seconds"] / self.dt)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(24,), dtype=np.float32)
        self.nominal_qpos = self.model.key_qpos[0].copy()
        self.nominal_legs = self.nominal_qpos[self.q_indices[:4]].copy()
        self.torque_limits = np.array([self.control["leg_torque_limit_Nm"]] * 4 + [self.control["wheel_torque_limit_Nm"]] * 2)
        self.speed_limits = np.array([self.control["leg_speed_limit_rad_s"]] * 4 + [self.control["wheel_speed_limit_rad_s"]] * 2)
        self.leg_low = np.array([self.model.joint(n).range[0] for n in names[:4]])
        self.leg_high = np.array([self.model.joint(n).range[1] for n in names[:4]])
        self.action = np.zeros(6, dtype=np.float64)
        self.commands = np.zeros(2)
        self.steps = 0
        self._total_reward = 0.0

    def _rotation(self):
        # CAD body axes rotate by pi at root; use CAD -x/-y as canonical x/y.
        return self.data.xmat[self.base_id].reshape(3, 3) @ np.diag([-1.0, -1.0, 1.0])

    def _kinematics(self):
        rotation = self._rotation()
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 0)
        angular = rotation.T @ velocity[:3]
        linear = rotation.T @ velocity[3:]
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        return angular, linear, gravity

    def _observation(self):
        angular, _, gravity = self._kinematics()
        obs = np.concatenate((angular * 0.25, gravity, self.commands * [2.0, 0.25],
                              self.data.qpos[self.q_indices[:4]] - self.nominal_legs,
                              self.data.qvel[self.v_indices] * 0.05, self.action))
        if obs.shape != (24,) or not np.isfinite(obs).all():
            raise RuntimeError("Invalid observation: numerical instability or wrong dimensions")
        return np.clip(obs, -10, 10).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        tilt = self.np_random.normal(0.0, self.config["task"]["reset_tilt_std_rad"], 2)
        # A rigid-base rotation preserves every internal closed-chain relation.
        cr, sr = np.cos(tilt[0] / 2), np.sin(tilt[0] / 2)
        cp, sp = np.cos(tilt[1] / 2), np.sin(tilt[1] / 2)
        delta = np.array([cr * cp, sr * cp, cr * sp, -sr * sp])
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, delta, self.nominal_qpos[3:7])
        self.data.qpos[3:7] = quat
        self.data.qvel[:6] = self.np_random.normal(0, self.config["task"]["reset_velocity_std"], 6)
        self.action[:] = 0.0
        self.commands[:] = [self.np_random.uniform(*self.config["task"]["command_forward_m_s"]),
                            self.np_random.uniform(*self.config["task"]["command_yaw_rad_s"])]
        self.steps = 0
        self._total_reward = 0.0
        mujoco.mj_forward(self.model, self.data)
        # Lift the entire reset slightly to avoid starting in ground penetration.
        min_wheel_z = min(self.data.xpos[i, 2] for i in self.wheel_ids)
        self.data.qpos[2] += max(0.0, self.config["physics"]["wheel_radius_m"] + 0.001 - min_wheel_z)
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {"simulation_only": True}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Expected six finite policy actions")
        action = np.clip(action, -1, 1)
        previous = self.action.copy()
        self.action[:] = action
        target_legs = np.clip(self.nominal_legs + action[:4] * self.control["leg_action_scale_rad"], self.leg_low, self.leg_high)
        target_wheels = action[4:] * self.control["wheel_action_scale_rad_s"]
        torque = np.zeros(6)
        with np.errstate(over="raise", invalid="raise"):
            for _ in range(self.control["decimation"]):
                speed = self.data.qvel[self.v_indices]
                torque[:4] = self.control["leg_kp"] * (target_legs - self.data.qpos[self.q_indices[:4]]) - self.control["leg_kd"] * speed[:4]
                torque[4:] = self.control["wheel_kv"] * (target_wheels - speed[4:])
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                torque[(np.abs(speed) > self.speed_limits) & (torque * speed > 0)] = 0
                self.data.ctrl[:] = torque
                mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or np.max(np.abs(self.data.qvel)) > 10000:
            raise RuntimeError("MuJoCo diverged; aborting rather than training on corrupted transitions")
        if abs(self.data.time - self.steps * self.dt) > 1e-6:
            raise RuntimeError("Simulator reset time after instability; stop and inspect model")
        angular, linear, gravity = self._kinematics()
        upright = float(np.clip(-gravity[2], -1, 1))
        height = float(np.mean(self.data.xpos[self.hip_ids, 2]))
        target_height = self.config["physics"]["wheel_radius_m"] + self.config["physics"]["rest_height_hip_above_wheel_m"]
        max_closure = float(np.linalg.norm(closure_error(self.model, self.data),axis=1).max())
        base_contact = False
        wheel_contacts = set()
        for contact in self.data.contact:
            pair = {contact.geom1, contact.geom2}
            if self.ground_id in pair:
                base_contact |= self.base_geom_id in pair
                wheel_contacts.update(pair.intersection(self.wheel_geoms))
        task = self.config["task"]
        reason = None
        if base_contact:
            reason = "base_contact"
        elif self.data.xpos[self.base_id,2] < task["terminate_base_height_m"]:
            reason = "low_base"
        elif upright < np.cos(task["terminate_tilt_rad"]):
            reason = "tilt"
        elif max_closure > task["terminate_max_closure_error_m"]:
            reason = "closure"
        terminated = reason is not None
        truncated = self.steps >= self.max_steps and not terminated
        weights = self.config["reward"]
        terms = {
            "upright": upright * weights["upright"],
            "hip_height": np.exp(-((height-target_height)/0.04)**2) * weights["hip_height"],
            "linear_tracking": np.exp(-((linear[0]-self.commands[0])**2 + linear[1]**2)/0.1) * weights["linear_tracking"],
            "yaw_tracking": np.exp(-(angular[2]-self.commands[1])**2/0.25) * weights["yaw_tracking"],
            "action_rate": -weights["action_rate"] * float(np.mean((action-previous)**2)),
            "normalized_effort": -weights["normalized_effort"] * float(np.mean((torque/self.torque_limits)**2)),
            "joint_speed": -weights["joint_speed"] * float(np.mean(self.data.qvel[self.v_indices]**2)),
        }
        reward = float(sum(terms.values()) - (weights["fall_penalty"] if terminated else 0))
        self._total_reward += reward
        info = {"simulation_only": True, "upright": upright, "hip_height_m": height,
                "closure_error_m": max_closure, "wheel_contacts": len(wheel_contacts),
                "forward_velocity_m_s": float(linear[0]), "yaw_rate_rad_s": float(angular[2]),
                "termination_reason": reason, "reward_terms": terms}
        if terminated or truncated:
            info["episode_metrics"] = {"reward": self._total_reward, "seconds": self.steps*self.dt, "reason": reason or "timeout"}
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model,height=480,width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.xpos[self.base_id]
        camera.distance = 1.1
        camera.azimuth = 135
        camera.elevation = -20
        option = mujoco.MjvOption()
        option.geomgroup[1] = 0  # Inspect primitives; source visual meshes can be toggled later.
        self.renderer.update_scene(self.data,camera=camera,scene_option=option)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
