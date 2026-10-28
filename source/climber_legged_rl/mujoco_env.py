"""Climber MJCF standing task for CAD dynamics and historical experiments."""
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
        self.stationary_hold = self.config['task'].get('stationary_hold', False)
        if self.stationary_hold:
            task = self.config['task']
            for key in ('position_sigma_m', 'heading_sigma_rad', 'linear_tracking_sigma_m_s', 'yaw_tracking_sigma_rad_s',
                        'stationary_max_displacement_m', 'stationary_max_heading_rad'):
                if not np.isfinite(task.get(key, 0)) or task.get(key, 0) <= 0:
                    raise ValueError(key + ' must be finite and positive for stationary hold')
            if any(value != 0 for key in ('command_forward_m_s', 'command_yaw_rad_s') for value in task[key]):
                raise ValueError('Stationary hold requires zero movement commands')
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
        self.cad_dynamics = self.config['experiment'].get('model_route') == 'cad_dynamics'
        reference_mesh = self.config['physics'].get('chassis_reference_mesh')
        if reference_mesh:
            from .chassis_frame import frame_from_mounting_plate
            self.chassis_axes = frame_from_mounting_plate(self.model, reference_mesh)
        else:
            # Preserve the observation contract of historical checkpoints.
            self.chassis_axes = np.diag([-1.0, -1.0, 1.0])
        self.use_motor_midpoint = self.config['physics'].get('standing_distance_definition') == 'motor_midpoint_to_wheel_center_side_view'
        if self.use_motor_midpoint:
            from .stance_geometry import motor_midpoints_in_base, wheel_centers_in_wheel_bodies
            self.motor_midpoints,_ = motor_midpoints_in_base(self.model,self.config['physics'])
            self.wheel_centers = wheel_centers_in_wheel_bodies(self.model,self.config['physics'])
            self.target_virtual_leg_length = self.config['physics']['rest_motor_midpoint_to_wheel_center_m']
            sigma_length = self.config['task'].get('virtual_leg_length_sigma_m',.02)
            if not np.isfinite(sigma_length) or sigma_length <= 0:
                raise ValueError('virtual_leg_length_sigma_m must be finite and positive')
        sigma = self.config['task'].get('upright_sigma_rad')
        if sigma is not None and (not np.isfinite(sigma) or sigma <= 0):
            raise ValueError('upright_sigma_rad must be finite and positive')
        self.base_geom_ids = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.base_id}
        self.wheel_geom_ids = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self.wheel_ids}
        self.spring_dofs = ([int(self.model.joint(s+'_gas_slide_joint').dofadr[0]) for s in ('left','right')]
                            if self.cad_dynamics else [])
        self.dt = self.model.opt.timestep * self.control["decimation"]
        from .push_disturbance import PushDisturbance
        self.push = PushDisturbance(self.config.get('push', {}), self.dt)
        if self.push.enabled and not self.stationary_hold:
            raise ValueError('Push recovery currently requires stationary hold observations')
        self.max_steps = math.ceil(self.config["task"]["episode_seconds"] / self.dt)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(30 if self.stationary_hold else 24,), dtype=np.float32)
        self.nominal_qpos = self.model.key_qpos[0].copy()
        self.nominal_legs = self.nominal_qpos[self.q_indices[:4]].copy()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self._set_hold_reference()
        self.target_hip_height = float(np.mean(self.data.xpos[self.hip_ids, 2]))
        self.torque_limits = np.array([self.control["leg_torque_limit_Nm"]] * 4 + [self.control["wheel_torque_limit_Nm"]] * 2)
        self.speed_limits = np.array([self.control["leg_speed_limit_rad_s"]] * 4 + [self.control["wheel_speed_limit_rad_s"]] * 2)
        self.leg_low = np.array([self.model.joint(n).range[0] for n in names[:4]])
        self.leg_high = np.array([self.model.joint(n).range[1] for n in names[:4]])
        self.action = np.zeros(6, dtype=np.float64)
        self.commands = np.zeros(2)
        self.steps = 0
        self._total_reward = 0.0
        self._last_warning_counts = np.array(self.data.warning.number).copy()

    def _rotation(self):
        return self.data.xmat[self.base_id].reshape(3, 3) @ self.chassis_axes

    def _kinematics(self):
        rotation = self._rotation()
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 0)
        angular = rotation.T @ velocity[:3]
        linear = rotation.T @ velocity[3:]
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        return angular, linear, gravity

    def _set_hold_reference(self):
        self.hold_position = self.data.xpos[self.base_id, :2].copy()
        rotation = self._rotation()
        self.hold_heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        self.hold_max_displacement = 0.
        self.hold_max_heading = 0.
        self.hold_speed_squared = 0.
        self.hold_yaw_squared = 0.

    def _hold_errors(self):
        displacement = self.data.xpos[self.base_id, :2] - self.hold_position
        rotation = self._rotation()
        heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        heading_error = float(np.arctan2(np.sin(heading-self.hold_heading), np.cos(heading-self.hold_heading)))
        return displacement, heading_error

    def stationary_success(self, reached_horizon, max_displacement_m, max_heading_error_rad,
                           planar_speed_rmse_m_s, yaw_rate_rmse_rad_s):
        task = self.config['task']
        return bool(reached_horizon
                    and max_displacement_m <= task.get('stationary_max_displacement_m', .05)
                    and max_heading_error_rad <= task.get('stationary_max_heading_rad', np.deg2rad(5))
                    and planar_speed_rmse_m_s <= task.get('linear_tracking_sigma_m_s', .05)
                    and yaw_rate_rmse_rad_s <= task.get('yaw_tracking_sigma_rad_s', .05))

    def _observation(self):
        angular, linear, gravity = self._kinematics()
        obs = np.concatenate((angular * 0.25, gravity, self.commands * [2.0, 0.25],
                              self.data.qpos[self.q_indices[:4]] - self.nominal_legs,
                              self.data.qvel[self.v_indices] * 0.05, self.action))
        if self.stationary_hold:
            displacement, heading_error = self._hold_errors()
            local_displacement = self._rotation().T @ np.r_[displacement, 0.]
            obs = np.r_[obs, local_displacement[:2] * 5., np.sin(heading_error),
                        np.cos(heading_error)-1., linear[:2] * 2.]
        if obs.shape != self.observation_space.shape or not np.isfinite(obs).all():
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
        self._set_hold_reference()
        self._last_warning_counts = np.array(self.data.warning.number).copy()
        if self.push.enabled:
            self.push.reset(np.random.default_rng(self.np_random.integers(0, 2**63)))
            self.data.xfrc_applied[self.base_id, :] = 0.
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
        if self.push.enabled:
            self.data.xfrc_applied[self.base_id, :] = 0.
            self.data.xfrc_applied[self.base_id, :3] = self.push.force(self.steps, self.max_steps)
        with np.errstate(over="raise", invalid="raise"):
            for _ in range(self.control["decimation"]):
                speed = self.data.qvel[self.v_indices]
                torque[:4] = self.control["leg_kp"] * (target_legs - self.data.qpos[self.q_indices[:4]]) - self.control["leg_kd"] * speed[:4]
                torque[4:] = self.control["wheel_kv"] * (target_wheels - speed[4:])
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                torque[(np.abs(speed) > self.speed_limits) & (torque * speed > 0)] = 0
                self.data.ctrl[:] = torque
                if self.spring_dofs:
                    self.data.qfrc_applied[:] = 0.0
                    self.data.qfrc_applied[self.spring_dofs] = self.config['spring']['force_N']
                mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or np.max(np.abs(self.data.qvel)) > 10000:
            raise RuntimeError("MuJoCo diverged; aborting rather than training on corrupted transitions")
        if abs(self.data.time - self.steps * self.dt) > 1e-6:
            raise RuntimeError("Simulator reset time after instability; stop and inspect model")
        if np.any(np.array(self.data.warning.number) > self._last_warning_counts):
            raise RuntimeError('MuJoCo issued a dynamics warning; inspect the model before continuing')
        angular, linear, gravity = self._kinematics()
        upright = float(np.clip(-gravity[2], -1, 1))
        tilt = float(np.arccos(upright))
        height = float(np.mean(self.data.xpos[self.hip_ids, 2]))
        target_height = (self.target_hip_height if self.cad_dynamics else
                         self.config["physics"]["wheel_radius_m"] + self.config["physics"]["rest_height_hip_above_wheel_m"])
        max_closure = float(np.linalg.norm(closure_error(self.model, self.data),axis=1).max())
        base_contact = False
        wheel_contacts = set()
        for contact in self.data.contact:
            pair = {contact.geom1, contact.geom2}
            if self.ground_id in pair:
                base_contact |= bool(pair.intersection(self.base_geom_ids))
                if self.cad_dynamics:
                    wheel_contacts.update(self.model.geom_bodyid[g] for g in pair.intersection(self.wheel_geom_ids))
                else:
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
        sigma = task.get('upright_sigma_rad')
        posture_reward = upright if sigma is None else np.exp(-(tilt/sigma)**2)
        terms = {
            "upright": posture_reward * weights["upright"],
            "hip_height": np.exp(-((height-target_height)/0.04)**2) * weights.get("hip_height",0.),
            "linear_tracking": np.exp(-((linear[0]-self.commands[0])**2 + linear[1]**2)/0.1) * weights["linear_tracking"],
            "yaw_tracking": np.exp(-(angular[2]-self.commands[1])**2/0.25) * weights["yaw_tracking"],
            "action_rate": -weights["action_rate"] * float(np.mean((action-previous)**2)),
            "normalized_effort": -weights["normalized_effort"] * float(np.mean((torque/self.torque_limits)**2)),
            "joint_speed": -weights["joint_speed"] * float(np.mean(self.data.qvel[self.v_indices]**2)),
        }
        displacement, heading_error = self._hold_errors()
        if self.stationary_hold:
            terms['linear_tracking'] = float(np.exp(-np.sum(linear[:2]**2)/task['linear_tracking_sigma_m_s']**2)) * weights['linear_tracking']
            terms['yaw_tracking'] = float(np.exp(-(angular[2]/task['yaw_tracking_sigma_rad_s'])**2)) * weights['yaw_tracking']
            terms['position_hold'] = float(np.exp(-np.sum(displacement**2)/task['position_sigma_m']**2)) * weights['position_hold']
            terms['heading_hold'] = float(np.exp(-(heading_error/task['heading_sigma_rad'])**2)) * weights['heading_hold']
        if 'alive' in weights:
            terms['alive'] = weights['alive'] if not terminated else 0.
        if 'angular_velocity' in weights:
            terms['angular_velocity'] = -weights['angular_velocity'] * float(np.sum(angular[:2]**2))
        leg_geometry = None
        if self.use_motor_midpoint:
            from .stance_geometry import virtual_leg_geometry
            leg_geometry = virtual_leg_geometry(self.model,self.data,self.motor_midpoints,self.wheel_centers,self.chassis_axes)
            sigma_length = task.get('virtual_leg_length_sigma_m',.02)
            terms['virtual_leg_length'] = float(np.mean(np.exp(-((leg_geometry['side_view_distance_m']-self.target_virtual_leg_length)/sigma_length)**2))) * weights['virtual_leg_length']
        reward = float(sum(terms.values()) - (weights["fall_penalty"] if terminated else 0))
        self._total_reward += reward
        info = {"simulation_only": True, "upright": upright, "hip_height_m": height,
                'tilt_rad':tilt,
                "closure_error_m": max_closure, "wheel_contacts": len(wheel_contacts),
                "forward_velocity_m_s": float(linear[0]), "yaw_rate_rad_s": float(angular[2]),
                "planar_displacement_m": float(np.linalg.norm(displacement)),
                "heading_error_rad": heading_error, "planar_speed_m_s": float(np.linalg.norm(linear[:2])),
                "termination_reason": reason, "reward_terms": terms}
        if terminated or truncated:
            info["episode_metrics"] = {"reward": self._total_reward, "seconds": self.steps*self.dt, "reason": reason or "timeout"}
        if self.stationary_hold:
            self.hold_max_displacement = max(self.hold_max_displacement, info['planar_displacement_m'])
            self.hold_max_heading = max(self.hold_max_heading, abs(heading_error))
            self.hold_speed_squared += info['planar_speed_m_s']**2
            self.hold_yaw_squared += float(angular[2])**2
            if self.push.enabled:
                self.push.observe(self.steps, info['planar_displacement_m'], heading_error,
                                  info['planar_speed_m_s'], float(angular[2]), tilt, task)
                info['push_force_world_N'] = self.data.xfrc_applied[self.base_id, :3].tolist()
            if terminated or truncated:
                info['is_success'] = self.stationary_success(
                    truncated, self.hold_max_displacement, self.hold_max_heading,
                    np.sqrt(self.hold_speed_squared/self.steps), np.sqrt(self.hold_yaw_squared/self.steps))
                if self.push.enabled:
                    info['push_metrics'] = self.push.metrics()
                    info['push_recovery_success'] = self.push.success(
                        truncated, info['is_success'], self.hold_max_displacement, self.hold_max_heading,
                        info['planar_displacement_m'], heading_error, info['planar_speed_m_s'],
                        float(angular[2]), tilt, task)
                    info['is_success'] = info['push_recovery_success']
        if leg_geometry is not None:
            info['virtual_leg_lengths_m'] = leg_geometry['side_view_distance_m'].tolist()
            info['virtual_leg_length_error_m'] = float(np.mean(leg_geometry['side_view_distance_m']-self.target_virtual_leg_length))
            info['motor_midpoint_height_m'] = float(leg_geometry['motor_midpoints_world_m'][:,2].mean())
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
        option.geomgroup[1] = int(self.cad_dynamics)
        self.renderer.update_scene(self.data,camera=camera,scene_option=option)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
