"""Opt-in, single-policy belly-down get-up, standing and push-recovery task."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .mujoco_env import ClimberMujocoEnv
from .mujoco_model import ROOT, closure_error, load_config
from .stance_geometry import virtual_leg_geometry
from .spring_geometry import configured_eye_limits, eye_center_distance


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration_hash(path):
    return hashlib.sha256(json.dumps(load_config(path), sort_keys=True).encode()).hexdigest()


def prepare_pose_bank(config_path):
    """Solve every passive joint, then translate the complete closed robot to ground."""
    config = load_config(config_path)
    spring_limits = configured_eye_limits(config, required=True)
    folder = ROOT / config['experiment']['output_directory']
    env = ClimberMujocoEnv(folder / 'robot_experiment.xml', config_path)
    model, data = env.model, env.data
    qpos = env.nominal_qpos.copy()
    wheel_q = set(env.q_indices[4:].tolist())
    indices = [i for i in range(7, model.nq) if i not in wheel_q]
    lows, highs = [], []
    for index in indices:
        joint = int(np.flatnonzero(model.jnt_qposadr == index)[0])
        lows.append(model.jnt_range[joint, 0] if model.jnt_limited[joint] else -np.inf)
        highs.append(model.jnt_range[joint, 1] if model.jnt_limited[joint] else np.inf)
    geometry = lambda: virtual_leg_geometry(model, data, env.motor_midpoints, env.wheel_centers, env.chassis_axes)
    standing_geometry = geometry()
    # Cache meshes in their geom frames; all vertices remain in the collision check.
    meshes = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]].copy()
        meshes.append((geom, vertices))
    poses = []
    extension_path = [qpos[env.q_indices[:4]].copy()]
    try:
        getup = config['getup']
        if getup.get('reset_pose') != 'rear_extended_belly_down':
            raise ValueError('Get-up requires a rear-extended belly-down reset definition')
        offsets = np.asarray(getup['rearward_offsets_m'], dtype=float)
        if not len(offsets) or not np.isfinite(offsets).all() or np.min(offsets) <= 0:
            raise ValueError('Rearward offsets must be finite and positive')
        # With a horizontal base, belly and tire support fix the vertical offset.
        # The signed sagittal displacement is independent of the standing 135 mm length.
        def mesh_bottoms():
            return {g: float((v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min())
                    for g, v in meshes}
        bottoms = mesh_bottoms()
        base_bottom = min(v for g, v in bottoms.items() if g in env.base_geom_ids)
        tire_radius = standing_geometry['wheel_centers_world_m'][:, 2].mean() - min(
            v for g, v in bottoms.items() if g in env.wheel_geom_ids)
        prone_vertical = standing_geometry['motor_midpoints_world_m'][:, 2].mean() - base_bottom - tire_radius
        previous_offset = float(standing_geometry['forward_offset_m'].mean())
        for target in sorted(offsets):
            for rearward in np.linspace(previous_offset, target, max(2, int(np.ceil((target-previous_offset)/.01))+1))[1:]:
                initial = qpos.copy()
                fraction = min(1., max(0., (rearward-standing_geometry['forward_offset_m'].mean())/.10))
                vertical = (1.-fraction)*standing_geometry['vertical_separation_m'].mean()+fraction*prone_vertical

                def residual(values):
                    data.qpos[:] = initial
                    data.qpos[indices] = values
                    mujoco.mj_forward(model, data)
                    shape = geometry()
                    return np.r_[closure_error(model, data).ravel() * 3000.,
                                 (shape['forward_offset_m'] - rearward) * 30.,
                                 (shape['vertical_separation_m'] - vertical) * 30.,
                                 (data.qpos[env.q_indices[:2]] - data.qpos[env.q_indices[2:4]]) * 2.,
                                 (values - initial[indices]) * 0.0001]

                fit = least_squares(residual, np.clip(initial[indices], np.array(lows) + 1e-9, np.array(highs) - 1e-9),
                                    bounds=(lows, highs), max_nfev=1200, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                residual(fit.x)
                qpos = data.qpos.copy()
                extension_path.append(qpos[env.q_indices[:4]].copy())
            previous_offset = float(target)
            # Tire mesh orientation can move its bottom slightly; solve actual support height.
            for _ in range(2):
                bottoms = mesh_bottoms()
                vertical = float(geometry()['vertical_separation_m'].mean() +
                             min(v for g, v in bottoms.items() if g in env.wheel_geom_ids) -
                             min(v for g, v in bottoms.items() if g in env.base_geom_ids))
                initial = qpos.copy()
                fit = least_squares(residual, qpos[indices], bounds=(lows, highs), max_nfev=600,
                                    ftol=1e-10, xtol=1e-10, gtol=1e-10)
                residual(fit.x)
                qpos = data.qpos.copy()
            bottoms = mesh_bottoms()
            shift = config['getup']['ground_clearance_m'] - min(bottoms.values())
            qpos[2] += shift
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            shape = geometry()
            error = float(np.linalg.norm(closure_error(model, data), axis=1).max())
            base_bottom = min(v for g, v in bottoms.items() if g in env.base_geom_ids) + shift
            wheel_bottom = min(v for g, v in bottoms.items() if g in env.wheel_geom_ids) + shift
            spring_lengths = {side: eye_center_distance(model, data, side) for side in ('left', 'right')}
            if any(not (spring_limits[side][0] or 0.)-1e-8 <= length <= spring_limits[side][1]+1e-8
                   for side, length in spring_lengths.items()):
                raise RuntimeError('Prone pose exceeds physical gas-spring pin-center limits')
            if (error > 1e-4 or max(abs(shape['forward_offset_m'] - target)) > .001
                    or base_bottom > .0015 or wheel_bottom > .0015):
                raise RuntimeError(f'Rear-extended reset IK failed: closure={error}, belly={base_bottom}, tire={wheel_bottom}')
            poses.append(dict(length_m=float(shape['side_view_distance_m'].mean()), rearward_offset_m=float(target),
                              forward_offset_m=shape['forward_offset_m'].tolist(),
                              vertical_separation_m=shape['vertical_separation_m'].tolist(),
                              qpos=qpos.tolist(), closure_error_m=error,
                              spring_eye_distances_m=spring_lengths,
                              base_bottom_m=min(v for g, v in bottoms.items() if g in env.base_geom_ids) + shift,
                              wheel_bottom_m=min(v for g, v in bottoms.items() if g in env.wheel_geom_ids) + shift,
                              leg_reference_rad=qpos[env.q_indices[:4]].tolist()))
            # Store only joint references; dynamics never receives these solved qpos.
            path = np.vstack([extension_path, qpos[env.q_indices[:4]]])[::-1]
            distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            keep = np.r_[True, np.diff(distance) > 1e-10]
            poses[-1]['rise_leg_references_rad'] = path[keep].tolist()
            poses[-1]['rise_reference_progress'] = (distance[keep] / distance[-1]).tolist()
    finally:
        env.close()
    report = dict(model_sha256=file_hash(folder / 'robot_experiment.xml'), config_sha256=configuration_hash(config_path),
                  implementation_sha256=file_hash(__file__), poses=poses, simulation_only=True,
                  reset_definition='chassis horizontal, belly down, both wheels and legs extended rearward; signed chassis-forward coordinate')
    (folder / 'getup_poses.json').write_text(json.dumps(report, indent=2))
    return report


def validate_pose_bank(config_path, folder=None):
    config = load_config(config_path)
    configured_eye_limits(config, required=True)
    folder = Path(folder) if folder is not None else ROOT / config['experiment']['output_directory']
    report = json.loads((folder / 'getup_poses.json').read_text())
    checks = [('model_sha256', folder / 'robot_experiment.xml'), ('implementation_sha256', __file__)]
    if (any(report.get(key) != file_hash(path) for key, path in checks)
            or report.get('config_sha256') != configuration_hash(config_path)):
        raise RuntimeError('Get-up pose bank is stale; rerun prepare_getup.py and smoke checks')
    return report


class ClimberGetupEnv(ClimberMujocoEnv):
    """One PPO actor, one action convention, with a guided PD reference during rising."""
    extra_observation_dimensions = 6

    def __init__(self, model_path, config_path, render_mode=None):
        super().__init__(model_path, config_path, render_mode)
        if not self.cad_dynamics or not self.stationary_hold or not self.use_motor_midpoint:
            raise ValueError('Combined get-up requires the stationary CAD midpoint model')
        self.getup = self.config['getup']
        if self.getup.get('reference_mode', 'joint_interpolation') not in ('joint_interpolation', 'closed_chain_waypoints'):
            raise ValueError('Unknown get-up reference mode')
        if self.getup.get('progress_definition','length_retraction') not in ('length_retraction','absolute_length_error'):
            raise ValueError('Unknown get-up progress definition')
        for key in ('stand_probability',):
            if not 0 <= self.getup[key] <= 1:
                raise ValueError('Invalid get-up probability')
        for key in ('rise_seconds', 'deadline_seconds', 'settle_seconds', 'leg_tolerance_m',
                    'height_tolerance_m', 'tilt_tolerance_rad', 'success_bonus', 'height_reward',
                    'max_displacement_m'):
            if not np.isfinite(self.getup[key]) or self.getup[key] <= 0:
                raise ValueError('Invalid get-up parameter: ' + key)
        if self.getup['rise_seconds'] + self.getup['settle_seconds'] >= self.getup['deadline_seconds']:
            raise ValueError('Rise schedule must leave time to settle')
        if self.getup['deadline_seconds'] >= self.config['task']['episode_seconds']:
            raise ValueError('Get-up deadline must precede episode end')
        self.pose_report = validate_pose_bank(config_path, Path(model_path).parent)
        self.pose_bank = self.pose_report['poses']
        self.reset_legs = self.nominal_legs.copy()
        self.rise_waypoints = None
        self.rise_progress = None
        self.reset_mode = 'standing'
        self.acquired = True
        self.acquisition_step = 0
        self.settled_steps = 0
        self.hold_after_rise_max_position = 0.
        self.hold_after_rise_max_heading = 0.
        self.hold_after_rise_speed2 = 0.
        self.hold_after_rise_yaw2 = 0.
        self.hold_after_rise_count = 0

    def _phase(self):
        if getattr(self, 'reset_mode', 'standing') == 'standing':
            return 1.
        return min(1., self.steps * self.dt / self.getup['rise_seconds'])

    def _leg_reference(self):
        if not hasattr(self, 'reset_legs'):
            return self.nominal_legs
        t = self._phase()
        blend = t * t * (3. - 2. * t)
        if getattr(self, 'rise_waypoints', None) is not None:
            return np.array([np.interp(blend, self.rise_progress, self.rise_waypoints[:, j]) for j in range(4)])
        return (1. - blend) * self.reset_legs + blend * self.nominal_legs

    def _extra_observation(self):
        reference = self._leg_reference() - self.nominal_legs
        return np.r_[reference, (self.data.xpos[self.hip_ids, 2].mean() - self.target_hip_height) * 5., self._phase()]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        rng = np.random.default_rng(self.np_random.integers(0, 2**63))
        mode = (options or {}).get('reset_mode')
        if mode not in (None, 'standing', 'folded'):
            raise ValueError('reset_mode must be standing or folded')
        folded = mode == 'folded' or (mode is None and rng.random() >= self.getup['stand_probability'])
        self.reset_mode = 'folded' if folded else 'standing'
        if folded:
            index = int(rng.integers(len(self.pose_bank)))
            pose = self.pose_bank[index]
            self.data.qpos[:] = pose['qpos']
            self.data.qvel[:] = 0.
            self.data.time = 0.
            self.data.ctrl[:] = 0.
            mujoco.mj_forward(self.model, self.data)
            self.reset_legs = self.data.qpos[self.q_indices[:4]].copy()
            if self.getup.get('reference_mode', 'joint_interpolation') == 'closed_chain_waypoints':
                self.rise_waypoints = np.asarray(pose['rise_leg_references_rad'])
                self.rise_progress = np.asarray(pose['rise_reference_progress'])
            self.reset_length_m = pose['length_m']
        else:
            self.reset_legs = self.nominal_legs.copy()
            self.rise_waypoints = self.rise_progress = None
            self.reset_length_m = self.target_virtual_leg_length
        self.acquired = not folded
        self.acquisition_step = 0 if self.acquired else None
        self.settled_steps = 0
        self.hold_after_rise_max_position = 0.
        self.hold_after_rise_max_heading = 0.
        self.hold_after_rise_speed2 = 0.
        self.hold_after_rise_yaw2 = 0.
        self.hold_after_rise_count = 0
        self._set_hold_reference()
        self._last_warning_counts = np.array(self.data.warning.number).copy()
        if folded:
            self.push.next_step = None
        return self._observation(), dict(simulation_only=True, reset_mode=self.reset_mode,
                                         reset_length_m=self.reset_length_m)

    def _push_force(self):
        return self.push.force(self.steps, self.max_steps) if self.acquired else np.zeros(3)

    def _termination_reason(self, base_contact, upright, max_closure):
        if self.acquired:
            return super()._termination_reason(base_contact, upright, max_closure)
        if max_closure > self.config['task']['terminate_max_closure_error_m']:
            return 'closure'
        if upright < np.cos(self.config['task']['terminate_tilt_rad']):
            return 'tilt'
        if self.steps * self.dt >= self.getup['deadline_seconds']:
            return 'getup_timeout'
        return None

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        first_success = False
        if not self.acquired:
            if self.getup.get('dense_progress_reward', 0.):
                # Horizontal prone posture alone must not earn full standing reward.
                lengths = np.asarray(info['virtual_leg_lengths_m'])
                progress = float(np.clip((self.reset_length_m - lengths.mean()) /
                    max(self.reset_length_m-self.target_virtual_leg_length, 1e-6), 0., 1.))
                if self.getup.get('progress_definition') == 'absolute_length_error':
                    initial_error=abs(self.reset_length_m-self.target_virtual_leg_length)
                    progress=float(np.clip(1.-np.mean(abs(lengths-self.target_virtual_leg_length))/max(initial_error,1e-6),0.,1.))
                shaped = progress * self.getup['dense_progress_reward']
                for key in ('upright', 'linear_tracking', 'yaw_tracking', 'position_hold', 'heading_hold'):
                    before = info['reward_terms'].get(key, 0.)
                    info['reward_terms'][key] = before * progress
                    shaped -= before * (1.-progress)
                shaped -= self.getup.get('unfinished_penalty', 0.)
                reward += shaped
                self._total_reward += shaped
                info['reward_terms']['getup_progress'] = progress * self.getup['dense_progress_reward']
                info['reward_terms']['getup_unfinished'] = -self.getup.get('unfinished_penalty', 0.)
                info['getup_progress'] = progress
            base_contact = any(self.ground_id in (c.geom1, c.geom2)
                               and bool({c.geom1, c.geom2}.intersection(self.base_geom_ids)) for c in self.data.contact)
            lengths = np.asarray(info['virtual_leg_lengths_m'])
            good = (not base_contact and info['wheel_contacts'] == 2
                    and max(abs(lengths - self.target_virtual_leg_length)) <= self.getup['leg_tolerance_m']
                    and abs(info['hip_height_m'] - self.target_hip_height) <= self.getup['height_tolerance_m']
                    and info['tilt_rad'] <= self.getup['tilt_tolerance_rad']
                    and info['planar_displacement_m'] <= self.config['task']['stationary_max_displacement_m']
                    and abs(info['heading_error_rad']) <= self.config['task']['stationary_max_heading_rad']
                    and info['planar_speed_m_s'] <= self.config['task']['linear_tracking_sigma_m_s']
                    and abs(info['yaw_rate_rad_s']) <= self.config['task']['yaw_tracking_sigma_rad_s'])
            self.settled_steps = self.settled_steps + 1 if good else 0
            if not done and self.settled_steps >= math.ceil(self.getup['settle_seconds'] / self.dt):
                self.acquired = first_success = True
                self.acquisition_step = self.steps
                if self.push.episode_enabled:
                    self.push.next_step = self.steps + math.ceil(self.push.config['first_push_seconds'] / self.dt)
            lift = np.exp(-((info['hip_height_m'] - self.target_hip_height) / .04) ** 2)
            extra = float(lift * self.getup['height_reward'])
            if self.getup.get('dense_progress_reward', 0.):
                extra *= info['getup_progress']
            info['reward_terms']['getup_height'] = extra
            reward += extra
            self._total_reward += extra
        if first_success:
            reward += self.getup['success_bonus']
            self._total_reward += self.getup['success_bonus']
            info['reward_terms']['getup_success'] = self.getup['success_bonus']
        info['reset_mode'] = self.reset_mode
        info['getup_acquired'] = self.acquired
        info['getup_seconds'] = self.acquisition_step * self.dt if self.acquisition_step is not None else None
        if self.acquired:
            self.hold_after_rise_count += 1
            self.hold_after_rise_max_position = max(self.hold_after_rise_max_position, info['planar_displacement_m'])
            self.hold_after_rise_max_heading = max(self.hold_after_rise_max_heading, abs(info['heading_error_rad']))
            self.hold_after_rise_speed2 += info['planar_speed_m_s'] ** 2
            self.hold_after_rise_yaw2 += info['yaw_rate_rad_s'] ** 2
        if done or truncated:
            reached = bool(truncated and self.acquired)
            count = max(1, self.hold_after_rise_count)
            no_push_good = bool(self.stationary_success(reached, self.hold_after_rise_max_position,
                                                        self.hold_after_rise_max_heading,
                                                        np.sqrt(self.hold_after_rise_speed2 / count),
                                                        np.sqrt(self.hold_after_rise_yaw2 / count))
                                and info['planar_displacement_m'] <= self.config['task']['stationary_max_displacement_m']
                                and abs(info['heading_error_rad']) <= self.config['task']['stationary_max_heading_rad']
                                and info['planar_speed_m_s'] <= self.config['task']['linear_tracking_sigma_m_s']
                                and abs(info['yaw_rate_rad_s']) <= self.config['task']['yaw_tracking_sigma_rad_s']
                                and info['tilt_rad'] <= self.getup['tilt_tolerance_rad']
                                and self.hold_max_displacement <= self.getup['max_displacement_m'])
            if self.reset_mode == 'standing':
                no_push_good = bool(reached and info.get('is_success', False))
            success = self.push.success(reached, no_push_good, self.hold_max_displacement, self.hold_max_heading,
                                        info['planar_displacement_m'], info['heading_error_rad'],
                                        info['planar_speed_m_s'], info['yaw_rate_rad_s'], info['tilt_rad'], self.config['task']) if self.push.enabled else no_push_good
            info['combined_success'] = info['is_success'] = bool(success)
            if self.push.enabled:
                info['push_recovery_success'] = bool(success)
            info['getup_metrics'] = dict(reset_mode=self.reset_mode, reset_length_m=self.reset_length_m,
                                         acquired=self.acquired, getup_seconds=info['getup_seconds'],
                                         combined_success=bool(success), guided_reference=True)
            info['episode_metrics']['reward'] = self._total_reward
            info['episode_metrics']['getup_metrics'] = info['getup_metrics']
        return self._observation(), float(reward), done, truncated, info
