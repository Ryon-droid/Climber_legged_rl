"""Offline expert labels and actor-only imitation for the unassisted get-up task.

State copying is confined to training data generation. The deployed student env
never imports this module or calls the teacher controller.
"""
from __future__ import annotations

import time
import mujoco
import numpy as np
import torch


def copy_training_state(source, target):
    target.data.qpos[:] = source.data.qpos
    target.data.qvel[:] = source.data.qvel
    target.data.time = source.data.time
    mujoco.mj_forward(target.model, target.data)
    for name in ('steps', 'acquired', 'reset_mode', 'reset_length_m', 'reset_hip_height',
                 'hold_heading', 'acquisition_step', 'settled_steps'):
        setattr(target, name, getattr(source, name))
    target.hold_position = source.hold_position.copy()
    target.commands[:] = source.commands


def expert_label(student, teacher, teacher_policy):
    """Convert teacher PD targets into the student's different rising action map."""
    teacher_action, _ = teacher_policy.predict(teacher._observation(), deterministic=True)
    executed = np.clip(teacher.assisted_action(teacher_action), -1., 1.)
    teacher._set_available_torque()
    student._set_motor_envelope()
    if student.acquired:
        label = executed.copy()
    else:
        q = student.data.qpos[student.q_indices[:4]]
        speed = student.data.qvel[student.v_indices[:4]]
        target = teacher._leg_targets(executed)
        if (student.control['leg_kp'] == teacher.control['leg_kp']
                and student.control['leg_kd'] == teacher.control['leg_kd']
                and np.array_equal(student.torque_limits[:4], teacher.torque_limits[:4])):
            delta = target - q
        else:
            torque = teacher.control['leg_kp']*(target-q)-teacher.control['leg_kd']*speed
            torque = np.clip(torque, -teacher.torque_limits[:4], teacher.torque_limits[:4])
            delta = (torque+student.control['leg_kd']*speed)/student.control['leg_kp']
        label = np.r_[delta/student.dynamic['incremental_target_scale_rad'], executed[4:]]
    return np.clip(label, -1., 1.), executed


def sample_weight(env):
    return 20. if not env.acquired and env.steps*env.dt < .6 else 5. if not env.acquired else 1.


def demonstrations(student, teacher, teacher_policy, seconds=6., standing_episodes=6, augmentation_samples=0,
                   focus_offset=None, focus_weight=1.):
    observations, labels, weights = [], [], []
    source_poses, target_poses = teacher.pose_bank, student.pose_bank
    specs = [('folded', p['rearward_offset_m'], 99000+i) for i,p in enumerate(source_poses)]
    specs += [('standing', None, 99100+i) for i in range(standing_episodes)]
    successes = []
    rng = np.random.default_rng(109)
    try:
        for mode, offset, seed in specs:
            if offset is not None:
                teacher.pose_bank = [p for p in source_poses if p['rearward_offset_m'] == offset]
                student.pose_bank = [p for p in target_poses if p['rearward_offset_m'] == offset]
            teacher.reset(seed=seed, options={'reset_mode':mode})
            student.reset(seed=seed, options={'reset_mode':mode})
            episode_data = []
            for _ in range(round(seconds/student.dt)):
                copy_training_state(teacher, student)
                observation = student._observation()
                label, _ = expert_label(student, teacher, teacher_policy)
                episode_data.append((observation, label, sample_weight(student)))
                qpos, qvel, previous_action = teacher.data.qpos.copy(), teacher.data.qvel.copy(), teacher.action.copy()
                student_previous = student.action.copy()
                for _ in range(augmentation_samples):
                    teacher.data.qpos[:] = qpos
                    teacher.data.qvel[:] = qvel
                    rotation = rng.normal(0., .006, 3)
                    angle = np.linalg.norm(rotation)
                    delta = np.r_[np.cos(angle/2), np.sin(angle/2)*rotation/max(angle, 1e-12)]
                    mujoco.mju_mulQuat(teacher.data.qpos[3:7], delta, qpos[3:7])
                    teacher.data.qpos[:2] += rng.normal(0., .002, 2)
                    # Synthetic encoder variations expose target-position feedback.
                    # These observations are labelled only; no physics is advanced here.
                    q_indices = teacher.q_indices[:4]
                    teacher.data.qpos[q_indices] = np.clip(qpos[q_indices]+rng.normal(0., .03, 4),
                                                          teacher.leg_low,teacher.leg_high)
                    teacher.data.qvel[:6] += rng.normal(0., .02, 6)
                    teacher.data.qvel[teacher.v_indices] += rng.normal(0., .05, 6)
                    teacher.action[:] = np.clip(previous_action+rng.normal(0., .01, 6), -1., 1.)
                    mujoco.mj_forward(teacher.model, teacher.data)
                    copy_training_state(teacher, student)
                    student.action[:] = np.clip(student_previous+rng.normal(0., .01, 6), -1., 1.)
                    augmented_label, _ = expert_label(student, teacher, teacher_policy)
                    episode_data.append((student._observation(), augmented_label, sample_weight(student)))
                teacher.data.qpos[:] = qpos
                teacher.data.qvel[:] = qvel
                teacher.action[:] = previous_action
                mujoco.mj_forward(teacher.model, teacher.data)
                action, _ = teacher_policy.predict(teacher._observation(), deterministic=True)
                _, _, done, truncated, info = teacher.step(action)
                student.action[:] = label
                if done or truncated:
                    break
            success = bool(teacher.acquired and not done)
            successes.append(dict(mode=mode, offset_m=offset, success=success))
            if success:
                multiplier = focus_weight if offset is not None and focus_offset is not None and abs(offset-focus_offset)<1e-8 else 1.
                for observation, label, weight in episode_data:
                    observations.append(observation); labels.append(label); weights.append(weight*multiplier)
    finally:
        teacher.pose_bank, student.pose_bank = source_poses, target_poses
    if not observations or not all(s['success'] for s in successes if s['mode']=='folded'):
        raise RuntimeError('All three expert prone demonstrations must succeed')
    return observations, labels, weights, successes


def collect_student_states(student, teacher, teacher_policy, policy, seconds=6., seed=99200, teacher_mix=0.,
                           focus_offset=None, focus_weight=1.):
    observations, labels, weights = [], [], []
    source_poses, target_poses = teacher.pose_bank, student.pose_bank
    outcomes = []
    try:
        for index, pose in enumerate(target_poses):
            offset = pose['rearward_offset_m']
            student.pose_bank = [pose]
            teacher.pose_bank = [p for p in source_poses if p['rearward_offset_m']==offset]
            obs, _ = student.reset(seed=seed+index, options={'reset_mode':'folded'})
            teacher.reset(seed=seed+index, options={'reset_mode':'folded'})
            done = truncated = False
            for _ in range(round(seconds/student.dt)):
                copy_training_state(student, teacher)
                label, executed = expert_label(student, teacher, teacher_policy)
                multiplier=focus_weight if focus_offset is not None and abs(offset-focus_offset)<1e-8 else 1.
                observations.append(obs.copy()); labels.append(label); weights.append(sample_weight(student)*multiplier)
                action, _ = policy.predict(obs, deterministic=True)
                action = (1.-teacher_mix)*np.clip(action, -1., 1.)+teacher_mix*label
                obs, _, done, truncated, info = student.step(action)
                teacher.action[:] = executed
                if done or truncated:
                    break
            outcomes.append(dict(offset_m=offset, acquired=bool(student.acquired), seconds=student.steps*student.dt,
                                 reason=info['termination_reason'], final_position_m=info['planar_displacement_m']))
    finally:
        teacher.pose_bank, student.pose_bank = source_poses, target_poses
    return observations, labels, weights, outcomes


def fit_actor(policy, observations, labels, weights, updates, batch_size, learning_rate, seed):
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=policy.device)
    y = torch.as_tensor(np.asarray(labels, dtype=np.float32), device=policy.device)
    probabilities = np.asarray(weights, dtype=float); probabilities /= probabilities.sum()
    optimizer = torch.optim.Adam(policy.policy.parameters(), lr=learning_rate)
    started = time.monotonic()
    last_loss = None
    for update in range(updates):
        for group in optimizer.param_groups:
            group['lr'] = learning_rate*(1.-.8*update/updates)
        indices = rng.choice(len(x), size=batch_size, p=probabilities)
        mean = policy.policy.get_distribution(x[indices]).distribution.mean
        loss = torch.mean((mean-y[indices])**2)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.policy.parameters(), 1.)
        optimizer.step()
        last_loss = float(loss.detach())
        if (update+1)%250 == 0 or update+1 == updates:
            remaining = (time.monotonic()-started)*(updates-update-1)/(update+1)
            print(f'IMITATION update={update+1}/{updates} loss={last_loss:.7f} remaining_seconds={remaining:.0f}', flush=True)
    return last_loss
