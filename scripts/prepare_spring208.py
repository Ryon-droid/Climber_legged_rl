"""Rebuild the measured 208 mm upper stop and a reachable belly-down reset."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'scripts')]
import mujoco
import numpy as np
from scipy.optimize import least_squares
import yaml
from climber_legged_rl.cad_dynamics import build_dynamics, validate_artifacts
from climber_legged_rl.getup import prepare_pose_bank
from climber_legged_rl.mujoco_env import ClimberMujocoEnv
from climber_legged_rl.mujoco_model import closure_error, load_config
from climber_legged_rl.mujoco_training import validate_policy_transfer
from climber_legged_rl.spring_geometry import eye_center_distance
from climber_legged_rl.stance_geometry import virtual_leg_geometry
from smoke_mujoco import run as smoke

CONFIG = ROOT/'configs/robot/cad_balance_spring208.yaml'
PARENT = ROOT/'outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_config(config):
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


def build_preserving_standing(source_folder):
    """Keep the parent standing solution, including unobservable passive spins."""
    xml, generated = build_dynamics(CONFIG)
    original = ET.parse(source_folder/'robot_experiment.xml').getroot()
    root = ET.parse(xml).getroot()
    for attribute in ('pos', 'quat'):
        root.find('./worldbody/body').set(attribute, original.find('./worldbody/body').get(attribute))
    root.remove(root.find('keyframe'))
    root.append(copy.deepcopy(original.find('keyframe')))
    ET.indent(root)
    ET.ElementTree(root).write(xml, encoding='utf-8', xml_declaration=True)
    # Unconstrained passive spin coordinates can differ when rerunning IK. Retain
    # the exact original reset so the migration changes only the two stop ranges.
    previous = json.loads((source_folder/'model_report.json').read_text())
    for key in ('pose_attempts', 'rest_height_hip_above_wheel_m', 'standing_base_z_m',
                'closure_error_m', 'chassis_axes_in_base', 'standing_geometry', 'motor_center_evidence'):
        if key in previous:
            generated[key] = previous[key]
    generated['generated_sha256'] = sha(xml)
    generated['spring208_provenance'] = dict(parent_model_sha256=sha(source_folder/'robot_experiment.xml'),
        exact_parent_standing_keyframe_preserved=True, only_two_slider_upper_ranges_changed=True)
    (xml.parent/'model_report.json').write_text(json.dumps(generated, indent=2)+'\n')
    return xml, generated


def main():
    source_config = ROOT/'configs/robot/cad_balance_estimator_round15.yaml'
    original = load_config(source_config)
    source_folder = ROOT/original['experiment']['output_directory']
    old_pose = json.loads((source_folder/'getup_poses.json').read_text())['poses'][0]
    config = copy.deepcopy(original)
    config['experiment']['output_directory'] = 'outputs/cad_balance_spring208'
    config['experiment']['approximation_notes'] = [
        note for note in original['experiment']['approximation_notes']
        if '155-220' not in note and '376.210431' not in note]
    config['experiment']['approximation_notes'] += [
        'User confirmed maximum mounting-eye center distance 208 mm on 2026-11-06; '
        'the inherited 155 mm minimum and nominal 450 N extension force are retained.',
        'Prone pose is recalculated under the 208 mm upper stop; '
        'the previous 376.210431 mm rearward offset is historical, not the new reset.',
        'The numerical stop compliance is measured in validation, not identified from hardware.']
    config['spring']['eye_center_limits_m'] = {side: [.155, .208] for side in ('left', 'right')}
    config['training']['allow_spring_limit_warm_start'] = True
    config['training']['spring_limit_migration'] = dict(mode='tighten_upper_stop',
        previous_eye_center_limits_m=copy.deepcopy(original['spring']['eye_center_limits_m']))
    # Old supervised trajectories have a different physical model fingerprint.
    config['training']['standing_rehearsal']['enabled'] = False
    config['training']['standing_rehearsal'].pop('dataset')
    config['training']['standing_rehearsal'].pop('dataset_sha256')
    save_config(config)
    xml, _ = build_preserving_standing(source_folder)
    env = ClimberMujocoEnv(xml, CONFIG)
    try:
        model, data = env.model, env.data
        initial = np.array(old_pose['qpos'])
        wheel_q = set(env.q_indices[4:].tolist())
        indices = [i for i in range(7, model.nq) if i not in wheel_q]
        lows, highs = [], []
        for i in indices:
            j = int(np.flatnonzero(model.jnt_qposadr == i)[0])
            lows.append(model.jnt_range[j, 0] if model.jnt_limited[j] else -np.inf)
            highs.append(model.jnt_range[j, 1] if model.jnt_limited[j] else np.inf)
        lows, highs = np.array(lows), np.array(highs)
        meshes = []
        for geom in range(model.ngeom):
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[geom]
                start = model.mesh_vertadr[mid]
                meshes.append((geom, model.mesh_vert[start:start+model.mesh_vertnum[mid]].copy()))

        def geometry():
            return virtual_leg_geometry(model, data, env.motor_midpoints, env.wheel_centers, env.chassis_axes)

        def residual(values):
            data.qpos[:] = initial
            data.qpos[indices] = values
            mujoco.mj_forward(model, data)
            bottoms = {g: float((v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min())
                       for g, v in meshes if g in env.wheel_geom_ids or g in env.base_geom_ids}
            support_difference = min(v for g, v in bottoms.items() if g in env.wheel_geom_ids) - min(
                v for g, v in bottoms.items() if g in env.base_geom_ids)
            eyes = [eye_center_distance(model, data, s) for s in ('left', 'right')]
            return np.r_[closure_error(model, data).ravel()*3000,
                         support_difference*30,
                         (np.array(eyes)-(.208-1e-6))*300,
                         (data.qpos[env.q_indices[:2]]-data.qpos[env.q_indices[2:4]])*2,
                         (values-initial[indices])*1e-5]

        fit = least_squares(residual, np.clip(initial[indices], lows+1e-9, highs-1e-9),
                            bounds=(lows, highs), max_nfev=1200, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        residual(fit.x)
        error = float(np.linalg.norm(closure_error(model, data), axis=1).max())
        if not fit.success or error > 1e-4 or max(abs(residual(fit.x)[:18])) > .3:
            raise RuntimeError('New upper-stop prone solve did not converge')
        config['getup']['rearward_offsets_m'] = [float(np.mean(geometry()['forward_offset_m']))]
    finally:
        env.close()
    save_config(config)
    xml, generated = build_preserving_standing(source_folder)
    poses = prepare_pose_bank(CONFIG)
    smoke(CONFIG, steps=1000)
    validate_artifacts(CONFIG)
    validate_policy_transfer(PARENT, xml, config, warm_start=True)
    new_pose = poses['poses'][0]
    report = dict(simulation_only=True, prepared=True, config=str(CONFIG),
        parent_checkpoint=str(PARENT), parent_checkpoint_sha256=sha(PARENT),
        previous_config=str(source_config), previous_config_sha256=sha(source_config),
        previous_model_sha256=sha(source_folder/'robot_experiment.xml'), model_sha256=sha(xml),
        previous_pose=old_pose, new_pose=new_pose, total_mass_kg=generated['total_mass_kg'],
        spring_travel=generated['spring_travel'], old_rehearsal_disabled=True,
        strict_warm_start_validated=True, training_started=False,
        note='Existing actor evaluation is model transfer, not evidence of real-hardware readiness.')
    (xml.parent/'spring208_preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(prepared=True, rearward_offset_m=new_pose['rearward_offset_m'],
        spring_eye_distances_m=new_pose['spring_eye_distances_m'], closure_error_m=new_pose['closure_error_m'],
        base_bottom_m=new_pose['base_bottom_m'], wheel_bottom_m=new_pose['wheel_bottom_m']), indent=2), flush=True)


if __name__ == '__main__':
    main()
