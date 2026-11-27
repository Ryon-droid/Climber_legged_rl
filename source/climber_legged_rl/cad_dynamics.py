"""Construct floating-base dynamics from the repaired CAD, without mass rescaling."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .cad_model import ROOT, sha256, closure_vectors
from .mujoco_model import load_config

DEFAULT_DYNAMICS_CONFIG = ROOT / 'configs/robot/cad_dynamics.yaml'


def input_hashes(config_path):
    config = load_config(config_path)
    sources = [ROOT/config['experiment']['source_model'],
               ROOT/'configs/robot/measurements/rigid_bodies.csv',
               ROOT/'configs/robot/mechanical_measurements.yaml',
               ROOT/'source/climber_legged_rl/cad_dynamics.py',
               ROOT/'source/climber_legged_rl/cad_repair.py']
    if config['physics'].get('chassis_reference_mesh'):
        sources.append(ROOT/'source/climber_legged_rl/chassis_frame.py')
    if config['physics'].get('standing_distance_definition'):
        sources.append(ROOT/'source/climber_legged_rl/stance_geometry.py')
    if config.get('task', {}).get('velocity_tracking'):
        sources.append(ROOT/'source/climber_legged_rl/velocity_env.py')
    if config.get('push', {}).get('enabled'):
        sources.append(ROOT/'source/climber_legged_rl/push_disturbance.py')
    if config.get('getup', {}).get('enabled'):
        sources.append(ROOT/'source/climber_legged_rl/getup.py')
        if config['getup'].get('dynamic_control',{}).get('enabled'):
            sources.append(ROOT/'source/climber_legged_rl/dynamic_getup.py')
        if config['getup'].get('support_control',{}).get('enabled'):
            sources.append(ROOT/'source/climber_legged_rl/supported_getup.py')
        if config['getup'].get('policy_control',{}).get('enabled'):
            sources.append(ROOT/'source/climber_legged_rl/policy_getup.py')
        if config['getup'].get('folded_reset_noise',{}).get('enabled'):
            sources.append(ROOT/'source/climber_legged_rl/robust_getup.py')
        if config['getup'].get('reset_curriculum',{}).get('enabled'):
            sources.append(ROOT/'source/climber_legged_rl/getup_curriculum.py')
    if config.get('getup', {}).get('enabled') or config['spring'].get('eye_center_limits_m'):
        sources.append(ROOT/'source/climber_legged_rl/spring_geometry.py')
    if config.get('observations') or config.get('robustness'):
        sources.append(ROOT/'source/climber_legged_rl/observation_history.py')
    if config.get('training', {}).get('history_standing_actor'):
        sources.append(ROOT/'source/climber_legged_rl/history_policy.py')
    if config.get('balance_recovery', {}).get('enabled'):
        sources.append(ROOT/'source/climber_legged_rl/balance_rewards.py')
    if config.get('training', {}).get('velocity_estimator', {}).get('enabled'):
        sources.append(ROOT/'source/climber_legged_rl/velocity_estimation.py')
    if config.get('training', {}).get('standing_rehearsal', {}).get('enabled'):
        sources.append(ROOT/'source/climber_legged_rl/standing_rehearsal.py')
        dataset = ROOT/config['training']['standing_rehearsal']['dataset']
        sources.extend([dataset, dataset.with_suffix('.json'), dataset.parent/'config.yaml'])
    if config.get('training', {}).get('hard_push_resets', {}).get('enabled'):
        sources.append(ROOT/'source/climber_legged_rl/push_reset_replay.py')
    if config.get('training', {}).get('hard_standing_resets', {}).get('enabled'):
        sources.append(ROOT/'source/climber_legged_rl/standing_reset_replay.py')
    if config['experiment'].get('physics_backend') == 'mujoco_warp':
        sources.append(ROOT/'source/climber_legged_rl/warp_backend.py')
    return {str(p.relative_to(ROOT)):sha256(p) for p in sources}


def validate_artifacts(config_path=DEFAULT_DYNAMICS_CONFIG):
    """Reject stale generation/smoke reports before starting training workers."""
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    folder = ROOT/config['experiment']['output_directory']
    report = json.loads((folder/'model_report.json').read_text())
    smoke = json.loads((folder/'smoke_report.json').read_text())
    if not smoke.get('passed') or report['input_sha256'] != input_hashes(config_path):
        raise RuntimeError('CAD dynamics inputs or checks changed; rebuild and rerun smoke checks')
    files = {'config_sha256':config_path, 'model_sha256':folder/'robot_experiment.xml',
             'env_sha256':ROOT/'source/climber_legged_rl/mujoco_env.py'}
    if any(smoke.get(key)!=sha256(path) for key,path in files.items()):
        raise RuntimeError('CAD dynamics smoke report is stale')
    if report['config_sha256'] != sha256(config_path) or report['generated_sha256'] != sha256(folder/'robot_experiment.xml'):
        raise RuntimeError('CAD dynamics generation report is stale')
    if smoke.get('input_sha256') != input_hashes(config_path):
        raise RuntimeError('CAD dynamics smoke input hashes are stale')
    if config.get('getup', {}).get('enabled'):
        from .getup import validate_pose_bank
        validate_pose_bank(config_path)
        if config['getup'].get('reset_curriculum',{}).get('enabled'):
            from .getup_curriculum import validate_curriculum_bank
            validate_curriculum_bank(config_path,folder)
    return report, smoke


def geom_vertices(model, data, geom_id):
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh_id]
    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]]
    return vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]


def solve_ground_pose(model, initial, config, chassis_frame=None):
    """Seek a closed, symmetric pose with wheel bottoms below CAD body geometry."""
    data = mujoco.MjData(model)
    wheels = [model.body(s + '_link 3').id for s in ('left', 'right')]
    hips = [model.body(s + '_link1').id for s in ('left', 'right')]
    wheel_q = [int(model.joint(n).qposadr[0]) for n in config['control']['wheel_joints']]
    indices = [i for i in range(model.nq) if i not in wheel_q]
    lows = np.array([model.jnt_range[i, 0] if model.jnt_limited[i] else -np.inf for i in indices])
    highs = np.array([model.jnt_range[i, 1] if model.jnt_limited[i] else np.inf for i in indices])
    radius = config['physics']['wheel_radius_m']
    leg_q = [int(model.joint(n).qposadr[0]) for n in config['control']['leg_joints']]
    base_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body('base').id]
    mujoco.mj_forward(model, data)
    physics = config['physics']
    use_midpoint = physics.get('standing_distance_definition') == 'motor_midpoint_to_wheel_center_side_view'
    if use_midpoint:
        from .stance_geometry import motor_midpoints_in_base, wheel_centers_in_wheel_bodies, virtual_leg_geometry
        from .chassis_frame import frame_from_mounting_plate
        midpoint, _ = motor_midpoints_in_base(model, physics)
        wheel_centers = wheel_centers_in_wheel_bodies(model, physics)
        if chassis_frame is None:
            ref_mesh = physics.get('chassis_reference_mesh')
            if ref_mesh:
                try:
                    chassis_frame = frame_from_mounting_plate(model, ref_mesh)
                except ValueError:
                    pass
            if chassis_frame is None:
                # Identity frame: base axes are world axes (source CAD is already level).
                chassis_frame = np.eye(3)
        frame = chassis_frame
        requested = physics['rest_motor_midpoint_to_wheel_center_m']
    else:
        requested = physics['rest_height_hip_above_wheel_m']
    attempts = []
    best = None
    base_id = model.body('base').id
    # Positive com_forward_offset_m places the standing COM ahead of the wheel axle
    # (chassis fore-aft), trading some static balance for forward-acceleration margin.
    com_offset = float(physics.get('com_forward_offset_m', 0.0))
    fore_aft_axes = np.eye(3) if chassis_frame is None else chassis_frame

    def fore_aft_error():
        """COM offset ahead of the wheel axle, measured in the chassis frame."""
        rotation = data.xmat[base_id].reshape(3, 3) @ fore_aft_axes
        com = rotation.T @ (data.subtree_com[base_id] - data.xpos[base_id])
        wheel = rotation.T @ ((data.xpos[wheels] - data.xpos[base_id]).mean(axis=0))
        return float(wheel[0] - com[0])

    # A confirmed physical length must not silently fall back to a different stance.
    for height in ((requested,) if use_midpoint else (requested, .25, .32, .38, .42)):
        def residual(values):
            data.qpos[:] = initial
            data.qpos[indices] = values
            mujoco.mj_forward(model, data)
            heights = data.xpos[hips, 2] - data.xpos[wheels, 2]
            forward_offset = np.zeros(2)
            if use_midpoint:
                geometry = virtual_leg_geometry(model,data,midpoint,wheel_centers,frame)
                heights = geometry['side_view_distance_m']
                forward_offset = geometry['forward_offset_m']
            q = data.qpos[leg_q]
            # Constrain both COM balance and the physical leg geometry in the
            # mounting-plate frame; distance alone also admits reversed stances.
            return np.r_[closure_vectors(model, data).ravel() * 2000,
                         (heights - height) * 20,
                         (fore_aft_error() + com_offset) * 5,
                         (q[:2] - q[2:]) * 2, values * 1e-6]
        fit = least_squares(residual, np.clip(initial[indices], lows+1e-9, highs-1e-9),
                            bounds=(lows, highs), max_nfev=1400,
                            ftol=1e-11, xtol=1e-11, gtol=1e-10)
        residual(fit.x)
        closure = float(np.linalg.norm(closure_vectors(model, data), axis=1).max())
        actual = data.xpos[hips, 2] - data.xpos[wheels, 2]
        forward_offset = np.zeros(2)
        if use_midpoint:
            geometry = virtual_leg_geometry(model,data,midpoint,wheel_centers,frame)
            actual = geometry['side_view_distance_m']
            forward_offset = geometry['forward_offset_m']
        base_bottom = min(geom_vertices(model, data, g)[:, 2].min() for g in base_geoms)
        body_clearance = float(base_bottom - data.xpos[wheels, 2].mean() + radius)
        com_error = fore_aft_error()
        record = {'requested_hip_wheel_m': height, 'actual_hip_wheel_m': actual.tolist(),
                  'closure_error_m': closure, 'base_bottom_clearance_m': body_clearance,
                  'com_wheel_x_error_m': com_error,
                  'motor_midpoint_wheel_forward_offset_m': forward_offset.tolist()}
        attempts.append(record)
        if use_midpoint:
            record['distance_definition'] = physics['standing_distance_definition']
            record['requested_motor_midpoint_to_wheel_center_m'] = height
            record['actual_motor_midpoint_to_wheel_center_m'] = actual.tolist()
            del record['requested_hip_wheel_m'], record['actual_hip_wheel_m']
        height_tolerance = 1e-6 if use_midpoint else .003
        if (closure < 1e-4 and max(abs(actual-height)) < height_tolerance
                and max(abs(forward_offset)) < .01 and abs(com_error + com_offset) < .005):
            best = data.qpos.copy()
            break
        initial = data.qpos.copy()
    if best is None:
        raise RuntimeError('No CAD ground-clear reset found: ' + json.dumps(attempts))
    return best, attempts


def build_dynamics(config_path=DEFAULT_DYNAMICS_CONFIG):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    if config['experiment'].get('model_route') != 'cad_dynamics':
        raise ValueError('Expected cad_dynamics configuration')
    from .spring_geometry import configured_eye_limits, apply_eye_limits
    configured_eye_limits(config, required=config.get('getup', {}).get('enabled', False))
    source = ROOT / config['experiment']['source_model']
    source_hash = sha256(source)
    root = ET.parse(source).getroot()
    fixed = mujoco.MjModel.from_xml_path(str(source))
    base_quaternion = np.array([0., 0., 0., 1.])
    chassis_frame = None
    reference_mesh = config['physics'].get('chassis_reference_mesh')
    solver_quaternion = np.array([0., 0., 0., 1.])
    if reference_mesh:
        from .chassis_frame import horizontal_base_quaternion
        try:
            solver_quaternion, chassis_frame = horizontal_base_quaternion(fixed, reference_mesh)
        except ValueError as exc:
            import warnings
            warnings.warn(f'chassis_reference_mesh {reference_mesh!r} rejected: {exc}; '
                          f'falling back to identity base quaternion for solver')
            reference_mesh = None
        else:
            fixed_base = root.find('./worldbody/body')
            fixed_base.attrib.pop('euler', None)
            fixed_base.set('quat', ' '.join(format(v, '.15g') for v in solver_quaternion))
            fixed = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    original_masses = {fixed.body(i).name: float(fixed.body_mass[i]) for i in range(1, fixed.nbody)}
    qpos = fixed.key_qpos[0].copy()
    qpos, attempts = solve_ground_pose(fixed, qpos, config, chassis_frame=chassis_frame)
    # Joint angles and the base quaternion form one pose: preserve the exact
    # CAD-root orientation used by the solver so the mounting plate stays level.
    base_quaternion = fixed.body_quat[fixed.body('base').id].copy()
    final_root = root
    final_fixed_base = final_root.find('./worldbody/body')
    final_fixed_base.attrib.pop('euler', None)
    final_fixed_base.set('quat', ' '.join(format(v, '.15g') for v in base_quaternion))
    data = mujoco.MjData(fixed)
    data.qpos[:] = qpos
    mujoco.mj_forward(fixed, data)
    physics = config['physics']
    control = config['control']
    radius = physics['wheel_radius_m']
    # The cad_repaired XML dumps every leg mesh into the base body too, so
    # min(all_bottoms) would pick up a phantom leg-part vertex far below the
    # actual chassis plate.  Instead, raise the base so the wheel bottoms
    # just touch the ground; the original solver-quaternion path had to
    # account for those duplicate meshes, but the level-quaternion artefact
    # only cares that the real wheel contact lands at z=0.
    # Find the largest (outer tire) geom in each wheel body FIRST, before
    # computing z.  Using wheel body xpos - radius was wrong because the
    # body frame origin is offset ~80mm from the actual tire contact surface.
    # Use the collision geom's own position and size instead.
    wheel_collision_ids = {}
    for side in ('left', 'right'):
        body_name = f"{side}_link 3"
        body_geoms = [g for g in range(fixed.ngeom)
                      if fixed.geom_bodyid[g] == mujoco.mj_name2id(fixed, mujoco.mjtObj.mjOBJ_BODY, body_name)]
        if not body_geoms:
            raise RuntimeError(f"No geoms in wheel body {body_name}")
        # Pick the geom with the largest z-extent (outer tire).
        largest = max(body_geoms, key=lambda g: float(np.ptp(geom_vertices(fixed, data, g), axis=0)[2]))
        wheel_collision_ids[side] = largest
    wheel_lowest = min(
        float(geom_vertices(fixed, data, wheel_collision_ids[s])[:, 2].min())
        for s in ('left', 'right')
    )
    z = -wheel_lowest + physics['reset_clearance_m']
    # Recheck COM balance in the physical mounting-plate frame before adding
    # the free joint; refuse to emit a stance the wheels cannot hold.
    level_base = fixed.body('base').id
    level_axes = np.eye(3) if chassis_frame is None else chassis_frame
    level_rotation = data.xmat[level_base].reshape(3, 3) @ level_axes
    level_origin = data.xpos[level_base]
    level_com = level_rotation.T @ (data.subtree_com[level_base] - level_origin)
    level_wheels = level_rotation.T @ (
        (data.xpos[[fixed.body(s + '_link 3').id for s in ('left', 'right')]] - level_origin).mean(axis=0)
    )
    level_com_error = float(level_wheels[0] - level_com[0])
    com_offset = float(physics.get('com_forward_offset_m', 0.0))
    if abs(level_com_error + com_offset) > 0.005:
        raise RuntimeError(
            f'Level standing pose leaves the COM {level_com_error * 1000:.1f} mm off the wheel axle'
        )
    root = final_root
    base = root.find('./worldbody/body')
    base.attrib.pop('euler', None)
    base.set('pos', f'0 0 {z:.15g}')
    base.set('quat', ' '.join(format(v, '.15g') for v in base_quaternion))
    base.insert(0, ET.Element('freejoint', name='floating_base'))
    for element in root.findall('actuator') + root.findall('keyframe'):
        root.remove(element)
    option = root.find('option')
    option.set('gravity', '0 0 -9.81')
    option.set('timestep', str(physics['timestep']))
    option.set('integrator', 'implicitfast')
    option.set('iterations', str(physics['solver_iterations']))
    option.set('solver', 'Newton')
    option.set('jacobian', 'dense')
    world = root.find('worldbody')
    ET.SubElement(world, 'geom', name='ground', type='plane', size='10 10 .1',
                  contype='1', conaffinity='2', condim='3', friction='.8 .005 .0001',
                  rgba='.55 .6 .65 1')
    # Mesh-ground contacts retain CAD surfaces. Robot-robot contact is disabled
    # because assembled closed-chain neighbours intentionally touch.
    for geom in root.findall('.//body/geom'):
        geom.set('contype', '2'); geom.set('conaffinity', '1')
        geom.set('condim', '3'); geom.set('friction', '.8 .005 .0001')
        geom.set('solref', '.01 1'); geom.set('group', '1')
    # Keep one named outer tire mesh per wheel for contact diagnostics.
    for side in ('left', 'right'):
        body = root.find(f'.//body[@name="{side}_link 3"]')
        geoms = body.findall('geom')
        largest = max(geoms, key=lambda g: float(np.ptp(geom_vertices(fixed, data, fixed.geom(g.get('name')).id), axis=0)[2]))
        largest.set('name', side + '_wheel_collision')
        largest.set('friction',f"{physics['wheel_friction']} .005 .0001")
    # Everything except the explicit wheel tire collides with nothing.  The
    # cad_repaired XML piles every leg mesh into the base body too, so those
    # phantom vertices sit far below the chassis plate in the level artefact.
    # Set inactive geoms to (0, 0) so MuJoCo skips them entirely.  The active
    # wheel tire keeps (2, 1) to match ground (1, 2):
    #   ground.conaffinity(2) & wheel.contype(2) = 2  ✓
    #   wheel.conaffinity(1)  & ground.contype(1) = 1  ✓
    for geom in root.findall('.//body/geom'):
        geom.set('contype', '0'); geom.set('conaffinity', '0')
    for side in ('left', 'right'):
        coll = root.find(f'.//body/geom[@name="{side}_wheel_collision"]')
        if coll is not None:
            coll.set('contype', '2'); coll.set('conaffinity', '1')
    # Env uses a set of all base mesh geoms; name one for legacy API compatibility.
    base.find('geom').set('name', 'base_collision')
    active = set(control['leg_joints'] + control['wheel_joints'])
    for joint in root.findall('.//body/joint'):
        joint.set('damping', str(physics['joint_damping']))
        joint.set('armature', str(physics['active_armature'] if joint.get('name') in active else physics['passive_armature']))
    # Apply opt-in get-up ranges AFTER solving the unchanged standing keyframe.
    # They replace CAD exporter branch guards in the explicit get-up task and in the
    # velocity stage that keeps the same 36-input standing action map.
    overrides = physics.get('joint_range_overrides_rad', {})
    preserved_standing = config.get('task', {}).get('velocity_observation_layout') == 'getup_standing'
    if overrides and not (config.get('getup', {}).get('enabled') or preserved_standing):
        raise ValueError('Joint range overrides require the explicit get-up task')
    for name, bounds in overrides.items():
        joint = root.find(f'.//body/joint[@name="{name}"]')
        if (joint is None or joint.get('type', 'hinge') != 'hinge'
                or len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]):
            raise ValueError('Invalid get-up joint range override: ' + name)
        joint.set('limited', 'true')
        joint.set('range', ' '.join(str(v) for v in bounds))
    actuator = ET.SubElement(root, 'actuator')
    for i, name in enumerate(control['leg_joints'] + control['wheel_joints']):
        limit = control['leg_torque_limit_Nm'] if i < 4 else control['wheel_torque_limit_Nm']
        ET.SubElement(actuator,'motor',name=f'motor_{i}',joint=name,gear='1',
                      ctrllimited='true',ctrlrange=f'{-limit} {limit}')
    spring = config['spring']
    spring_travel = apply_eye_limits(root, fixed, qpos, config)
    if spring['model'] != 'constant_extension_force_at_reconstructed_pin_sites' or not np.isfinite(spring['force_N']) or spring['force_N'] < 0:
        raise ValueError('Unsupported spring model or invalid nominal extension force')
    for side in ('left', 'right'):
        # The repaired slider's positive axis increases eye distance. A constant
        # generalized slider force therefore models nominal extension push.
        joint = root.find(f'.//joint[@name="{side}_gas_slide_joint"]')
        joint.set('springref', '0')
        joint.set('stiffness', '0')
    full_qpos = np.r_[[0, 0, z], base_quaternion, qpos]
    keys = ET.SubElement(root,'keyframe')
    ET.SubElement(keys,'key',name='standing',qpos=' '.join(format(v,'.15g') for v in full_qpos))
    root.set('model','Climber_repaired_CAD_dynamics')
    output = ROOT / config['experiment']['output_directory']
    output.mkdir(parents=True,exist_ok=True)
    path = output / 'robot_experiment.xml'
    ET.indent(root); ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model,data,0); mujoco.mj_forward(model,data)
    masses = {model.body(i).name: float(model.body_mass[i]) for i in range(1,model.nbody)}
    if masses != original_masses or model.nu != 6 or model.neq != 6:
        raise RuntimeError('CAD masses, six motors or six closures were not preserved')
    closure = float(np.linalg.norm(closure_vectors(model,data),axis=1).max())
    if closure > 1e-4:
        raise RuntimeError('Floating-base reset closure failed')
    tire_bottom = min(
        float(geom_vertices(model, data, model.geom(f'{s}_wheel_collision').id)[:, 2].min())
        for s in ('left', 'right')
    )
    if abs(tire_bottom - physics['reset_clearance_m']) > 0.002:
        raise RuntimeError(
            f'Standing keyframe tire bottom is {tire_bottom * 1000:.1f} mm, '
            f'expected {physics["reset_clearance_m"] * 1000:.1f} mm above the ground'
        )
    keyframe_com = model.body('base').id
    keyframe_axes = np.eye(3) if chassis_frame is None else chassis_frame
    keyframe_rotation = data.xmat[keyframe_com].reshape(3, 3) @ keyframe_axes
    keyframe_com_local = keyframe_rotation.T @ (data.subtree_com[keyframe_com] - data.xpos[keyframe_com])
    keyframe_wheels = keyframe_rotation.T @ (
        (data.xpos[[model.body(s + '_link 3').id for s in ('left', 'right')]] - data.xpos[keyframe_com]).mean(axis=0)
    )
    keyframe_com_error = float(keyframe_wheels[0] - keyframe_com_local[0])
    if abs(keyframe_com_error + com_offset) > 0.005:
        raise RuntimeError(f'Standing keyframe COM is {keyframe_com_error * 1000:.1f} mm off the wheel axle')
    hip_height = float(np.mean([data.body(s+'_link1').xpos[2]-data.body(s+'_link 3').xpos[2] for s in ('left','right')]))
    report = {'simulation_only':True,'model_route':'cad_dynamics','generated_model':str(path),
              'source_sha256':source_hash,'config_sha256':sha256(config_path),
              'generated_sha256':sha256(path),'builder_sha256':sha256(__file__),
              'total_mass_kg':float(model.body_mass.sum()),'body_masses_kg':masses,
              'nq':model.nq,'nv':model.nv,'nu':model.nu,'closures':model.neq,
              'collision_mesh_count':len(root.findall('.//body/geom')),
              'pose_attempts':attempts,'rest_height_hip_above_wheel_m':hip_height,
              'standing_base_z_m':z,'closure_error_m':closure,
              'level_com_wheel_x_error_m':level_com_error,
              'keyframe_com_wheel_x_error_m':keyframe_com_error,
              'tire_ground_clearance_m':tire_bottom,
              'chassis_reference_mesh':reference_mesh,
              'chassis_axes_in_base':chassis_frame.tolist() if chassis_frame is not None else None,
              'spring':spring,'approximations':config['experiment']['approximation_notes'],
              'joint_range_overrides_rad':overrides,
              'spring_travel':spring_travel,
              'input_sha256':input_hashes(config_path),
              'hardware_ready':False}
    if physics.get('standing_distance_definition'):
        from .stance_geometry import motor_midpoints_in_base, wheel_centers_in_wheel_bodies, virtual_leg_geometry
        midpoint, evidence = motor_midpoints_in_base(model,physics)
        wheel_centers = wheel_centers_in_wheel_bodies(model,physics)
        rep_frame = chassis_frame if chassis_frame is not None else np.eye(3)
        geometry = virtual_leg_geometry(model,data,midpoint,wheel_centers,rep_frame)
        target_distance = physics['rest_motor_midpoint_to_wheel_center_m']
        if max(abs(geometry['side_view_distance_m'] - target_distance)) >= 1e-6:
            raise RuntimeError('Standing keyframe does not preserve the requested physical leg distance')
        if max(abs(geometry['forward_offset_m'])) >= .01:
            raise RuntimeError('Standing keyframe has excessive motor-midpoint/wheel-center forward offset')
        report['standing_geometry'] = {k:v.tolist() for k,v in geometry.items()}
        report['standing_geometry']['definition'] = physics['standing_distance_definition']
        report['standing_geometry']['target_distance_m'] = physics['rest_motor_midpoint_to_wheel_center_m']
        report['motor_center_evidence'] = evidence
    (output/'model_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if sha256(source) != source_hash:
        raise RuntimeError('Repaired source changed during model generation')
    return path, report
