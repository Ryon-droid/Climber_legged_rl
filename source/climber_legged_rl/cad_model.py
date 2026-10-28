"""Audit original CAD geometry and inertia without inventing physical parameters."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.optimize import least_squares
import yaml
from .physical_inputs import load_body_masses

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAD_CONFIG = ROOT / 'configs/robot/cad_model.yaml'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnostic_tree(source):
    source = Path(source).resolve()
    root = ET.parse(source).getroot()
    compiler = root.find('compiler')
    compiler.set('balanceinertia', 'false')
    compiler.set('boundinertia', '0')
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', str((source.parent / mesh.get('file')).resolve()))
        if 'surface_shell' in mesh.get('name', ''):
            mesh.set('inertia', 'shell')
    for actuator in root.findall('actuator'):
        root.remove(actuator)
    for joint in root.findall('.//body/joint'):
        joint.attrib.pop('actuatorfrcrange', None)
        joint.attrib.pop('actuatorfrclimited', None)
    for geom in root.findall('.//body/geom'):
        geom.set('contype', '0')
        geom.set('conaffinity', '0')
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('gravity', '0 0 0')
    return root


def closure_vectors(model, data):
    return np.array([data.site_xpos[model.eq_obj1id[i]]-data.site_xpos[model.eq_obj2id[i]]
                     for i in range(model.neq) if model.eq_type[i] == mujoco.mjtEq.mjEQ_CONNECT])


def motion_probe(model, settings, leg_names, wheel_names):
    """Prescribe one motor; hold other motors; solve passive hinge positions."""
    data = mujoco.MjData(model)
    motor_indices = [int(model.joint(n).qposadr[0]) for n in leg_names+wheel_names]
    passive = np.array([i for i in range(model.nq) if i not in motor_indices])
    lows = np.array([model.jnt_range[i, 0] if model.jnt_limited[i] else -np.pi for i in passive])
    highs = np.array([model.jnt_range[i, 1] if model.jnt_limited[i] else np.pi for i in passive])
    initial = model.qpos0.copy()
    records = []
    for name in leg_names:
        index = int(model.joint(name).qposadr[0])
        for delta in settings['motion_test_deltas_rad']:
            prescribed = initial.copy()
            prescribed[index] += delta
            def residual(values):
                data.qpos[:] = prescribed
                data.qpos[passive] = values
                mujoco.mj_forward(model, data)
                return closure_vectors(model, data).ravel()
            result = least_squares(residual, np.clip(initial[passive], lows+1e-9, highs-1e-9),
                                   bounds=(lows, highs), max_nfev=settings['solver_max_nfev'],
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12)
            residual(result.x)
            error = float(np.linalg.norm(closure_vectors(model, data), axis=1).max())
            records.append({'joint': name, 'delta_rad': delta, 'max_closure_error_m': error,
                            'passed': bool(result.success and error <= settings['closure_tolerance_m']),
                            'solver_success': bool(result.success),
                            'hip_wheel_vertical_m': {s: float(data.body(s+'_link1').xpos[2]-data.body(s+'_link 3').xpos[2]) for s in ('left','right')},
                            'qpos_rad': data.qpos.tolist()})
    return {'passed': all(r['passed'] for r in records), 'samples': records,
            'note': 'Local bounded solve at CAD zero with other motors held. Not global reachability proof or physical calibration.'}


def audit_model(config_path=DEFAULT_CAD_CONFIG, run_motion=True):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    if cfg['model_policy'] != 'source_cad_only':
        raise ValueError('CAD route requires source_cad_only')
    source = ROOT/cfg['source_model']
    original_hash = sha256(source)
    output = ROOT/cfg['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    root = diagnostic_tree(source)
    path = output/'robot_cad_kinematic.xml'
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    measurements = yaml.safe_load((ROOT/cfg['mechanical_measurements']).read_text())
    target = measurements['mass']['total_mass_for_simulation_kg']
    target_side = measurements['mass'].get('side_total_kg')
    confirmed_bodies = load_body_masses()
    user = json.loads((ROOT/cfg['source_assembly']).read_text())
    parts = json.loads((ROOT/cfg['source_parts']).read_text())['parts']
    assigned = {p for link in user['links'] for p in link.get('part_ids', [])}
    unassigned = [p for p in parts if p['part_id'] not in assigned]
    (output/'unassigned_parts.json').write_text(json.dumps(unassigned, ensure_ascii=False, indent=2))
    with (output/'rigid_body_inventory.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = ['body', 'source_mass_kg', 'source_local_com_m', 'source_inertia_quat_wxyz',
                  'source_diaginertia_kg_m2', 'assigned_part_count', 'confirmed_mass_kg',
                  'confirmed_local_com_m', 'confirmed_fullinertia_kg_m2', 'evidence']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for body in root.findall('.//body'):
            inertial = body.find('inertial')
            if inertial is None:
                continue
            link = next((link for link in user['links'] if link['name'] == body.get('name')), {})
            writer.writerow({'body': body.get('name'), 'source_mass_kg': inertial.get('mass'),
                             'source_local_com_m': inertial.get('pos'),
                             'source_inertia_quat_wxyz': inertial.get('quat', '1 0 0 0'),
                             'source_diaginertia_kg_m2': inertial.get('diaginertia'),
                             'assigned_part_count': len(link.get('part_ids', [])),
                             'confirmed_mass_kg': confirmed_bodies.get(body.get('name')),
                             'evidence': 'configs/robot/measurements/rigid_bodies.csv' if body.get('name') in confirmed_bodies else ''})
    mass = float(model.body_mass.sum())
    sides = {}
    base = root.find('./worldbody/body')
    for side in ('left','right'):
        bodies = [b for branch in base.findall('body') if branch.get('name','').startswith(side) for b in [branch,*branch.findall('.//body')]]
        value = sum(float(b.find('inertial').get('mass')) for b in bodies if b.find('inertial') is not None)
        sides[side] = {'export_kg': value, 'confirmed_kg': target_side,
                       'missing_kg': target_side-value if target_side is not None else None}
    errors = np.linalg.norm(closure_vectors(model,data),axis=1)
    jacobians = []
    for i in range(model.neq):
        ja = np.zeros((3,model.nv)); jb = ja.copy()
        mujoco.mj_jacSite(model,data,ja,None,model.eq_obj1id[i])
        mujoco.mj_jacSite(model,data,jb,None,model.eq_obj2id[i])
        jacobians.extend(ja-jb)
    _, singular, vt = np.linalg.svd(np.array(jacobians),full_matrices=True)
    rank = int(np.sum(singular > cfg['audit']['jacobian_rank_tolerance_m_per_rad']))
    nullspace = vt[rank:].T
    tangent = {}
    for side,names in zip(('left','right'),(cfg['expected_leg_joints'][:2],cfg['expected_leg_joints'][2:])):
        indices = [int(model.joint(n).dofadr[0]) for n in names]
        tangent[side] = np.linalg.svd(nullspace[indices],compute_uv=False).tolist()
    wheel_geometry = {}
    for side,joint in zip(('left','right'),cfg['expected_wheel_joints']):
        j = model.joint(joint).id
        body_id = model.jnt_bodyid[j]
        meshes = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != body_id or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = model.geom_dataid[g]
            start,count = model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start+count]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
            delta = vertices-data.xanchor[j]
            axial = delta@data.xaxis[j]
            radius = np.linalg.norm(delta-np.outer(axial,data.xaxis[j]),axis=1)
            meshes.append({'geom':model.geom(g).name,'max_radius_m':float(radius.max()),'axial_min_m':float(axial.min()),'axial_max_m':float(axial.max())})
        wheel_geometry[side] = {'axis_in_source_world':data.xaxis[j].tolist(),
                                'outermost_mesh':max(meshes,key=lambda x:x['max_radius_m']),
                                'source':'CAD mesh vertices; not friction/deflection measurement'}
    blockers = []
    if target is not None and abs(mass-target)>cfg['audit']['mass_tolerance_kg']:
        blockers.append(f'CAD mass {mass:.9f} kg differs from confirmed {target:g} kg; no scaling applied.')
    mass_differences = []
    for name, confirmed in confirmed_bodies.items():
        body = root.find(f'.//body[@name="{name}"]')
        actual = float(body.find('inertial').get('mass')) if body is not None and body.find('inertial') is not None else None
        if actual is None or abs(actual-confirmed)>1e-8:
            mass_differences.append({'body':name,'export_kg':actual,'confirmed_kg':confirmed})
    if mass_differences:
        blockers.append('Source CAD body masses differ from the latest confirmed inputs; use the repaired candidate with mass overrides.')
    if unassigned:
        blockers.append(f'{len(unassigned)} CAD parts are unassigned; account for each before dynamics release.')
    if not errors.size or errors.max()>cfg['audit']['closure_tolerance_m']:
        blockers.append('CAD zero-pose closure failed.')
    motion = motion_probe(model,cfg['audit'],cfg['expected_leg_joints'],cfg['expected_wheel_joints']) if run_motion else {'passed':False,'skipped':True}
    if not motion['passed']:
        blockers.append('Local independent motor sweep did not pass all samples; review topology/assembly branch.')
    blockers.extend('Unverified: '+key for key,value in cfg['verification'].items() if value is not True)
    report = {'model_policy':cfg['model_policy'],'ready_for_training':False,'training_started':False,'hardware_ready':False,
              'scope':'CAD kinematic diagnostic only; no physical dynamics release',
              'source_sha256':original_hash,'diagnostic_sha256':sha256(path),'config_sha256':sha256(config_path),
              'audit_code_sha256':sha256(__file__),'mujoco_version':mujoco.__version__,
              'input_sha256':{key:sha256(ROOT/cfg[key]) for key in ('source_assembly','source_parts','source_export_report','mechanical_measurements')},
              'preserved':['body/joint transforms','explicit inertials','CAD mesh geometry','six closures','exported ranges'],
              'adapter_changes':['absolute mesh paths','shell compile mode','strict inertia validation','export actuator removal','contact/gravity disabled for kinematic audit'],
              'export_mass_kg':mass,'confirmed_mass_kg':target,
              'mass_gap_kg':target-mass if target is not None else None,'side_masses':sides,
              'confirmed_body_masses_kg':confirmed_bodies,'body_mass_differences':mass_differences,
              'body_mass_inputs_sha256':sha256(ROOT/'configs/robot/measurements/rigid_bodies.csv'),
              'unassigned_part_count':len(unassigned),'nq':model.nq,'nv':model.nv,'nu':model.nu,
              'base_fixed_for_kinematic_audit':True,'closure_error_at_cad_zero_m':errors.tolist(),
              'constraint_jacobian_rank':rank,'local_motor_tangent_singular_values':tangent,
              'wheel_geometry':wheel_geometry,'motion_probe':motion,'blockers':blockers,
              'release_note':'Audit cannot release training by booleans alone. Implement and validate physical dynamics, stance, contacts and springs first.'}
    if sha256(source)!=original_hash:
        raise RuntimeError('Source model changed during audit')
    (output/'audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report
