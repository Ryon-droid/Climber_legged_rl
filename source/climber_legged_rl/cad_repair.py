"""Reconstruct CAD gas-spring geometry as a hinge/slider diagnostic candidate.

No hardware stroke/force or 15 kg mass distribution is invented. Explicit
spring inertials are recomputed using the source export's inferred density,
strictly to preserve the provenance of the uncalibrated geometric model.
"""
from __future__ import annotations
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
import yaml
from .cad_model import ROOT, diagnostic_tree, closure_vectors, sha256

DEFAULT_REPAIR_CONFIG = ROOT/'configs/robot/cad_repair.yaml'


def numbers(values):
    return ' '.join(f'{float(x):.15g}' for x in values)


def stl_triangles(path):
    raw=Path(path).read_bytes()
    count=struct.unpack_from('<I',raw,80)[0]
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    if len(raw)!=84+50*count:
        raise ValueError(f'Expected binary STL: {path}')
    return np.frombuffer(raw,dtype=dtype,offset=84,count=count)['vertices'].astype(float)*0.001


def mesh_inertial(triangles,density,origin):
    vertices=triangles-origin
    volume=np.einsum('ij,ij->i',vertices[:,0],np.cross(vertices[:,1],vertices[:,2]))/6
    total=float(volume.sum())
    if total<=0:
        raise ValueError('Solid CAD STL must have positive signed volume')
    first=(volume[:,None]*vertices.sum(axis=1)/4).sum(axis=0)
    sums=vertices.sum(axis=1)
    second=(np.einsum('ni,nj->nij',sums,sums)+np.einsum('nki,nkj->nij',vertices,vertices))
    second=np.einsum('n,nij->ij',volume,second)/20
    com=first/total
    central=second-total*np.outer(com,com)
    inertia=density*(np.trace(central)*np.eye(3)-central)
    if np.linalg.eigvalsh(inertia).min()<=0:
        raise ValueError('CAD solid integration produced invalid inertia')
    return ET.Element('inertial',mass=f'{density*total:.15g}',pos=numbers(com),
                      fullinertia=numbers([inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]))


def fit_axis(triangles):
    points=np.unique(triangles.reshape(-1,3),axis=0)
    _,_,vt=np.linalg.svd(points-points.mean(axis=0),full_matrices=False)
    axis=vt[0]
    if axis[2]<0:axis=-axis
    return axis


def solve_positions(model, prescribed=None, height=None, initial=None, continuity=False):
    data=mujoco.MjData(model)
    motor_names=['left_link1_joint','left_link_rear_1_joint','right_link1_joint','right_rear_link1_joint']
    wheels=['left_link 3_joint','right_link 3_joint']
    fixed={int(model.joint(n).qposadr[0]):0.0 for n in wheels}
    if prescribed is not None:
        fixed.update({int(model.joint(n).qposadr[0]):float(v) for n,v in prescribed.items()})
    indices=np.array([i for i in range(model.nq) if i not in fixed])
    low=[];high=[]
    for i in indices:
        limited=bool(model.jnt_limited[i])
        low.append(model.jnt_range[i,0] if limited else -np.inf)
        high.append(model.jnt_range[i,1] if limited else np.inf)
    initial=model.qpos0.copy() if initial is None else initial.copy()
    passive_pin_indices=[int(model.joint(n).qposadr[0]) for n in
                         ('base_linkother2_joint','base_linkother_joint','right_other 1_joint','right_other 1_joint_2')]
    # These source helper bodies have their only closure point on their own
    # hinge axis; its free spin is not set by a connect constraint. Keep their
    # previous orientation during prescribed playback, without changing physics.
    free_positions=[int(np.where(indices==i)[0][0]) for i in passive_pin_indices if i in indices]
    if continuity:
        for pos in free_positions:
            low[pos]=max(low[pos],initial[indices[pos]]-1e-6)
            high[pos]=min(high[pos],initial[indices[pos]]+1e-6)
    def residual(values):
        data.qpos[:]=initial
        data.qpos[indices]=values
        for i,v in fixed.items():data.qpos[i]=v
        mujoco.mj_forward(model,data)
        values=[closure_vectors(model,data).ravel()*1000]
        if height is not None:
            values.append(np.array([data.body(s+'_link1').xpos[2]-data.body(s+'_link 3').xpos[2]-height for s in ('left','right')])*1000)
            q=[data.qpos[int(model.joint(n).qposadr[0])] for n in motor_names]
            values.append((np.array(q[:2])-q[2:])*0.1)
        if continuity:
            values.append((data.qpos[indices]-initial[indices])*0.001)
        return np.concatenate(values)
    fit=least_squares(residual,np.clip(initial[indices],np.array(low)+1e-10,np.array(high)-1e-10),
                      bounds=(low,high),max_nfev=1200,ftol=1e-12,xtol=1e-12,gtol=1e-11)
    residual(fit.x)
    return data,fit


def build_repaired(config_path=DEFAULT_REPAIR_CONFIG):
    config_path=Path(config_path).resolve();cfg=yaml.safe_load(config_path.read_text())
    if cfg['training_permitted'] is not False:raise ValueError('Uncalibrated repair is diagnostic only')
    source=ROOT/cfg['source_model'];source_hash=sha256(source)
    source_input_hashes={key:sha256(ROOT/cfg[key]) for key in ('source_model','source_assembly','source_parts','mechanical_measurements')}
    root=diagnostic_tree(source)
    original=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    original_data=mujoco.MjData(original);mujoco.mj_forward(original,original_data)
    assembly=json.loads((ROOT/cfg['source_assembly']).read_text())
    parts={p['part_id']:p for p in json.loads((ROOT/cfg['source_parts']).read_text())['parts']}
    links={l['name']:l for l in assembly['links']}
    triangles=lambda pid:stl_triangles(source.parent/parts[pid]['mesh_file'])
    # Left pin centers are existing CAD joint axes. Move along each pin axis to
    # the eye's median plane (does not change the mechanical pivot axis).
    lower=original_data.body('base_splink').xpos.copy()
    upper=original_data.body('base_spring link 2').xpos.copy()
    base_origin=np.array(assembly['base_origin_xyz'])*0.001
    center_y=float(np.median(triangles('part_583').reshape(-1,3)[:,1])-base_origin[1])
    lower[1]=center_y;upper[1]=center_y
    axis=(upper-lower)/np.linalg.norm(upper-lower)
    axis_evidence={}
    for side,tube,rod in [('left','part_582','part_581'),('right','part_434','part_433')]:
        tube_axis=fit_axis(triangles(tube));rod_axis=fit_axis(triangles(rod))
        axis_evidence[side]={'cylinder_axis':tube_axis.tolist(),'rod_axis':rod_axis.tolist(),
                             'mount_axis_alignment_abs_dot':float(abs(tube_axis@axis)),
                             'rod_cylinder_alignment_abs_dot':float(abs(tube_axis@rod_axis))}
        if abs(tube_axis@axis)<0.99999 or abs(tube_axis@rod_axis)<0.99999:
            raise ValueError('CAD spring mesh axes disagree with existing pin centers')
    # closure_003 was placed halfway along the left rear coupler and off
    # the tiny pin body. Match the physical pin axes already present in CAD;
    # right closure_r_005 supplies the symmetric endpoint topology.
    repaired_anchor=original_data.body('base_linkother2').xpos.copy()
    left_eq=next(eq for eq in root.find('equality') if eq.get('name')=='closure_003')
    old_site=original.site(left_eq.get('site1')).id
    repaired_anchor[1]=original_data.site_xpos[old_site,1]
    closure_patch={'constraint':'closure_003','reason':'Left midpoint/off-pin anchor disagrees with right endpoint topology and existing CAD pin axes.',
                   'original_anchor_m':original_data.site_xpos[old_site].tolist(),'repaired_anchor_m':repaired_anchor.tolist(),
                   'site_positions':{}}
    for site_name in [left_eq.get('site1'),left_eq.get('site2')]:
        site=root.find(f'.//site[@name="{site_name}"]')
        sid=original.site(site_name).id;bid=original.site_bodyid[sid]
        R=original_data.xmat[bid].reshape(3,3)
        local=R.T@(repaired_anchor-original_data.xpos[bid])
        site.set('pos',numbers(local));closure_patch['site_positions'][site_name]=local.tolist()
    removed=['base_splink','base_spring link 2','right_spring_link1','right_spring_link2']
    for body in list(root.findall('.//body')):
        for child in list(body.findall('body')):
            if child.get('name') in removed:body.remove(child)
    # Retain constraint names but connect the actual rod pin to the upper leg.
    for eq in root.find('equality'):
        if eq.get('name') in ('closure_007','closure_008'):
            side='left' if eq.get('name')=='closure_008' else 'right'
            eq.set('site1',side+'_gas_rod_pin');eq.set('site2',side+'_gas_upper_mount')
    # Remove stale pair exclusions that mention replaced assemblies.
    contact=root.find('contact')
    for ex in list(contact):
        if ex.get('body1') in removed or ex.get('body2') in removed:contact.remove(ex)
    assets={m.get('name'):m for m in root.findall('./asset/mesh')}
    mapping=[];geometry_checks=[]
    for side in ('left','right'):
        mirror=np.diag([1,-1,1]) if side=='right' else np.eye(3)
        lo=mirror@lower;hi=mirror@upper;direction=(hi-lo)/np.linalg.norm(hi-lo)
        hip_name=cfg['spring_parts'][side]['lower_mount_body'];upper_name=cfg['spring_parts'][side]['upper_mount_body']
        hip=root.find(f'.//body[@name="{hip_name}"]');leg=root.find(f'.//body[@name="{upper_name}"]')
        h_id=original.body(hip_name).id;u_id=original.body(upper_name).id
        R=original_data.xmat[h_id].reshape(3,3);U=original_data.xmat[u_id].reshape(3,3)
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,np.ascontiguousarray(R.T).ravel())
        cylinder=ET.SubElement(hip,'body',name=side+'_gas_cylinder',pos=numbers(R.T@(lo-original_data.xpos[h_id])),quat=numbers(quat))
        ET.SubElement(cylinder,'joint',name=side+'_gas_pin_joint',type='hinge',axis='0 1 0',limited='false',damping='0',armature='0')
        rod=ET.SubElement(cylinder,'body',name=side+'_gas_rod',pos=numbers(hi-lo))
        ET.SubElement(rod,'joint',name=side+'_gas_slide_joint',type='slide',axis=numbers(direction),limited='false',damping='0',armature='0')
        ET.SubElement(rod,'site',name=side+'_gas_rod_pin',pos='0 0 0',size='.002',rgba='1 0.4 0 1')
        ET.SubElement(leg,'site',name=side+'_gas_upper_mount',pos=numbers(U.T@(hi-original_data.xpos[u_id])),size='.002',rgba='0 1 0 1')
        groups=[(cylinder,'cylinder_and_lower_eye',lo),(rod,'rod_and_upper_eye',hi)]
        for body,key,pivot in groups:
            ids=cfg['spring_parts'][side][key]
            # Retain source density basis, explicitly not measured steel/aluminium.
            reference=links['base_splink'] if side=='left' else links['right_spring_link1']
            density=reference['mass']/reference['mass_properties']['volume_m3']
            body.insert(1,mesh_inertial(np.concatenate([triangles(pid) for pid in ids]),density,base_origin+pivot))
            for pid in ids:
                mesh_name=next((n for n,a in assets.items() if Path(a.get('file')).name==Path(parts[pid]['mesh_file']).name),None)
                if mesh_name is None:
                    mesh_name=pid+'_restored_cad';ET.SubElement(root.find('asset'),'mesh',name=mesh_name,file=str((source.parent/parts[pid]['mesh_file']).resolve()),scale='0.001 0.001 0.001')
                ET.SubElement(body,'geom',name=side+'_'+pid+'_cad',type='mesh',mesh=mesh_name,pos=numbers(-base_origin-pivot),group='1',contype='0',conaffinity='0')
                mapping.append({'part_id':pid,'body':body.get('name'),'source_density_kg_m3':density,'mass_status':'source_geometry_density_not_physical_calibration'})
        geometry_checks.append({'side':side,'lower_pin_source_m':lo.tolist(),'upper_pin_source_m':hi.tolist(),
                                'pin_distance_at_cad_zero_m':float(np.linalg.norm(hi-lo)),
                                'axis_from_cad_pins':direction.tolist()})
    output=ROOT/cfg['output_directory'];output.mkdir(parents=True,exist_ok=True)
    root.set('model','climber_CAD_REPAIRED_KINEMATIC_CANDIDATE')
    world=root.find('worldbody')
    ET.SubElement(world,'light',name='cad_review_key',pos='0 -1 2',dir='0 0 -1',directional='true',diffuse='.9 .9 .9',ambient='.3 .3 .3')
    ET.SubElement(world,'light',name='cad_review_fill',pos='0 1 1',dir='0 -1 -1',directional='true',diffuse='.6 .6 .6')
    ET.indent(root);path=output/'robot_cad_repaired.xml';ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    zero_error=float(np.linalg.norm(closure_vectors(model,data),axis=1).max())
    if zero_error>1e-6:raise ValueError(f'Reconstructed CAD zero closure error {zero_error}')
    # Validate world mesh locations at the original assembly pose.
    for entry in mapping:
        pid=entry['part_id'];gid=model.geom(entry['body'].split('_gas_')[0]+'_'+pid+'_cad').id
        mid=model.geom_dataid[gid];v=model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
        original_vertices=triangles(pid).reshape(-1,3)-base_origin
        err=float(max(np.max(np.abs(v.min(0)-original_vertices.min(0))),np.max(np.abs(v.max(0)-original_vertices.max(0)))))
        entry['cad_zero_mesh_bbox_error_m']=err
        if err>2e-7:raise ValueError(f'Geometry transform error for {pid}: {err}')
    motor_names=['left_link1_joint','left_link_rear_1_joint','right_link1_joint','right_rear_link1_joint']
    motion=[]
    for name in motor_names:
        for delta in [-.1,-.05,-.02,.02,.05,.1]:
            values={n:(delta if n==name else 0) for n in motor_names}
            d,fit=solve_positions(model,prescribed=values)
            error=float(np.linalg.norm(closure_vectors(model,d),axis=1).max())
            motion.append({'joint':name,'delta_rad':delta,'max_closure_error_m':error,'passed':bool(fit.success and error<1e-4),
                           'slide_displacements_m':{s:float(d.qpos[int(model.joint(s+'_gas_slide_joint').qposadr[0])]) for s in ('left','right')}})
    measurements=yaml.safe_load((ROOT/cfg['mechanical_measurements']).read_text())
    height=measurements['geometry']['nominal_standing_pose']['hip_to_wheel_vertical_separation_m']
    stance,fit=solve_positions(model,height=height)
    heights=[float(stance.body(s+'_link1').xpos[2]-stance.body(s+'_link 3').xpos[2]) for s in ('left','right')]
    stance_error=float(np.linalg.norm(closure_vectors(model,stance),axis=1).max())
    stance_report={'target_hip_wheel_vertical_m':height,'hip_wheel_vertical_m':heights,'max_closure_error_m':stance_error,
                   'passed':bool(fit.success and stance_error<1e-4 and max(abs(h-height) for h in heights)<.003),
                   'qpos':stance.qpos.tolist(),'motor_positions_rad':{n:float(stance.qpos[int(model.joint(n).qposadr[0])]) for n in motor_names},
                   'spring_pin_distances_m':{s:float(np.linalg.norm(stance.site(s+'_gas_rod_pin').xpos-stance.body(s+'_gas_cylinder').xpos)) for s in ('left','right')},
                   'note':'Numerical pose only. Unknown physical spring travel and encoder zeros remain unresolved.'}
    pose=root.find('keyframe')
    if pose is not None:root.remove(pose)
    if stance_report['passed']:
        keys=ET.SubElement(root,'keyframe');ET.SubElement(keys,'key',name='cad_135mm_candidate',qpos=numbers(stance.qpos))
        ET.indent(root);ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    report={'model_policy':cfg['model_policy'],'ready_for_training':False,'training_started':False,'hardware_ready':False,
            'source_input_sha256':source_input_hashes,'source_sha256':source_hash,'source_assembly_sha256':sha256(ROOT/cfg['source_assembly']),
            'source_parts_sha256':sha256(ROOT/cfg['source_parts']),'config_sha256':sha256(config_path),'code_sha256':sha256(__file__),
            'generated_model':str(path),'generated_sha256':sha256(path),'geometry_source':'actual original CAD STL, including restored right rod part_433',
            'repair_evidence':['user-confirmed bilateral symmetry','left original CAD pin axes','spring solid PCA axis alignment','four paired source STL groups'],
            'closure_anchor_repair':closure_patch,'geometry_axis_checks':axis_evidence,'mounts':geometry_checks,'part_mapping':mapping,
            'zero_pose_max_closure_error_m':zero_error,'motion_samples':motion,'motion_passed':all(r['passed'] for r in motion),
            'stance':stance_report,'nq':model.nq,'nu':model.nu,'closure_count':model.neq,
            'source_export_mass_kg':float(original.body_mass.sum()),'candidate_mass_kg':float(model.body_mass.sum()),
            'confirmed_mass_kg':measurements['mass']['total_mass_for_simulation_kg'],
            'restored_part_ids':['part_433'],'remaining_unassigned_part_count':len(set(parts)-{pid for link in assembly['links'] for pid in link['part_ids']}-{'part_433'}),
            'blockers':['Per-body mass, COM and inertia are not physically reconciled to 15 kg.',
                        'Spring travel, force-length curve, damping and mount reconstruction still require physical/CAD review.',
                        'Remaining 72 unassigned parts require body/material review.',
                        'Contacts, floating-base dynamics and actual six-channel actuator envelopes are not validated.'],
            'kinematic_only_settings':{'base_fixed':True,'gravity_disabled':True,'collisions_disabled':True,'slide_limits_unknown':True,'spring_force_omitted':True},
            'scope':'Repair candidate for kinematics, with actual CAD geometry; never a certified 15 kg training model.'}
    if sha256(source)!=source_hash:raise RuntimeError('Original CAD was modified')
    (output/'repair_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (output/'status.json').write_text(json.dumps({'state':'kinematic_candidate_validated' if report['motion_passed'] and stance_report['passed'] else 'kinematic_validation_failed','training_permitted':False,'source_preserved':True},indent=2))
    return path,report
