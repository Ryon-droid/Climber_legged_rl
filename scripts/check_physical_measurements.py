"""Validate measured body tensors and spring records; never fill missing values."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import numpy as np
import xml.etree.ElementTree as ET
from climber_legged_rl.cad_model import ROOT
from climber_legged_rl.physical_inputs import load_body_masses


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def check(directory):
    directory=Path(directory);blockers=[];complete=[];total=0.
    body_path=directory/'rigid_bodies.csv'
    if not body_path.exists():return {'ready_for_dynamics_identification':False,'blockers':['Missing rigid_bodies.csv']}
    rows=read_csv(body_path)
    try:
        supplied_masses = load_body_masses(directory)
    except (ValueError, TypeError) as exc:
        supplied_masses = {}
        blockers.append('Invalid body mass inputs: ' + str(exc))
    names=[r['body'] for r in rows]
    model_path=ROOT/'outputs/cad_repaired/robot_cad_repaired.xml'
    expected_names={b.get('name') for b in ET.parse(model_path).getroot().findall('.//body')}
    if set(names)!=expected_names:blockers.append('Body inventory does not match current repaired CAD model')
    if len(set(names))!=len(names):blockers.append('Duplicate body names')
    for r in rows:
        name=r['body'];fields=['mass_kg','com_x_m','com_y_m','com_z_m','ixx_kg_m2','iyy_kg_m2','izz_kg_m2','ixy_kg_m2','ixz_kg_m2','iyz_kg_m2']
        missing = [f for f in fields if not r[f].strip()]
        if missing:
            blockers.append('Missing body fields: ' + name + ': ' + ', '.join(missing))
            continue
        try:
            a=np.array([float(r[f]) for f in fields])
            mass,x,y,z,xx,yy,zz,xy,xz,yz=a
            I=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
            eigen=np.linalg.eigvalsh(I)
            if not np.isfinite(a).all() or mass<=0 or eigen.min()<=0 or eigen.max()>eigen.sum()-eigen.max()+1e-10:
                raise ValueError('mass must be positive; inertia must be finite positive definite and satisfy triangle inequality')
            if r['scope']!='complete_body_including_attached_parts':raise ValueError('unsupported mass scope')
            if not r['source'].strip():raise ValueError('source/evidence missing')
            if r['reference_frame']!='MJCF_body_local_at_COM':raise ValueError('unsupported COM/inertia frame')
            complete.append(name);total+=mass
        except (ValueError,TypeError) as exc:blockers.append(name+': '+str(exc))
    dimensions=directory/'spring_dimensions.csv';samples=directory/'spring_samples.csv'
    if dimensions.exists():
        dim=read_csv(dimensions)
        for side in ('left','right'):
            matches=[r for r in dim if r['side']==side]
            if len(matches)!=1:blockers.append('Need one dimension row per side: '+side);continue
            r=matches[0]
            try:
                low=float(r['usable_min_eye_distance_m']);high=float(r['usable_max_eye_distance_m'])
                extended=float(r['extended_eye_center_distance_m']);compressed=float(r['compressed_eye_center_distance_m'])
                if not np.isfinite([low,high,extended,compressed]).all() or not 0<compressed<=low<high<=extended:raise ValueError('invalid measured travel')
                if r['force_direction'] not in ('extension_push','retraction_pull'):raise ValueError('unknown force direction')
                if not r['source'].strip():raise ValueError('missing source')
            except (TypeError,ValueError) as exc:blockers.append('Spring dimensions '+side+': '+str(exc))
    else:blockers.append('Missing spring_dimensions.csv')
    if samples.exists():
        all_samples=read_csv(samples)
        for side in ('left','right'):
            for direction in ('compression','extension'):
                records=[r for r in all_samples if r['side']==side and r['motion_direction']==direction]
                try:
                    vals=np.array([[float(r['mount_distance_m']),float(r['axial_force_N'])] for r in records])
                    if len(vals)<3 or not np.isfinite(vals).all() or len(np.unique(vals[:,0]))<3 or (vals<=0).any():raise ValueError('need >=3 distinct positive finite length/force samples')
                    if any(not r['source'].strip() for r in records):raise ValueError('source missing')
                except (ValueError,TypeError) as exc:blockers.append('Spring curve '+side+'/'+direction+': '+str(exc))
    else:blockers.append('Missing spring_samples.csv')
    if (directory/'unassigned_parts.csv').exists():
        part_rows=read_csv(directory/'unassigned_parts.csv')
        if len(part_rows)!=72:blockers.append('Need all 72 remaining CAD parts in the review inventory')
        unresolved=[r['part_id'] for r in part_rows if not r['confirmed_body'].strip() or r['included_in_mass_scope'] not in ('yes','excluded_with_evidence') or not r['evidence'].strip()]
        if unresolved:blockers.append(f'{len(unresolved)} remaining CAD parts lack body/mass scope/evidence')
    else:blockers.append('Missing unassigned_parts.csv')
    return {'ready_for_dynamics_identification':not blockers,'hardware_ready':False,'completed_body_count':len(complete),'expected_body_count':len(rows),
            'completed_mass_sum_kg':total,
            'provided_mass_count':len(supplied_masses),'provided_mass_subtotal_kg':math.fsum(supplied_masses.values()),
            'provided_body_masses_kg':supplied_masses,
            'all_body_masses_provided':bool(rows) and len(supplied_masses)==len(rows),
            'blockers':blockers,'note':'Masses are accepted independently of pending COM/inertia. Complete inputs permit further model construction, not training/hardware release.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=ROOT/'configs/robot/measurements');p.add_argument('--output',type=Path,default=ROOT/'outputs/cad_repaired/physical_inputs_report.json');a=p.parse_args()
    r=check(a.directory);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))
    raise SystemExit(0 if r['ready_for_dynamics_identification'] else 2)
