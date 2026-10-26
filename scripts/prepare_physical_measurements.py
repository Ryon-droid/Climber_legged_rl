"""Create non-overwriting physical identification templates from the CAD candidate."""
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import numpy as np
from climber_legged_rl.cad_model import ROOT


def rows_csv(path,fields,rows):
    if path.exists():return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def prepare():
    folder=ROOT/'configs/robot/measurements';folder.mkdir(parents=True,exist_ok=True)
    source=ROOT/'outputs/cad_repaired/robot_cad_repaired.xml'
    if not source.exists():raise RuntimeError('Run repair_cad_model.py first')
    root=ET.parse(source).getroot()
    fields=['body','scope','mass_kg','com_x_m','com_y_m','com_z_m','ixx_kg_m2','iyy_kg_m2','izz_kg_m2','ixy_kg_m2','ixz_kg_m2','iyz_kg_m2','reference_frame','source','measured_at']
    rows=[{'body':b.get('name'),'scope':'complete_body_including_attached_parts','reference_frame':'MJCF_body_local_at_COM'} for b in root.findall('.//body')]
    rows_csv(folder/'rigid_bodies.csv',fields,rows)
    rows_csv(folder/'spring_samples.csv',['side','mount_distance_m','axial_force_N','motion_direction','speed_m_s','temperature_C','source','measured_at'],[{'side':side,'motion_direction':direction} for side in ('left','right') for direction in ('compression','extension') for _ in range(3)])
    rows_csv(folder/'spring_dimensions.csv',['side','extended_eye_center_distance_m','compressed_eye_center_distance_m','usable_min_eye_distance_m','usable_max_eye_distance_m','mass_kg','force_direction','source','measured_at'],[{'side':s} for s in ('left','right')])
    parts=json.loads((ROOT/'sim2sim/climber_mjcf/parts.json').read_text())['parts'];u=json.loads((ROOT/'sim2sim/climber_mjcf/user_model.json').read_text())
    assigned={p for l in u['links'] for p in l['part_ids']}|{'part_433'}
    fields=['part_id','cad_name','bbox_center_x_mm','bbox_center_y_mm','bbox_center_z_mm','cad_volume_mm3','confirmed_body','included_in_mass_scope','material','density_kg_m3','evidence']
    rows=[]
    for part in parts:
        if part['part_id'] in assigned:continue
        center=(np.array(part['bbox']['min'])+part['bbox']['max'])/2
        rows.append({'part_id':part['part_id'],'cad_name':part['name'],'bbox_center_x_mm':center[0],'bbox_center_y_mm':center[1],'bbox_center_z_mm':center[2],'cad_volume_mm3':part['volume_estimate']})
    rows_csv(folder/'unassigned_parts.csv',fields,rows)
    print(json.dumps({'directory':str(folder),'body_rows':len(root.findall('.//body')),'unassigned_rows':len(rows),'existing_files_preserved':True}))

if __name__=='__main__':prepare()
