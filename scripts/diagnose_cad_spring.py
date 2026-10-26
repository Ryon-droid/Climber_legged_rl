"""Diagnose original CAD spring closures in memory; never release a training model."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import mujoco
import numpy as np
from scipy.optimize import least_squares
import yaml
from climber_legged_rl.cad_model import ROOT,DEFAULT_CAD_CONFIG,diagnostic_tree,closure_vectors,motion_probe,sha256


def stance_probe(model,target,attempts=8):
    data=mujoco.MjData(model)
    indices=np.array([int(model.jnt_qposadr[i]) for i in range(model.njnt)
                      if model.joint(i).name not in ('left_link 3_joint','right_link 3_joint')])
    lower=model.jnt_range[indices,0];upper=model.jnt_range[indices,1]
    rng=np.random.default_rng(42);records=[]
    def residual(values):
        data.qpos[:]=model.qpos0;data.qpos[indices]=values;mujoco.mj_forward(model,data)
        heights=np.array([data.body(s+'_link1').xpos[2]-data.body(s+'_link 3').xpos[2] for s in ('left','right')])
        return np.r_[closure_vectors(model,data).ravel(),heights-target]
    for k in range(attempts):
        initial=model.qpos0[indices] if k==0 else rng.uniform(lower*.6,upper*.6)
        fit=least_squares(residual,initial,bounds=(lower,upper),max_nfev=600,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        residual(fit.x)
        heights=[float(data.body(s+'_link1').xpos[2]-data.body(s+'_link 3').xpos[2]) for s in ('left','right')]
        error=float(np.linalg.norm(closure_vectors(model,data),axis=1).max())
        rec={'attempt':k,'max_closure_error_m':error,'hip_wheel_vertical_m':heights,'qpos_rad':data.qpos.tolist(),
             'solver_success':bool(fit.success),'passed':bool(fit.success and error<=1e-4 and max(abs(h-target) for h in heights)<=.003)}
        records.append(rec)
        if rec['passed']:break
    return {'target_hip_wheel_vertical_m':target,'attempts':records,'passed':any(r['passed'] for r in records),
            'seed':42,'note':'Bounded numeric search; not global infeasibility proof or physical zero calibration.'}


def main():
    cfg=yaml.safe_load(DEFAULT_CAD_CONFIG.read_text());source=ROOT/cfg['source_model']
    measurements=yaml.safe_load((ROOT/cfg['mechanical_measurements']).read_text())
    target=measurements['geometry']['nominal_standing_pose']['hip_to_wheel_vertical_separation_m']
    results=[]
    for disabled in ([],['closure_008'],['closure_007'],['closure_007','closure_008']):
        root=diagnostic_tree(source)
        for eq in list(root.find('equality')):
            if eq.get('name') in disabled:root.find('equality').remove(eq)
        model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        probe=motion_probe(model,cfg['audit'],cfg['expected_leg_joints'],cfg['expected_wheel_joints'])
        entry={'disabled_for_diagnosis_only':disabled,'passed_count':sum(r['passed'] for r in probe['samples']),
               'total':len(probe['samples']),'failed_samples':[{'joint':r['joint'],'delta_rad':r['delta_rad'],'error_m':r['max_closure_error_m']}
                                                          for r in probe['samples'] if not r['passed']]}
        if not disabled or disabled==['closure_008']:
            entry['stance_probe']=stance_probe(model,target,attempts=8 if not disabled else 1)
        results.append(entry)
    report={'diagnostic_only':True,'ready_for_training':False,'source_sha256':sha256(source),
            'config_sha256':sha256(DEFAULT_CAD_CONFIG),'code_sha256':sha256(__file__),
            'measurements_sha256':sha256(ROOT/cfg['mechanical_measurements']),
            'results':results,'note':'Ablation locates coupling; not a repair. Reconstruct actual spring mount, slider, travel and forces before training. No ablated model saved.'}
    output=ROOT/cfg['output_directory'];output.mkdir(parents=True,exist_ok=True)
    (output/'closure_ablation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (output/'stance_probe.json').write_text(json.dumps(results[0]['stance_probe'],ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
