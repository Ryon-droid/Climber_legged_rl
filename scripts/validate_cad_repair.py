"""Continuous motion validation and optional CAD playback; no dynamics claim."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import mujoco
import numpy as np
from climber_legged_rl.cad_repair import ROOT,solve_positions
from climber_legged_rl.cad_model import closure_vectors,sha256


def validate(model_path,frames=101,video=None):
    model_path=Path(model_path)
    m=mujoco.MjModel.from_xml_path(str(model_path))
    names=['left_link1_joint','left_link_rear_1_joint','right_link1_joint','right_rear_link1_joint']
    nominal=m.key_qpos[0].copy()
    reference=np.array([nominal[int(m.joint(n).qposadr[0])] for n in names])
    samples=[];poses=[];previous=nominal
    for index,phase in enumerate(np.linspace(0,2*np.pi,frames)):
        values=reference+np.array([.04*np.sin(phase),.04*np.cos(phase),.04*np.sin(phase),.04*np.cos(phase)])
        d,fit=solve_positions(m,prescribed=dict(zip(names,values)),initial=previous,continuity=True)
        error=float(np.linalg.norm(closure_vectors(m,d),axis=1).max())
        wheels=[d.body(s+'_link 3').xpos.copy() for s in ('left','right')]
        mirrored_error=float(np.linalg.norm((wheels[0]-wheels[1])[[0,2]]))
        lengths=[float(np.linalg.norm(d.site(s+'_gas_rod_pin').xpos-d.body(s+'_gas_cylinder').xpos)) for s in ('left','right')]
        step=float(np.max(np.abs(d.qpos-previous)))
        samples.append({'frame':index,'max_closure_error_m':error,'left_right_wheel_xz_difference_m':mirrored_error,
                        'spring_pin_distances_m':lengths,'max_configuration_step':step,
                        'solver_success':bool(fit.success),'passed':bool(fit.success and error<1e-4 and mirrored_error<1e-4 and min(lengths)>0 and (index==0 or step<.05))})
        previous=d.qpos.copy();poses.append(previous)
    # Test all axes in the nullspace with the correct model coordinate order.
    d=mujoco.MjData(m);d.qpos[:]=nominal;mujoco.mj_forward(m,d)
    J=[]
    for i in range(m.neq):
        a=np.zeros((3,m.nv));b=a.copy()
        mujoco.mj_jacSite(m,d,a,None,m.eq_obj1id[i]);mujoco.mj_jacSite(m,d,b,None,m.eq_obj2id[i]);J.extend(a-b)
    _,singular,vt=np.linalg.svd(J,full_matrices=True);rank=int(sum(singular>1e-6));N=vt[rank:].T
    motor_rank={s:np.linalg.svd(N[[int(m.joint(n).dofadr[0]) for n in ns]],compute_uv=False).tolist()
                for s,ns in [('left',names[:2]),('right',names[2:])]}
    lengths=np.array([r['spring_pin_distances_m'] for r in samples])
    report={'kinematic_only':True,'passed':all(r['passed'] for r in samples),'frames':frames,
            'model_sha256':sha256(model_path),'validation_code_sha256':sha256(__file__),
            'max_closure_error_m':max(r['max_closure_error_m'] for r in samples),
            'max_left_right_wheel_xz_difference_m':max(r['left_right_wheel_xz_difference_m'] for r in samples),
            'max_step_after_first_frame':max(r['max_configuration_step'] for r in samples[1:]),
            'motor_tangent_singular_values':motor_rank,
            'spring_pin_distance_range_m':{s:[float(lengths[:,i].min()),float(lengths[:,i].max())] for i,s in enumerate(('left','right'))},
            'samples':samples,'continuity_method':'Previous-solution regularization; free-spinning pin-helper orientations held within 1e-6 rad only during prescribed playback.',
            'warning':'Prescribed CAD motion only. No gravity/contact/actuator/force/stroke validation; this is not an RL or physical standing rollout.'}
    if video:
        import imageio.v2 as imageio
        renderer=mujoco.Renderer(m,height=480,width=640)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.02,0,.04];camera.distance=.95;camera.azimuth=135;camera.elevation=-15
        video=Path(video);video.parent.mkdir(parents=True,exist_ok=True)
        try:
            with imageio.get_writer(str(video),fps=25) as writer:
                for q in poses:
                    d.qpos[:]=q;mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera)
                    frame=renderer.render();writer.append_data(frame)
                imageio.imwrite(str(video.with_suffix('.png')),frame)
        finally:renderer.close()
        report['video']=str(video)
    (model_path.parent/'motion_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    np.savez(model_path.parent/'prescribed_motion.npz',qpos=np.array(poses),motor_order=np.array(names),model_sha256=sha256(model_path))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,default=ROOT/'outputs/cad_repaired/robot_cad_repaired.xml')
    p.add_argument('--frames',type=int,default=101);p.add_argument('--video',type=Path);args=p.parse_args()
    if args.frames<3:p.error('--frames must be >= 3')
    os.environ.setdefault('MUJOCO_GL','egl')
    r=validate(args.model,args.frames,args.video)
    print(json.dumps({k:v for k,v in r.items() if k!='samples'},ensure_ascii=False,indent=2))
    raise SystemExit(0 if r['passed'] else 2)
