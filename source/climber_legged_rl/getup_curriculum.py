"""Explicit rise-finish reset curriculum; full-prone evaluation stays separate."""
from pathlib import Path
import json
import mujoco
import numpy as np
from scipy.optimize import least_squares
from .getup import ClimberGetupEnv, configuration_hash, file_hash
from .mujoco_model import ROOT, closure_error, load_config
from .spring_geometry import configured_eye_limits, eye_center_distance
from .stance_geometry import virtual_leg_geometry


def prepare_curriculum_bank(config_path):
    config=load_config(config_path)
    settings=config.get('getup',{}).get('reset_curriculum',{})
    if not settings.get('enabled'):
        return None
    progress=np.asarray(settings['reference_progress'],dtype=float)
    probability=settings['full_prone_probability']
    if (progress.ndim!=1 or not len(progress) or not np.isfinite(progress).all()
            or np.any(progress<=0) or np.any(progress>=1) or not 0<=probability<=1):
        raise ValueError('Invalid get-up reset curriculum')
    rise=settings.get('rise_seconds',config['getup']['rise_seconds'])
    if not np.isfinite(rise) or rise<=0 or rise+config['getup']['settle_seconds']>=config['getup']['deadline_seconds']:
        raise ValueError('Curriculum rise duration must leave settling time before the deadline')
    folder=ROOT/config['experiment']['output_directory']
    env=ClimberGetupEnv(folder/'robot_experiment.xml',config_path)
    poses=[]
    try:
        model,data=env.model,env.data
        limits=configured_eye_limits(config,required=True)
        indices=[int(model.jnt_qposadr[j]) for j in range(1,model.njnt) if model.jnt_qposadr[j] not in env.q_indices]
        joints=[int(np.flatnonzero(model.jnt_qposadr==q)[0]) for q in indices]
        low=np.array([model.jnt_range[j,0] if model.jnt_limited[j] else -np.inf for j in joints])
        high=np.array([model.jnt_range[j,1] if model.jnt_limited[j] else np.inf for j in joints])
        for parent in env.pose_bank:
            path=np.asarray(parent['rise_leg_references_rad'])
            phase=np.asarray(parent['rise_reference_progress'])
            for start in progress:
                reference=np.array([np.interp(start,phase,path[:,j]) for j in range(4)])
                initial=env.nominal_qpos.copy()
                initial[env.q_indices[:4]]=reference
                def residual(values):
                    data.qpos[:]=initial
                    data.qpos[indices]=values
                    mujoco.mj_forward(model,data)
                    return closure_error(model,data).ravel()
                fit=least_squares(residual,np.clip(initial[indices],low+1e-10,high-1e-10),
                    bounds=(low,high),max_nfev=1200,ftol=1e-12,xtol=1e-12,gtol=1e-12)
                residual(fit.x)
                error=float(np.linalg.norm(closure_error(model,data),axis=1).max())
                bottoms=[]
                for g in range(model.ngeom):
                    if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:
                        continue
                    mesh=model.geom_dataid[g];a=model.mesh_vertadr[mesh];n=model.mesh_vertnum[mesh]
                    points=model.mesh_vert[a:a+n]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
                    bottoms.append(float(points[:,2].min()))
                data.qpos[2]+=config['getup']['ground_clearance_m']-min(bottoms)
                mujoco.mj_forward(model,data)
                lengths={side:eye_center_distance(model,data,side) for side in ('left','right')}
                if error>1e-4 or any(not limits[s][0]<=lengths[s]<=limits[s][1] for s in lengths):
                    raise RuntimeError('Curriculum reset violates closure or spring travel')
                geometry=virtual_leg_geometry(model,data,env.motor_midpoints,env.wheel_centers,env.chassis_axes)
                mask=phase>start
                poses.append(dict(qpos=data.qpos.tolist(),length_m=float(geometry['side_view_distance_m'].mean()),
                    curriculum_start_progress=float(start),rearward_offset_m=float(geometry['forward_offset_m'].mean()),
                    parent_rearward_offset_m=parent['rearward_offset_m'],closure_error_m=error,
                    spring_eye_distances_m=lengths,leg_reference_rad=reference.tolist(),
                    rise_leg_references_rad=np.vstack([reference,path[mask]]).tolist(),
                    rise_reference_progress=np.r_[0.,(phase[mask]-start)/(1.-start)].tolist()))
    finally:
        env.close()
    report=dict(poses=poses,model_sha256=file_hash(folder/'robot_experiment.xml'),
        config_sha256=configuration_hash(config_path),implementation_sha256=file_hash(__file__),
        definition='supported rise-finish resets; these are NOT full rear-extended prone resets')
    (folder/'getup_curriculum_poses.json').write_text(json.dumps(report,indent=2))
    return report


def validate_curriculum_bank(config_path,folder):
    report=json.loads((Path(folder)/'getup_curriculum_poses.json').read_text())
    if (report['config_sha256']!=configuration_hash(config_path)
            or report['model_sha256']!=file_hash(Path(folder)/'robot_experiment.xml')
            or report['implementation_sha256']!=file_hash(__file__)):
        raise RuntimeError('Stale get-up curriculum; rerun prepare_getup.py')
    return report


class ClimberGetupCurriculumEnv(ClimberGetupEnv):
    def __init__(self,model_path,config_path,render_mode=None):
        super().__init__(model_path,config_path,render_mode)
        self.curriculum_bank=validate_curriculum_bank(config_path,Path(model_path).parent)['poses']
        self.settings=self.getup['reset_curriculum']

    def _phase(self):
        if getattr(self,'reset_mode','standing')=='curriculum':
            return min(1.,self.steps*self.dt/self.settings.get('rise_seconds',self.getup['rise_seconds']))
        return super()._phase()

    def reset(self,*,seed=None,options=None):
        mode=(options or {}).get('reset_mode')
        if mode not in (None,'standing','folded','curriculum'):
            raise ValueError('Unknown reset mode')
        rng=np.random.default_rng(seed) if seed is not None else self.np_random
        if mode is None:
            mode=('standing' if rng.random()<self.getup['stand_probability'] else
                  'folded' if rng.random()<self.settings['full_prone_probability'] else 'curriculum')
        if mode!='curriculum':
            return super().reset(seed=seed,options={'reset_mode':mode})
        full_bank=self.pose_bank
        self.pose_bank=self.curriculum_bank
        try:
            obs,info=super().reset(seed=seed,options={'reset_mode':'folded'})
        finally:
            self.pose_bank=full_bank
        self.reset_mode='curriculum'
        info['reset_mode']='curriculum'
        info['full_prone']=False
        return self._observation(),info
