"""Opt-in bounded rigid-base perturbations of solved, closed-chain prone poses."""
import mujoco
import numpy as np

from .mujoco_model import closure_error
from .policy_getup import ClimberPolicyGetupEnv
from .spring_geometry import eye_center_distance


class ClimberRobustGetupEnv(ClimberPolicyGetupEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise = self.getup['folded_reset_noise']
        for name, cap in (('roll_half_width_rad', .05), ('pitch_half_width_rad', .05),
                          ('base_speed_half_width_m_s', .03), ('base_rate_half_width_rad_s', .1)):
            value = self.noise.get(name)
            if not np.isfinite(value) or not 0 <= value <= cap:
                raise ValueError('Invalid bounded folded reset noise: '+name)
        self.meshes = []
        for geom in range(self.model.ngeom):
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.model.geom_dataid[geom]
                start = self.model.mesh_vertadr[mid]
                self.meshes.append((geom, self.model.mesh_vert[start:start+self.model.mesh_vertnum[mid]].copy()))
        self._reset_rng = np.random.default_rng(208341)

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if self.reset_mode != 'folded':
            return observation, info
        if seed is not None:
            self._reset_rng = np.random.default_rng(np.random.SeedSequence([seed, 208341]))
        roll = self._reset_rng.uniform(-self.noise['roll_half_width_rad'], self.noise['roll_half_width_rad'])
        pitch = self._reset_rng.uniform(-self.noise['pitch_half_width_rad'], self.noise['pitch_half_width_rad'])
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        delta = np.array([cr*cp, sr*cp, cr*sp, -sr*sp])
        q = np.empty(4)
        mujoco.mju_mulQuat(q, delta, self.data.qpos[3:7])
        self.data.qpos[3:7] = q
        mujoco.mj_forward(self.model, self.data)
        bottom = min(float((vertices @ self.data.geom_xmat[g].reshape(3, 3).T+
                            self.data.geom_xpos[g])[:, 2].min()) for g, vertices in self.meshes)
        self.data.qpos[2] += self.getup['ground_clearance_m']-bottom
        # Every internal coordinate stays at a bounded IK solution. Rigid-base
        # velocity preserves the relative velocities at the six closure points.
        self.data.qvel[:] = 0.
        self.data.qvel[:3] = self._reset_rng.uniform(-self.noise['base_speed_half_width_m_s'],
            self.noise['base_speed_half_width_m_s'], 3)
        self.data.qvel[3:6] = self._reset_rng.uniform(-self.noise['base_rate_half_width_rad_s'],
            self.noise['base_rate_half_width_rad_s'], 3)
        mujoco.mj_forward(self.model, self.data)
        error = float(np.linalg.norm(closure_error(self.model, self.data), axis=1).max())
        if error > 1e-4:
            raise RuntimeError('Folded perturbation broke a closed chain')
        eyes = {side: eye_center_distance(self.model, self.data, side) for side in ('left', 'right')}
        for side, length in eyes.items():
            low, high = self.config['spring']['eye_center_limits_m'][side]
            if not low-1e-8 <= length <= high+1e-8:
                raise RuntimeError('Folded perturbation exceeds measured spring travel')
        self._set_hold_reference()
        self.reset_hip_height = float(np.mean(self.data.xpos[self.hip_ids, 2]))
        info.update(folded_reset_noise=dict(roll_rad=float(roll), pitch_rad=float(pitch),
            base_qvel=self.data.qvel[:6].tolist(), spring_eye_m=eyes, closure_error_m=error))
        return self._observation(), info
