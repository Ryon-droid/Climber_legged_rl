"""Measure the physical spring load along the closed-chain get-up reference.

Only reference configurations are solved; no rollout state is changed.
Reported spring torques are quasi-static loads, not a dynamic impossibility proof.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import mujoco
import numpy as np
from scipy.optimize import least_squares
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config, closure_error
from climber_legged_rl.spring_geometry import eye_center_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/robot/cad_getup_stand_round5_dynamic.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    env = make_mujoco_env(Path(cfg['experiment']['output_directory']) / 'robot_experiment.xml', args.config)
    model, data = env.model, env.data
    pose = env.pose_bank[len(env.pose_bank)//2]
    active = env.q_indices[:4]
    passive = np.array([int(model.jnt_qposadr[j]) for j in range(1, model.njnt)
                        if model.jnt_qposadr[j] not in env.q_indices])
    joints = [np.flatnonzero(model.jnt_qposadr == q)[0] for q in passive]
    low = np.array([model.jnt_range[j, 0] if model.jnt_limited[j] else -np.inf for j in joints])
    high = np.array([model.jnt_range[j, 1] if model.jnt_limited[j] else np.inf for j in joints])
    def solve(initial, targets):
        def residual(values):
            data.qpos[:] = initial
            data.qpos[active] = targets
            data.qpos[passive] = values
            mujoco.mj_forward(model, data)
            return closure_error(model, data).ravel()
        fit = least_squares(residual, np.clip(initial[passive],low+1e-9,high-1e-9),
                            bounds=(low,high), max_nfev=500, ftol=1e-12,xtol=1e-12,gtol=1e-12)
        residual(fit.x)
        return data.qpos.copy(), float(np.max(np.abs(residual(fit.x))))
    rows = []
    initial = np.asarray(pose['qpos'])
    path = np.asarray(pose['rise_leg_references_rad'])
    for number, targets in enumerate(path):
        initial, error = solve(initial, targets)
        lengths = np.array([eye_center_distance(model,data,s) for s in ('left','right')])
        derivatives = []
        for index in range(4):
            shifted = targets.copy(); shifted[index] += 1e-5
            solve(initial, shifted)
            derivatives.append((np.array([eye_center_distance(model,data,s) for s in ('left','right')])-lengths)/1e-5)
        spring_torque = cfg['spring']['force_N']*np.asarray(derivatives).sum(axis=1)
        rows.append(dict(index=number,active_rad=targets.tolist(),spring_m=lengths.tolist(),
                         spring_generalized_torque_Nm=spring_torque.tolist(),closure_error_m=error))
    report = dict(simulation_only=True,mass_kg=float(model.body_mass.sum()),rows=rows,
                  interpretation='Required opposing motor torque from spring alone, quasi-static, with base fixed; contacts and gravity excluded.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(output=str(args.output),first=rows[0],last=rows[-1],
                          max_spring_torque_Nm=float(np.max(np.abs([r['spring_generalized_torque_Nm'] for r in rows]))))),flush=True)
    env.close()


if __name__ == '__main__':
    main()
