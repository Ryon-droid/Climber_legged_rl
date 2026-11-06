"""Compare known successful recovery teacher and baseline without training."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import sys
import json
import numpy as np
import torch
ROOT = Path('/home/climber/Climber_legged_rl')
sys.path.insert(0, str(ROOT/'source'))
from stable_baselines3 import PPO
from climber_legged_rl.mujoco_env import make_mujoco_env


def main():
    torch.set_num_threads(1)
    cfg = ROOT/'configs/robot/cad_balance_estimator_round15.yaml'
    xml = ROOT/'outputs/cad_balance_estimator_round15/robot_experiment.xml'
    folder = ROOT/'outputs/round17_preflight_20261105'
    folder.mkdir(exist_ok=False)
    checkpoints = dict(baseline=ROOT/'outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip',
        teacher=ROOT/'outputs/mujoco_runs/cad_balance_estimator_round10_20261101_204316/checkpoints/climber_120000_steps.zip')
    specs = [('standing', 261409, False), ('standing', 262219, False),
             ('standing', 261106, True), ('standing', 261704, True),
             ('standing', 275000, False), ('standing', 275001, False),
             ('standing', 275002, False), ('folded', 275000, False)]
    reports = {}
    for label, checkpoint in checkpoints.items():
        policy = PPO.load(checkpoint, device='cpu')
        for mode, seed, pushes in specs:
            env = make_mujoco_env(xml, cfg)
            env.unwrapped.push.config = dict(env.unwrapped.push.config, episode_probability=float(pushes))
            trace = []
            try:
                obs, _ = env.reset(seed=seed, options={'reset_mode':mode})
                while True:
                    action = policy.predict(obs, deterministic=True)[0]
                    obs, _, done, truncated, info = env.step(action)
                    raw = env.unwrapped
                    if raw.steps % 10 == 0 or done or truncated:
                        trace.append(dict(seconds=raw.steps*raw.dt,
                            heading_deg=float(np.degrees(info['heading_error_rad'])),
                            tilt_deg=float(np.degrees(info['tilt_rad'])),
                            displacement_mm=1000*info['planar_displacement_m'],
                            action=action.tolist(), torque_Nm=info['motor_torque_Nm'],
                            contacts=info['wheel_contacts'],
                            wheel_targets_rad_s=(np.clip(action[4:], -1, 1)*raw.control['wheel_action_scale_rad_s']).tolist(),
                            wheel_actual_rad_s=raw.data.qvel[raw.v_indices[4:]].tolist()))
                    if done or truncated:
                        break
                key = f'{label}_{mode}_{seed}'
                reports[key] = dict(checkpoint=str(checkpoint), episode=info['episode_metrics'], trace=trace)
                print(json.dumps(dict(case=key, success=bool(info['combined_success']),
                    reason=info['episode_metrics']['reason'], seconds=raw.steps*raw.dt,
                    recovered=info['push_metrics']['recovered_push_count'],
                    pulses=info['push_metrics']['push_count'])), flush=True)
            finally:
                env.close()
    (folder/'teacher_comparison.json').write_text(json.dumps(reports, indent=2)+'\n')
    required = ('teacher_standing_261409', 'teacher_standing_261106', 'teacher_standing_261704')
    if not all(reports[key]['episode']['getup_metrics']['combined_success'] for key in required):
        raise RuntimeError('Previously successful recovery teacher cases did not reproduce')
    print('ROUND17_RECOVERY_TEACHER_REPRODUCED', flush=True)


if __name__ == '__main__':
    main()
