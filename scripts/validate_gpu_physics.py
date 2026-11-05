"""Measure CPU/CUDA dynamics differences, independent worlds and actual throughput."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'


def factory(xml, config):
    from climber_legged_rl.mujoco_env import make_mujoco_env
    return lambda: make_mujoco_env(xml, config)


def validate(config_path, worlds=(1, 8, 16), steps=128):
    import mujoco
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from climber_legged_rl.cad_dynamics import build_dynamics, input_hashes
    from climber_legged_rl.getup import prepare_pose_bank
    from climber_legged_rl.mujoco_env import make_mujoco_env
    from climber_legged_rl.mujoco_model import load_config
    from climber_legged_rl.warp_backend import WarpVecEnv
    from smoke_mujoco import run as smoke
    torch.set_num_threads(1)
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    build_dynamics(config_path)
    prepare_pose_bank(config_path)
    smoke(config_path, steps=1000)
    folder = ROOT / config['experiment']['output_directory']
    xml = folder / 'robot_experiment.xml'
    settings = config.get('gpu_physics', {})
    results = []
    cpu = make_mujoco_env(xml, config_path)
    gpu = WarpVecEnv([factory(xml, config_path)], settings)
    try:
        for mode in ('standing', 'folded'):
            cpu.push.enabled = gpu.bases[0].push.enabled = False
            cpu.reset(seed=112000, options={'reset_mode': mode})
            gpu.envs[0].reset(seed=112000, options={'reset_mode': mode})
            qpos_error = qvel_error = max_closure = max_force_error = 0.
            # A short open-loop comparison isolates dynamics, not policy feedback.
            for k in range(20):
                action = np.array([.005, -.005, .005, -.005, .02, .02]) * np.sin(k / 5)
                _, _, done, truncated, ci = cpu.step(action)
                with patch.object(mujoco, 'mj_step', side_effect=AssertionError('CPU integration fallback')):
                    _, _, gd, gi = gpu.step(action[None])
                if done or truncated or gd[0]:
                    raise AssertionError(f'Unexpected termination in dynamics comparison: {mode} {k}')
                qpos_error = max(qpos_error, float(np.max(np.abs(cpu.data.qpos - gpu.bases[0].data.qpos))))
                qvel_error = max(qvel_error, float(np.max(np.abs(cpu.data.qvel - gpu.bases[0].data.qvel))))
                max_closure = max(max_closure, gi[0]['closure_error_m'])
                max_force_error = max(max_force_error, float(np.max(np.abs(np.array(ci['motor_torque_Nm']) - gi[0]['motor_torque_Nm']))))
                if np.max(np.abs(gpu.bases[0].data.ctrl) - gpu.bases[0].torque_limits) > 1e-5:
                    raise AssertionError('CUDA torque exceeded hardware envelope')
            results.append(dict(mode=mode, seconds=.2, qpos_max_abs_error=qpos_error,
                qvel_max_abs_error=qvel_error, max_motor_torque_difference_Nm=max_force_error,
                gpu_max_closure_error_m=max_closure, no_cpu_integration_fallback=True))
            if qpos_error > 2e-4 or qvel_error > .01 or max_force_error > .02:
                raise AssertionError('Short CPU/CUDA dynamics comparison exceeds acceptance bounds')
            if max_closure > config['task']['terminate_max_closure_error_m']:
                raise AssertionError('CUDA closed-chain error exceeds the task limit')
        model = cpu.model
        model_contract = dict(mass_kg=float(model.body_mass.sum()), nq=model.nq, nv=model.nv,
            equality_constraints=model.neq, collision_geoms=model.ngeom,
            timestep_seconds=model.opt.timestep, decimation=config['control']['decimation'],
            spring_force_N=config['spring']['force_N'], spring_eye_limits_m=config['spring']['eye_center_limits_m'])
    finally:
        gpu.close()
        cpu.close()
    # Same action/state must be invariant to a neighboring world's different
    # action and reset; compare one world with an isolated one-world simulation.
    pair = WarpVecEnv([factory(xml, config_path)] * 2, settings)
    single = WarpVecEnv([factory(xml, config_path)], settings)
    try:
        for env in (*pair.envs, *single.envs):
            env.unwrapped.push.enabled = False
            env.reset(seed=112000, options={'reset_mode': 'standing'})
        independence_error = 0.
        for step in range(30):
            if step == 15:
                pair.envs[1].reset(seed=112004, options={'reset_mode': 'folded'})
            _, _, _, info = pair.step(np.array([[0.] * 6, [0., 0., 0., 0., .3, -.3]]))
            single.step(np.zeros((1, 6)))
            independence_error = max(independence_error, float(np.max(np.abs(pair.bases[0].data.qpos - single.bases[0].data.qpos))))
        # Float32 reduction order can vary by batch size; exact bit parity is not
        # expected. This is a leakage check over a short stable trajectory.
        if independence_error > .002:
            raise AssertionError(f'World independence difference too large: {independence_error}')
    finally:
        pair.close()
        single.close()
    benchmarks = []
    for backend in ('mujoco', 'mujoco_warp'):
        for count in worlds:
            if backend == 'mujoco' and count > 8:
                continue
            factories = [factory(xml, config_path)] * count
            vector = (WarpVecEnv(factories, settings) if backend == 'mujoco_warp' else
                      SubprocVecEnv(factories, start_method='spawn') if count > 1 else DummyVecEnv(factories))
            try:
                vector.seed(112000)
                vector.reset()
                action = np.zeros((count, 6))
                for _ in range(16):
                    vector.step(action)
                started = time.monotonic()
                for _ in range(steps):
                    vector.step(action)
                elapsed = time.monotonic() - started
                row = dict(backend=backend, worlds=count, control_steps_per_world=steps,
                    elapsed_seconds=elapsed, transitions_per_second=count * steps / elapsed,
                    includes_host_tasks=True, excludes_initial_compilation=True)
                benchmarks.append(row)
                print('GPU_BENCHMARK ' + json.dumps(row), flush=True)
            finally:
                vector.close()
    # Close-loop transfer is deliberately a separate measurement; float32 GPU
    # solver differences can affect an unstable balancing policy over seconds.
    parent = ROOT/'outputs/mujoco_runs/cad_getup_stand_round22_guarded_balance_20261031_183529/best/best_model.zip'
    policy = PPO.load(parent, device='cpu')
    transfer = []
    vector = WarpVecEnv([factory(xml, config_path)], settings)
    try:
        for mode in ('standing', 'folded'):
            obs, _ = vector.envs[0].reset(seed=112000, options={'reset_mode': mode})
            for _ in range(2000):
                action, _ = policy.predict(obs[-36:], deterministic=True)
                observations, _, done, infos = vector.step(action[None])
                obs = observations[0]
                if done[0]:
                    transfer.append(dict(mode=mode, **infos[0]['episode_metrics'],
                        success=infos[0]['combined_success'], getup=infos[0]['getup_metrics']))
                    break
    finally:
        vector.close()
    report = dict(simulation_only=True, integration_checks_passed=True,
        model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(), input_sha256=input_hashes(config_path),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), model_contract=model_contract,
        gpu_settings=settings,
        short_dynamics_comparisons=results, world_independence_max_qpos_difference=independence_error,
        parent_cpu_to_gpu_closed_loop=transfer, benchmarks=benchmarks,
        limitation='Host task logic/copies remain. GPU float32 solver is not numerically identical to CPU float64; throughput is measured under current host load.')
    (folder/'gpu_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/robot/cad_balance_gpu.yaml')
    parser.add_argument('--worlds', type=int, nargs='+', default=[1, 8, 16])
    parser.add_argument('--steps', type=int, default=128)
    args = parser.parse_args()
    if not 32 <= args.steps <= 2048 or any(not 1 <= n <= 64 for n in args.worlds):
        parser.error('Require 32..2048 steps and 1..64 worlds')
    validate(args.config, args.worlds, args.steps)


if __name__ == '__main__':
    main()
