"""Batched CUDA integration of the unchanged CAD model with host task bookkeeping.

MuJoCo Warp integrates every physical substep. Task rewards, episode state,
resets and final-state mj_forward stay on the host to share the CPU reference.
This intentionally is not an entirely GPU resident environment.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

try:
    import warp as wp
    import mujoco_warp as mw
except ImportError as error:
    raise ImportError('MuJoCo Warp backend requires requirements-gpu.txt in the GPU Python environment') from error


def validate_gpu_report(config_path):
    """Require evidence for this physical model and backend before formal training."""
    from .cad_dynamics import input_hashes
    from .mujoco_model import ROOT, load_config
    config = load_config(config_path)
    # Estimator and ordinary GPU variants share identical integration. Accept
    # either report only when all physical inputs and backend source match.
    candidates = [ROOT/config['experiment']['output_directory']/'gpu_validation.json',
                  ROOT/'outputs/cad_balance_gpu/gpu_validation.json']
    model_path = ROOT/config['experiment']['output_directory']/'robot_experiment.xml'
    expected = input_hashes(config_path)
    expected.pop('source/climber_legged_rl/velocity_estimation.py', None)
    for path in candidates:
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        previous = dict(report.get('input_sha256', {}))
        previous.pop('source/climber_legged_rl/velocity_estimation.py', None)
        if (report.get('integration_checks_passed') and previous == expected
                and report.get('gpu_settings') == config.get('gpu_physics', {})
                and report.get('model_sha256') == hashlib.sha256(model_path.read_bytes()).hexdigest()):
            return report
    raise RuntimeError('GPU validation is missing or stale. Run scripts/validate_gpu_physics.py first')


@wp.kernel
def motor_control(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                  qindex: wp.array(dtype=int), vindex: wp.array(dtype=int),
                  targets: wp.array2d(dtype=float), gains: wp.array2d(dtype=float),
                  limits: wp.array2d(dtype=float), speed_limits: wp.array(dtype=float),
                  strength: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float),
                  trace: wp.array3d(dtype=float), saturation: wp.array2d(dtype=float), substep: int):
    world, motor = wp.tid()
    speed = qvel[world, vindex[motor]]
    torque = float(0.)
    if motor < 4:
        torque = gains[world, 0] * (targets[world, motor] - qpos[world, qindex[motor]]) - gains[world, 1] * speed
    else:
        torque = gains[world, 2] * (targets[world, motor] - speed)
    torque *= strength[world, motor]
    limit = limits[world, motor]
    if wp.abs(torque) >= limit:
        saturation[world, motor] += 1.
    torque = wp.clamp(torque, -limit, limit)
    if wp.abs(speed) > speed_limits[motor] and torque * speed > 0.:
        torque = 0.
    ctrl[world, motor] = torque
    trace[world, substep, motor] = torque


class WarpPhysics:
    def __init__(self, bases, settings=None):
        self.bases = bases
        self.count = len(bases)
        self.settings = settings or {}
        self.device = self.settings.get('device', 'cuda:0')
        wp.init()
        if not wp.get_device(self.device).is_cuda:
            raise ValueError('mujoco_warp physics requires a CUDA device')
        self.decimation = bases[0].control['decimation']
        if any(base.control['decimation'] != self.decimation for base in bases):
            raise ValueError('Batched environments must share the same physical timestep')
        self.targets_host = np.zeros((self.count, 6), np.float32)
        with wp.ScopedDevice(self.device):
            self.model = mw.put_model(bases[0].model, batch_sizes={'geom_friction': self.count})
            self.data = mw.put_data(bases[0].model, bases[0].data, nworld=self.count,
                nconmax=self.settings.get('nconmax', 512), njmax=self.settings.get('njmax', 1024))
            self.qindex = wp.array(bases[0].q_indices.astype(np.int32), dtype=int)
            self.vindex = wp.array(bases[0].v_indices.astype(np.int32), dtype=int)
            self.speed_limits = wp.array(bases[0].speed_limits.astype(np.float32))
            self.targets = wp.zeros((self.count, 6), dtype=float)
            self.gains = wp.zeros((self.count, 3), dtype=float)
            self.limits = wp.zeros((self.count, 6), dtype=float)
            self.strength = wp.zeros((self.count, 6), dtype=float)
            self.trace = wp.zeros((self.count, self.decimation, 6), dtype=float)
            self.saturation = wp.zeros((self.count, 6), dtype=float)
        self.graph = None
        self.warmed = False

    def _upload(self):
        bases = self.bases
        # Rebase the float32 physical clock from integer episode step counts on
        # each policy tick; its returned value is still checked after integration.
        fields = {name: np.stack([getattr(b.data, name) for b in bases]).astype(np.float32)
                  for name in ('qpos', 'qvel', 'qacc_warmstart', 'xfrc_applied')}
        fields['time'] = np.array([b.steps * b.dt for b in bases], np.float32)
        fields['qfrc_applied'] = np.stack([b.data.qfrc_applied.copy() for b in bases]).astype(np.float32)
        for i, base in enumerate(bases):
            if base.spring_dofs:
                fields['qfrc_applied'][i] = 0.
                fields['qfrc_applied'][i, base.spring_dofs] = base.config['spring']['force_N']
            base.physics_time_tolerance = max(1e-6,
                16 * np.finfo(np.float32).eps * max(1., (base.steps + 1) * base.dt))
        for name, value in fields.items():
            getattr(self.data, name).assign(value)
        self.targets.assign(self.targets_host)
        self.gains.assign(np.array([[b.control['leg_kp'], b.control['leg_kd'], b.control['wheel_kv']] for b in bases], np.float32))
        self.limits.assign(np.stack([b.torque_limits for b in bases]).astype(np.float32))
        self.strength.assign(np.array([getattr(b, 'motor_strength', np.ones(6)) for b in bases], np.float32))
        # The leading model batch dimension preserves independent wheel/ground
        # friction sampling; no world silently inherits another world's model.
        self.model.geom_friction.assign(np.stack([b.model.geom_friction for b in bases]).astype(np.float32))
        self.saturation.zero_()
        self.data.overflow.zero_()

    def _steps(self):
        for substep in range(self.decimation):
            wp.launch(motor_control, dim=(self.count, 6), inputs=[self.data.qpos, self.data.qvel,
                self.qindex, self.vindex, self.targets, self.gains, self.limits, self.speed_limits,
                self.strength, self.data.ctrl, self.trace, self.saturation, substep])
            mw.step(self.model, self.data)

    def advance(self):
        with wp.ScopedDevice(self.device):
            self._upload()
            if not self.warmed:
                # Compile outside capture, restore uploaded states, then capture.
                self._steps()
                wp.synchronize_device(self.device)
                self._upload()
                if self.settings.get('cuda_graph', True):
                    with wp.ScopedCapture() as capture:
                        self._steps()
                    self.graph = capture.graph
                    wp.capture_launch(self.graph)
                else:
                    self._steps()
                self.warmed = True
            elif self.graph is None:
                self._steps()
            else:
                wp.capture_launch(self.graph)
            qpos, qvel = self.data.qpos.numpy(), self.data.qvel.numpy()
            warmstart, times = self.data.qacc_warmstart.numpy(), self.data.time.numpy()
            trace, saturation = self.trace.numpy(), self.saturation.numpy()
            overflow = self.data.overflow.numpy()
            if np.any(overflow):
                raise RuntimeError(f'MuJoCo Warp contact/constraint capacity overflow: {overflow.tolist()}; enlarge buffers')
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                raise RuntimeError('Nonfinite CUDA simulation state')
            results = []
            for i, base in enumerate(self.bases):
                base.data.qpos[:] = qpos[i]
                base.data.qvel[:] = qvel[i]
                base.data.qacc_warmstart[:] = warmstart[i]
                base.data.time = float(times[i])
                base.data.ctrl[:] = trace[i, -1]
                for torque in trace[i]:
                    base._observe_motor_torque(torque)
                results.append((trace[i, -1].astype(float), saturation[i].astype(float)))
            return results


class WarpVecEnv(VecEnv):
    """SB3 vector environment with one synchronized CUDA batch per policy tick."""
    def __init__(self, factories, settings=None):
        self.envs = [factory() for factory in factories]
        self.bases = [env.unwrapped for env in self.envs]
        self.engine = WarpPhysics(self.bases, settings)
        self.pool = ThreadPoolExecutor(max_workers=len(self.envs), thread_name_prefix='warp-task')
        self.results = None
        self.futures = None
        self.closed = False
        self.barrier = threading.Barrier(len(self.envs), action=self._integrate, timeout=180)
        for index, base in enumerate(self.bases):
            base.physics_stepper = self._hook(index)
        super().__init__(len(self.envs), self.envs[0].observation_space, self.envs[0].action_space)

    def _integrate(self):
        try:
            self.results = self.engine.advance()
        except BaseException as error:
            self.physics_failure = error
            raise

    def _hook(self, index):
        def stepper(base, legs, wheels):
            self.engine.targets_host[index] = np.r_[legs, wheels]
            try:
                self.barrier.wait()
            except threading.BrokenBarrierError as error:
                raise RuntimeError('CUDA batch failed; all worlds stopped') from getattr(self, 'physics_failure', error)
            return self.results[index]
        return stepper

    def reset(self):
        observations = []
        for i, env in enumerate(self.envs):
            obs, self.reset_infos[i] = env.reset(seed=self._seeds[i], options=self._options[i])
            observations.append(obs)
        self._reset_seeds()
        self._reset_options()
        return np.stack(observations)

    def _step(self, index, action):
        try:
            obs, reward, terminated, truncated, info = self.envs[index].step(action)
            done = terminated or truncated
            info['TimeLimit.truncated'] = truncated and not terminated
            if done:
                info['terminal_observation'] = obs.copy()
                obs, self.reset_infos[index] = self.envs[index].reset()
            return obs, reward, done, info
        except BaseException:
            self.barrier.abort()
            raise

    def step_async(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, 6) or not np.isfinite(actions).all():
            raise ValueError('Expected one finite six-motor action per world')
        if self.futures is not None:
            raise RuntimeError('step_async called while a batch is pending')
        self.futures = [self.pool.submit(self._step, i, action.copy()) for i, action in enumerate(actions)]

    def step_wait(self):
        try:
            results = [future.result() for future in self.futures]
            obs, reward, done, info = zip(*results)
            return np.stack(obs), np.asarray(reward, np.float32), np.asarray(done, bool), list(info)
        finally:
            self.futures = None

    def close(self):
        if not self.closed:
            self.closed = True
            self.barrier.abort()
            self.pool.shutdown(wait=True, cancel_futures=True)
            for base in self.bases:
                base.physics_stepper = None
            for env in self.envs:
                env.close()

    def get_attr(self, attr_name, indices=None):
        return [self.envs[i].get_wrapper_attr(attr_name) for i in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        for i in self._get_indices(indices):
            self.envs[i].set_wrapper_attr(attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [self.envs[i].get_wrapper_attr(method_name)(*method_args, **method_kwargs) for i in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        from stable_baselines3.common.env_util import is_wrapped
        return [is_wrapped(self.envs[i], wrapper_class) for i in self._get_indices(indices)]
