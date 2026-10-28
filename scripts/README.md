CAD repair commands: repair_cad_model.py, validate_cad_repair.py, prepare_physical_measurements.py, check_physical_measurements.py. See ../docs/CAD修正结果.md. These produce a kinematic candidate and explicit physical-input blockers.

2026-10-28: `prepare_cad_dynamics.py` builds the floating six-motor CAD model. Run `smoke_mujoco.py --config configs/robot/cad_dynamics.yaml`, then `train.py --backend cad-dynamics`. `training_status.py` prints the current background run and checkpoints. See [CAD动力学训练](../docs/CAD动力学训练.md) for actual reset height and simulation assumptions.

# Current route: actual CAD

Run audit_cad_model.py to inspect preserved source geometry, mass and per-motor motion. train.py and check_training_readiness.py default to cad and report physical-model blockers. See ../docs/真实模型训练.md. Historical experiments require explicit --backend mujoco.

# Experiment commands

The MuJoCo-only approximate standing task is implemented. See
[`../docs/MuJoCo首轮试训.md`](../docs/MuJoCo首轮试训.md) for limitations and commands.

- `prepare_mujoco_experiment.py`: derive a separate floating-base, six-motor model
  and solve a feasible reset; never overwrite original CAD assets.
- `smoke_mujoco.py`: validate the dynamics/interface and bind results to hashes.
- `check_training_readiness.py --backend mujoco`: check the approximate route.
  The default Isaac Lab readiness check intentionally still reports its blockers.
- `train.py --allow-approximate-model`: bounded PPO, checkpoints, logs and evaluation.
- `play.py --checkpoint ...`: seeded evaluation and optional headless video.
- `export.py --checkpoint ... --output ...`: simulation-only TorchScript actor
  with interface metadata and numerical parity checks; not CubeAI deployment.

Isaac Lab/RSL-RL task implementation is still pending. No command drives hardware.
