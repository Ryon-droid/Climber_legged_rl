# Checks

Run `python -m unittest discover -s tests -p 'test_*.py' -v`.

The standard-library readiness tests run without simulator dependencies.
The MuJoCo tests require `requirements-mujoco.txt` plus a separately generated
experiment model; they verify six motors, free base, mass, coordinate-side
ordering, seeded reset, finite observations, torque caps and timeout handling.
They skip explicitly when dependencies or the generated model are absent.

Also run `scripts/smoke_mujoco.py` before PPO. `scripts/export.py` performs
random-input and real-trajectory parity checks for the simulation-only actor.
No test certifies hardware safety, real gas springs, or obstacle/jump policies.

Real CAD tests preserve body/joint transforms, inertials, sites and closures, check unscaled source mass and strict compilation, and reproduce the left constraint failure. Passing tests mean the audit detects the issue; they do not certify training readiness.

CAD repair tests additionally integrate an analytic cube inertia, verify retained CAD geometry and source hashes, restored rod ownership, pin-axis closure, passive slide semantics, 135 mm stance and continuous 101-frame symmetry/branch continuity.
