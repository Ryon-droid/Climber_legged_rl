# Climber robot assets

The current input is an OpenUSD package exported as `generated_robot.usdc`.
The usable copy is [`climber.usdc`](climber.usdc); the original USDZ is kept at
[`source/climber_openusd.usdz`](source/climber_openusd.usdz) for provenance.

The USD asset is from the earlier model export and has not been replaced with
the latest MJCF source. It has not passed an Isaac Lab load or physics review.
The replacement MJCF reports 73 unassigned CAD parts; verify which export is
authoritative and keep the USD and MJCF synchronized before training.

The currently supplied nominal geometry is a 56 mm wheel radius and two leg
link lengths of 210 mm and 250 mm. The supplied actuator information is 8009P
with a 9:1 reduction and 20 N m rated torque for leg joints, plus a 3508 wheel
motor with a 268:17 gearbox. These values are recorded in
`configs/robot/robot.yaml`; motor torque reference, peak/continuous limits and
wheel joint names still need confirmation.

The latest supplied MuJoCo model and STL resources are under
[`../../sim2sim/climber_mjcf/`](../../sim2sim/climber_mjcf/). Keep the two
formats synchronized by joint name, zero pose, mass properties and coordinate
frame. Do not rename meshes or move files referenced by the MJCF without
updating `sim2sim/climber_mjcf/robot.xml`.
