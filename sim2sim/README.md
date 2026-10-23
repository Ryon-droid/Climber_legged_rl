# Climber MuJoCo model

The imported model is [`climber_mjcf/robot.xml`](climber_mjcf/robot.xml), with
its referenced STL files in [`climber_mjcf/meshes/`](climber_mjcf/meshes/).
The replacement export was copied from `/home/lsx/wheel_leg/local_mv8374xu_dbbsu0_mjcf_stl (2)` on 2026-10-19. The previous in-repository export is preserved at [`climber_mjcf_previous_20261019/`](climber_mjcf_previous_20261019/).

The export is structurally useful but not yet training-ready:

- 22 hinge joints and 8 position actuators are present.
- 6 closed-chain `connect` equality constraints are present.
- The exporter reports 73 CAD parts that were not assigned to links.
- The MJCF has no free joint, so its base is fixed and locomotion training is not yet possible.
- The left and right wheel rotation joints exist as `left_link 3_joint` and `right_link 3_joint` (also demonstrated by the user's CAD video); neither is included in the exported actuator list.
- Floating-base freedom and the two wheel motor actuators can be added to a simulation-specific MJCF after validating the closed-chain model; a mechanical re-export is not inherently required.
- The actuator gains, force limits, joint order, zero pose and physical spring
  interpretation still need confirmation from the mechanical and electrical
  teams.

The replacement XML parses and all 510 referenced mesh files exist. MuJoCo is
not installed in the current environment, so dynamics loading is not verified.
Before sim2sim, load the XML in the installed MuJoCo version and check the
closed-chain solver, initial pose, contacts, actuator direction and units. The
MJCF uses millimeter STL meshes with `scale="0.001 0.001 0.001"` in the XML.
