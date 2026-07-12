# Robot Safety Boundary

Robot safety is an external authorization layer, not a learned-policy
responsibility. `robot_control.RobotSafetyController` currently validates the
scripted simulation path before execution.

## Enforced Today

The controller checks:

- the requested stone belongs to Black or White;
- the robot is not already holding a stone;
- matching supply inventory is available;
- supply and placement poses lie inside configured Cartesian workspace limits;
- the target cell is in range, legal, and optionally matches the current player;
- every scripted end-effector trajectory point contains a pose inside the workspace.

`SafetyReport.raise_if_unsafe()` converts a failed report into a clear
`ValueError`. Callers must stop execution on a failed report.

The MuJoCo episode collector additionally records execution, grasp, placement,
and safety results so unusable demonstrations can be excluded during export.

## Not Yet Guaranteed

The current controller does not provide production robot safety. It does not
yet enforce:

- link-level or swept-volume collision avoidance;
- joint position, velocity, acceleration, torque, or thermal limits;
- self-collision and singularity avoidance;
- real emergency-stop state and watchdog timeouts;
- perception confidence or human-presence exclusion zones;
- friction-stable physical grasp validation.

Constraint-style MuJoCo stone attachment is useful for data generation but is
not evidence that a real grasp is safe or stable.

## Rules for Extensions

- Never let a VLA bypass the safety controller.
- Validate the chosen cell before generating or executing a trajectory.
- Fail closed when required state is missing or stale.
- Preserve a machine-readable reason for every rejection.
- Add focused tests for every new safety limit and failure mode.
- Keep simulation-only success distinct from real-robot readiness.

Relevant tests are in `tests/test_robot_control.py` and the collection and
environment test modules.
