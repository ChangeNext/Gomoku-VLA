# Repository Agent Guide

## Mission

Build a Gomoku-aware VLA that observes the board, selects a strong legal move,
and safely executes it. Target cells, world poses, and robot trajectories are
labels or downstream execution metadata; do not expose them as model-input
instructions for the strategic policy.

## Repository Map

- `board/`: authoritative Gomoku and Renju rules
- `gomoku_ai/`: board encoding, tactics, MCTS, self-play, training, and inference
- `simulation/`: MuJoCo scene, robot models, trajectories, and episode collection
- `robot_control/`: execution-time safety validation
- `vision/`: calibrated board-state perception baseline
- `scripts/`: CLI, viewer, training, evaluation, and data tools
- `tests/`: `unittest` regression and integration tests
- `docs/index.md`: documentation map and source-of-truth guide

Read the relevant page linked from `docs/index.md` before changing a subsystem.
Third-party robot assets under `third_party/` are vendored dependencies, not
project-owned implementation.

## Required Work Loop

1. Inspect the relevant code, tests, and linked documentation.
2. Reproduce the behavior or establish a baseline.
3. Define observable acceptance criteria.
4. Make the smallest coherent change.
5. Add or update focused tests and documentation.
6. Run focused tests, then the full suite when practical.
7. Review the diff and report unverified assumptions explicitly.

Run the full suite from the repository root:

```bash
python -m unittest discover -s tests
```

For MuJoCo visual changes, also run the relevant viewer or
`python -m scripts.render_snapshot` and inspect the result.

## Non-negotiable Invariants

- `board/` must not depend on MuJoCo, UI, learning, or robot modules.
- Simulation and policies consume board rules; they must not redefine legality.
- Strategy selection and physical execution remain separately evaluable.
- Policy output must be checked for legality before board or robot execution.
- Keep safety checks outside learned policies. Never bypass illegal-move,
  inventory, workspace, grasp/place, collision, or emergency-stop checks.
- Do not weaken or delete a test merely to make a change pass.
- Preserve deterministic seeds for stochastic tests and evaluation tools.

## Documentation and Plans

Update the relevant `docs/` page with every feature or behavior change. Keep
this file short; detailed design and procedures belong under `docs/`.

For work spanning multiple subsystems, create
`docs/plans/active/<short-name>.md` with goals, non-goals, acceptance criteria,
risks, steps, and validation commands. Move it to `docs/plans/completed/` after
verification. Small local changes do not need a checked-in plan.

## Style

Use Python 3.10+, four-space indentation, type hints, `snake_case` names, and
`PascalCase` classes. Prefer small methods with explicit validation and clear
`ValueError` messages. Tests use `unittest`, `test_*.py`, `*Test`, and `test_*`.
