# Architecture

Gomoku-VLA separates board perception, strategic move choice, and physical
execution so each failure can be measured independently. The main objective is
to choose and execute a strong legal move from observation. Target coordinates
and trajectories are supervision or execution data, not strategic input hints.

## Dependency Direction

```text
camera/image
    -> vision detector
    -> GomokuBoard-compatible state
    -> tactical guard + policy/value network + MCTS
    -> selected legal (row, col)
    -> safety validation
    -> board/world conversion + IK/scripted trajectory
    -> MuJoCo or future real robot
```

The board package is the rule authority. Policy, UI, simulation, and robot code
may consume it, but must not maintain competing legality implementations.

## Implemented Modules

### `board/`

Owns `Player`, `GomokuBoard`, free-rule and Renju legality, center-opening
enforcement, turn transitions, wins, and draws. See [Game rules](game_rules.md).

### `gomoku_ai/`

Contains board/action encoding, a policy/value protocol, residual PyTorch
network, PUCT MCTS, tactical move selection, self-play, replay storage,
training, checkpoint promotion/evaluation, inference, policy episode records,
and OpenVLA-OFT-style manifest export.

The board-only AlphaZero loop is intentionally independent of MuJoCo so
strategy training remains fast and deterministic.

### `simulation/`

`GomokuMujocoEnv` owns MuJoCo scene construction and updates, cameras,
board/world conversion, kinematic/Panda/SO-101 models, joint control, stone
supply and held/active-stone state. `scripted_robot` builds Cartesian
pick/place traces. `policy_collection` executes and records SO-101-oriented
multi-view demonstrations.

### `robot_control/`

Provides the current external safety gate for supply, legality, player,
workspace, and scripted trajectory checks. It is deliberately separate from
learned policy code. See [Robot safety](robot_safety.md).

### `vision/`

Provides a calibrated top-down grid sampler and brightness/contrast stone
classifier. This is a controlled-image baseline, not a production perception
system. See [Vision](vision.md).

### `scripts/` and `web/`

Expose terminal play, click UI, MuJoCo viewers, rendering, AlphaZero training,
checkpoint evaluation, episode generation/export, and browser-based human
evaluation. The browser frontend is static content served by the evaluation
server.

### `tests/`

Uses `unittest` to cover rules, policy/search/training components, checkpoint
and episode formats, MuJoCo state and robots, data export, safety, vision, and
human evaluation state.

## Core Data Flows

### Board-only strategy training

```text
GomokuBoard
  -> current-player feature planes
  -> policy/value network
  -> legal-move-masked MCTS
  -> self-play (state, policy target, outcome)
  -> replay buffer
  -> policy/value optimization
  -> candidate evaluation and checkpoint promotion
```

Self-play follows MCTS visit distributions. Hand-written tactical overrides are
used in evaluation/play inference, not to replace self-play targets.

### Simulation demonstration collection

```text
pre-action board + board/wrist images
  -> checkpoint policy selects move
  -> board legality + RobotSafetyController
  -> selected cell maps to world target and SO-101 joint sequence
  -> phase execution and multi-view capture
  -> raw JSONL + image assets
  -> filtered OpenVLA-style manifest
```

Model inputs default to pre-action `board_top` and `wrist_cam` observations plus
a non-leaking instruction. Move tokens, target cells, world poses, policy/value
signals, and action sequences stay under targets or metadata. `robot_full` is a
QA/ablation view by default.

## Architectural Invariants

- Game legality has one authority: `GomokuBoard`.
- Observations must represent information available before the action.
- Strategic and execution labels must not leak into model inputs.
- Selected moves are validated again at the board/execution boundary.
- Safety remains outside VLA inference and fails closed.
- Strategy strength, perception accuracy, and manipulation success are
  evaluated separately before end-to-end reporting.

## Planned Boundary

There is no production `vla/` package yet. Current code prepares and exports a
custom multi-view autoregressive training contract; integrating an actual
OpenVLA/OFT trainer and runtime remains future work. Real-robot drivers,
full collision checking, emergency-stop integration, robust perception, and
closed-loop grasp/placement correction also remain outside the current MVP.
