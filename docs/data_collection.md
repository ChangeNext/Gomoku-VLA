# Data Collection

The first collection stage records board-only policy episodes. Each JSONL row is one selected move, not one completed game. This shape is intentional: robot poses, rendered frames, camera observations, and trajectories can be attached to the same move record later.

## Generate Policy Episodes

Use a trained checkpoint as the strategic teacher:

```bash
python -m scripts.generate_policy_episodes \
  --checkpoint gomoku_ai/runs/<run-name>/checkpoints/best.pt \
  --games 10 \
  --simulations 64 \
  --device auto
```

By default this writes:

```text
gomoku_ai/runs/<run-name>/data/best_policy_episodes.jsonl
```

Use `--output-jsonl` to write somewhere else. Use `--max-moves` for short smoke tests.

## Generate MuJoCo Policy Episodes

Use MuJoCo collection when the selected move should be paired with rendered observations and scripted pick/place actions:

```bash
python -m scripts.generate_mujoco_policy_episodes \
  --checkpoint gomoku_ai/runs/<run-name>/checkpoints/best.pt \
  --games 10 \
  --simulations 64 \
  --device auto
```

`--robot-model` defaults to `so101`. Use `--robot-model kinematic` only for quick controller smoke tests.

The default image size is `640x640` for each camera. Keep training collection at `640x640` or higher unless storage is the limiting factor. The CLI rejects images below `224x224` by default because they are only useful for smoke-testing the pipeline, not for VLA training. Use `--allow-low-res-smoke` only with tiny `--max-moves` checks.

By default this writes:

```text
gomoku_ai/runs/<run-name>/data/best_mujoco_policy_episodes.jsonl
gomoku_ai/runs/<run-name>/data/assets/<game-id>/*.png
```

The scripted controller is kinematic in this stage. It records a deterministic Cartesian trace for:

```text
home -> pre_pick -> pick -> grasp -> lift -> pre_place -> place -> release -> retreat
```

The action vectors use:

```text
[x, y, z, qw, qx, qy, qz, gripper]
```

`joint_trajectory` is `null` until a real joint/IK controller is introduced.

Collection now runs the same safety and inventory interfaces used by the viewer. Each move validates supply availability, target legality, workspace bounds, and action trace bounds before execution. The environment records black/white supply counts and held-stone state before and after the move.

## Move-Level Schema

Each row contains:

- `game_id`
- `step`
- `timestamp`
- `board_before`
- `board_after`
- `board_size`
- `win_length`
- `rule_set`
- `enforce_center_opening`
- `current_player`
- `current_player_value`
- `selected_move`
- `action_index`
- `policy_source`
- `policy_probs`
- `value`
- `legal`
- `used_tactical_move`
- `winner`
- `winner_value`
- `terminal`
- `checkpoint`
- `robot_action`
- `observation`
- `error`

`robot_action` and `observation` are intentionally `null` in the board-only stage. MuJoCo collection fills them with pick/place poses, end-effector trajectories, rendered frames, language instructions, placement success, and placement error.

## VLA-Oriented Fields

MuJoCo records use LeRobot/OpenVLA-friendly grouping:

- `observation.language_instruction`: natural-language task, for example `place the black stone at row 7 column 8`
- `observation.images.*`: rendered camera image paths such as `top_before`, `top_after`, `iso_before`, and `robot_full_after`
- `observation.image_metadata`: image width, height, camera names, and whether the sample meets the minimum training resolution
- `observation.state`: board vector, target cell, target world pose, current player, and initial robot state
- `robot_action.action`: Cartesian action sequence with names in `robot_action.action_names`
- `robot_action.ee_trajectory`: phase-labelled end-effector trajectory
- `robot_action.execution_success`: whether the scripted placement matched the selected cell
- `robot_action.placement_error_world` and `robot_action.placement_error_cell`: execution quality signals
- `robot_action.safety`: pick/place/trace validation results
- `robot_action.supply_before` and `robot_action.supply_after`: remaining black/white stones
- `robot_action.attachment_mode`: currently `scripted_held_stone`
- `observation.action`: action sequence grouped with its format and controller type for VLA loaders
- `observation.state.board_after_flat`: post-action board state for supervised consistency checks

## Intended Pipeline

```text
GomokuBoard state
-> CheckpointPolicy.predict()
-> move-level JSONL record
-> MuJoCo step or scripted robot controller
-> same record extended with observation and robot_action fields
```

The VLA dataset should keep the strategic teacher fields (`selected_move`, `policy_probs`, `value`) separate from execution fields (`robot_action`, `observation`, placement result). This lets policy mistakes and robot placement failures be evaluated independently.
