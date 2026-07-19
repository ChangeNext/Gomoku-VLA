# Data Collection

Project goal: collect data for a Gomoku-aware VLA that decides the next move from board images. The target row/column must not appear in model inputs; `selected_move`, `target_cell`, `target_world_xyz`, and robot trajectories are labels or metadata.

The first collection stage records board-only policy episodes. Each JSONL row is one selected move, not one completed game. This shape is intentional: robot poses, rendered frames, camera observations, and trajectories can be attached to the same move record later.

## Generate Policy Episodes

Use a trained checkpoint as the strategic teacher:

```bash
python -m scripts.generate_policy_episodes \
  --checkpoint gomoku_ai/runs/<run-name>/checkpoints/best.pt \
  --games 10 \
  --simulations 64 \
  --device auto \
  --sample-moves \
  --temperature 1.0 \
  --temperature-moves 12 \
  --late-temperature 0.1 \
  --root-noise
```

By default this writes:

```text
gomoku_ai/runs/<run-name>/data/best_policy_episodes.jsonl
```

Use `--output-jsonl` to write somewhere else. Use `--max-moves` for short smoke tests.

## Generate External-Engine Policy Episodes

Rapfi, Embryo, Yixin-style engines can be used as strategic teachers through
the Gomocup/Piskvork stdin/stdout protocol. Keep the engine binary outside this
repository and pass it as a command; the collector records the selected move as
a one-hot policy because these engines do not provide an MCTS visit distribution
through the basic protocol.

```bash
python -m scripts.generate_policy_episodes \
  --engine-command /path/to/rapfi \
  --board-size 15 \
  --rule-set renju \
  --center-opening \
  --games 10 \
  --max-moves 80 \
  --engine-timeout-turn-ms 1000 \
  --output-jsonl data/rapfi_policy_episodes.jsonl
```

The adapter sends board positions with Piskvork `BOARD` commands, converts
engine `x,y` replies to repository `row,col` coordinates, and still validates
the selected move with `GomokuBoard.is_legal_move()` before recording or
executing it. This keeps `board/` as the legality authority even when the
teacher is an external threat-search/alpha-beta engine.

## Generate MuJoCo Policy Episodes

Use MuJoCo collection when the selected move should be paired with rendered observations and SO-101 pick/place actions:

```bash
python -m scripts.generate_mujoco_policy_episodes \
  --checkpoint gomoku_ai/runs/<run-name>/checkpoints/best.pt \
  --games 10 \
  --simulations 64 \
  --device auto \
  --sample-moves \
  --temperature 1.0 \
  --temperature-moves 12 \
  --late-temperature 0.1 \
  --root-noise
```

`--robot-model` defaults to `so101`. Use `--robot-model kinematic` only for quick controller smoke tests.
Unless `--win-length` is provided, collection uses the checkpoint's stored
`win_length` so 5x5/9x9 smoke checkpoints do not silently switch rules.

The default 3.5 cm cell spacing produces a 49 cm playable span, which is not
fully reachable by the fixed-base SO-101 from one side. Production SO-101
collection should use the robot-scale board geometry below unless the scene is
changed to add a safe base repositioning mechanism:

```text
--cell-size 0.021 --stone-radius 0.007
```

This keeps the 15x15 rules and 225 move tokens unchanged while shrinking only
the simulated physical board. A nine-point reachability diagnostic measured a
maximum static placement error below the current 5.5 cm collection threshold;
the full dataset validator remains the authoritative execution gate.

For real dataset collection, do not run many games with the deterministic default policy. The default is kept deterministic for debugging, but `--games 100` without sampling can repeat the same opening and produce low-diversity data. Use:

- `--sample-moves`: select the teacher move from the MCTS policy distribution
- `--temperature 1.0`: keep early moves diverse
- `--temperature-moves 12`: use the high temperature for the opening
- `--late-temperature 0.1`: become more deterministic after the opening
- `--root-noise`: add root Dirichlet noise to MCTS
- `--seed`: make sampled collection reproducible when needed

Recommended smoke check:

```bash
python -m scripts.generate_mujoco_policy_episodes \
  --checkpoint gomoku_ai/runs/<run-name>/checkpoints/best.pt \
  --games 2 \
  --max-moves 5 \
  --simulations 64 \
  --robot-model so101 \
  --image-width 768 \
  --image-height 768 \
  --capture-phase-images \
  --sample-moves \
  --temperature 1.0 \
  --temperature-moves 12 \
  --late-temperature 0.1 \
  --root-noise
```

The default image size is `768x768` for each camera. Training collection must stay at `640x640` or higher unless storage is the limiting factor. The CLI rejects images below `640x640` by default because they are only useful for smoke-testing the pipeline, not for VLA training. Use `--allow-low-res-smoke` only with tiny `--max-moves` checks; those records are marked with `training_usable=false` and `split=smoke`.

The default cameras are `board_top,wrist_cam,robot_full`. `board_top_before` is the strategy image input, `wrist_cam_before` is the gripper-local manipulation input, and `robot_full` is the default QA/ablation view. By default each move records only `before` and `after` images for the selected cameras. This is the stable move-level format: `before` is the visual policy input, and `after` is the placement verification image. Use `--capture-phase-images` when the dataset should also include one rendered image set for each SO-101 pick/place phase.

SO-101 picks stones from visible black/white bowls outside the Gomoku board frame. `stone_supply_world(player)` returns the pick point inside the matching bowl, not a board-edge or board-intersection position.

Collection renders hide the interactive selection cursor. This prevents the before image from leaking the teacher-selected move. The selected cell appears only through labels such as `selected_move`, `<MOVE_k>`, and the after image once the stone is placed.

By default this writes:

```text
gomoku_ai/runs/<run-name>/data/best_mujoco_policy_episodes.jsonl
gomoku_ai/runs/<run-name>/data/assets/<game-id>/*.png
```

The same external-engine teacher can drive MuJoCo collection without an
AlphaZero checkpoint:

```bash
python -m scripts.generate_mujoco_policy_episodes \
  --engine-command /path/to/rapfi \
  --board-size 15 \
  --rule-set renju \
  --center-opening \
  --games 2 \
  --max-moves 5 \
  --random-prefix-moves 12 \
  --random-prefix-seed 20260719 \
  --skip-failed-games \
  --robot-model so101 \
  --cell-size 0.021 \
  --stone-radius 0.007 \
  --image-width 768 \
  --image-height 768 \
  --output-jsonl data/rapfi_mujoco/episodes.jsonl \
  --assets-dir data/rapfi_mujoco/assets
```

`--random-prefix-moves` intentionally records from a non-empty mid-game board.
In those records, `observation.is_first=false` unless the board is truly empty,
while `observation.is_first_recorded_frame=true` marks the first saved frame for
that episode. The recorded metadata also includes `prefix_moves`,
`board_ply_before`, and `board_ply_after`. Keep `--skip-failed-games` enabled
for long external-engine runs so a rare engine timeout or repo-legality rejection
skips that game instead of stopping the whole shard. The MuJoCo collector
restarts external Piskvork engines once per game so protocol state, illegal
replies, or process failures cannot bleed into later games.

The SO-101 controller records an IK-generated joint trajectory for:

```text
home -> pre_pick -> pick -> grasp -> lift -> pre_place -> place -> release -> retreat
```

The SO-101 action vectors use:

```text
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
```

`ee_trajectory` is still stored as debugging metadata, but the VLA execution label is the SO-101 joint sequence.

Collection now runs the same safety and inventory interfaces used by the viewer. Each move validates supply availability, target legality, workspace bounds, and action trace bounds before execution. The environment records black/white supply counts, active-stone state, grasp distance, place distance, final cell, and placement success.

SO-101 collection uses an `active_stone_body` and a constraint-style gripper lock while carrying the stone. This is stronger than visual-only attachment and is validated numerically, but it is not yet friction-only contact-stable grasping.

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

## Gomoku-aware VLA Fields

MuJoCo records use LeRobot/OpenVLA-friendly grouping:

- `observation.language_instruction`: non-leaking strategic task, for example `play the strongest legal Gomoku move as black`
- `observation.images.*`: all rendered camera image paths such as `board_top_before`, `wrist_cam_before`, `robot_full_before`, and after images
- `observation.phase_images`: optional phase-aligned image records, enabled by `--capture-phase-images`
- `observation.image_metadata`: image width, height, camera names, split, QA contact sheet, cursor-hidden flag, and whether the sample meets the minimum training resolution
- `observation.model_input`: fields allowed as model input: non-leaking instruction, `board_top_before` plus `wrist_cam_before`, board-before vector, current player, and pre-action robot/supply state
- `observation.state`: compatibility view of pre-action state only; it must not contain `target_cell` or `target_world_xyz`
- `observation.supervision.strategy`: strategic labels such as `selected_move`, `action_index`, `policy_probs`, `value`, and `board_after_flat`
- `observation.supervision.execution`: execution labels such as `target_cell`, `target_world_xyz`, supply/held-stone after state, and action sequence
- `robot_action.action`: SO-101 joint action sequence with names in `robot_action.action_names`
- `robot_action.ee_trajectory`: phase-labelled end-effector trajectory
- `robot_action.execution_success`: whether the scripted placement matched the selected cell
- `robot_action.placement_error_world` and `robot_action.placement_error_cell`: execution quality signals
- `robot_action.grasp_report`: supply bowl name, pick source, grasp/place/release distances, final cell, mode, and failure reason
- `robot_action.safety`: pick/place/trace validation results
- `robot_action.supply_before` and `robot_action.supply_after`: remaining black/white stones
- `robot_action.attachment_mode`: `constraint_style_active_stone` for SO-101 collection, `scripted_held_stone` for kinematic smoke tests
- `observation.supervision.execution.action`: action sequence grouped with its format and controller type for VLA loaders
- `observation.supervision.strategy.board_after_flat`: post-action board state for supervised consistency checks

## Intended Pipeline

```text
GomokuBoard state
-> CheckpointPolicy.predict()
-> move-level JSONL record
-> MuJoCo step or scripted robot controller
-> same record extended with observation and robot_action fields
```

The VLA dataset should keep model inputs separate from labels. Strategic teacher fields (`selected_move`, `policy_probs`, `value`) and execution fields (`target_cell`, `target_world_xyz`, `robot_action`, placement result) must not be inserted into the instruction or model-input state. This prevents coordinate leakage and lets policy mistakes and robot placement failures be evaluated independently.

For the final autoregressive OpenVLA plan, these same records should be converted into one target sequence:

```text
<MOVE_k> <ACT_1> <ACT_2> ... <ACT_N> <EOS>
```

- `<MOVE_k>`: one of 225 board-cell tokens, supervised by `selected_move` / `action_index`
- `<ACT_i>`: SO-101 collection uses `so101_joint_tokens_v1` tokens aligned to joint action waypoints; kinematic smoke tests still use `scripted_phase_v1`
- `value` or final-game `value_target`: optional auxiliary scalar supervision
- `policy_probs`: optional distillation target over the 225 move tokens

At inference time, the same decoder first generates the move token, then continues generating the action tokens conditioned on its own selected move.

Current status: the final OpenVLA tokenizer is not implemented yet. The JSONL record now stores move tokens, SO-101 joint waypoint tokens, and the continuous joint sequence needed to build real OpenVLA action chunks later.

## Export for OpenVLA-OFT / Custom Multi-view

Raw MuJoCo collection is not the final training directory. Keep it as the auditable source of truth, then export a training-facing dataset view:

```bash
python -m scripts.export_openvla_oft_dataset \
  --input-jsonl gomoku_ai/runs/<run-name>/data/best_mujoco_policy_episodes.jsonl \
  --output-dir gomoku_ai/runs/<run-name>/data/openvla_oft_multiview
```

The exporter writes:

```text
openvla_oft_multiview/
  metadata.json
  manifest.jsonl
  samples/<game-id>_step_0000/
    inputs/
      board_top_before.png
      wrist_cam_before.png
    qa/
      robot_full_before.png
      contact_sheet.png
```

This is the preferred path for OpenVLA-OFT/custom multi-view training because the model-facing manifest has exactly the intended inputs:

- `input.images.board_top_before`: strategic board view
- `input.images.wrist_cam_before`: gripper-local SO-101 view
- `input.language_instruction`: non-leaking prompt
- `input.state`: pre-action board/player/robot state

## Production Quality Validation

After a production SO-101 collection, validate the complete raw dataset before
exporting it:

```bash
python -m scripts.validate_so101_dataset \
  --input-jsonl data/<collection>/raw/episodes.jsonl \
  --expected-games 100 \
  --image-width 768 \
  --image-height 768 \
  --output-json data/<collection>/quality/raw_quality.json \
  --output-md data/<collection>/quality/raw_quality.md
```

The validator checks complete games, board transitions, legality, SO-101 action
shape and finite values, safety/grasp/place/execution results, input/target
separation, cursor hiding, image existence/integrity/resolution, duplicate
steps, and diversity indicators. It exits unsuccessfully when a required
condition fails so collection and export workflows can use it as a gate.

Validate the exported model-facing view separately:

```bash
python -m scripts.validate_openvla_manifest \
  --manifest data/<collection>/openvla_oft/manifest.jsonl \
  --expected-samples <raw-record-count> \
  --output-json data/<collection>/quality/export_quality.json
```

This second gate verifies exact model-input image keys, target/input separation,
SO-101 token and continuous-action targets, quality flags, unique sample IDs,
and all copied input/QA image paths.

The labels stay under `target`:

- `target.move_token` and `target.move`
- `target.text`: `<MOVE_k> <ACT_SO101_0000> ... <EOS>`
- `target.action.sequence`: continuous SO-101 joint trajectory
- `target.policy_probs` and `target.value`

`robot_full_before` and `contact_sheet` are copied under `qa.images` only. They are useful for dataset review and ablations, but they are not default model inputs.

By default the exporter filters out records that are not training-ready:

- `training_usable=false`
- illegal moves
- failed SO-101 execution
- failed safety/grasp reports
- non-SO-101 controllers

Use `--include-unusable` only for debugging exports and `--allow-non-so101` only for old kinematic smoke data. Those options should not be used for the main SO-101 VLA dataset.
