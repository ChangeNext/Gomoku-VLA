# VLA Fine-tuning

Project goal: fine-tune a Gomoku-aware VLA that decides the next strong legal move from board images. Do not train the main model with instructions that reveal `row`, `column`, `target_cell`, or `target_world_xyz`.

The main fine-tuning path is strategy-first. The model input is a board observation plus a non-leaking instruction such as `play the strongest legal Gomoku move as black`.

## Main: Gomoku-aware VLA

Target base model: OpenVLA. Use OpenVLA as the vision-language-action backbone, but extend the generated token sequence so the same autoregressive decoder emits the Gomoku move first and robot action tokens second. This preserves the feeling of one VLA choosing and playing the move.

Allowed inputs:

- board image observations such as `board_top_before`
- gripper-local observations such as `wrist_cam_before`
- non-leaking instruction
- current player
- optional pre-action board vector during simulation training

Supervision labels:

- `selected_move`
- `action_index`
- `policy_probs`
- `value` or final-game `value_target`
- optional execution labels: `target_world_xyz` and robot action sequence

## Autoregressive OpenVLA Fine-tuning Recipe

1. Generate strategic demonstrations with AlphaZero/MCTS:
   `board_before -> selected_move/policy_probs/value`.
2. Add discrete move tokens:
   `<MOVE_0>` through `<MOVE_224>` for the 15x15 board cells.
3. Convert each selected move into an executable SO-101 joint action target:
   `selected_move -> target_world_xyz -> IK -> so101_joint_trajectory_v1`.
4. Build one target token sequence:
   `<MOVE_k> <ACT_SO101_0000> <ACT_SO101_0001> ... <EOS>`.
5. Fine-tune the autoregressive decoder with:
   `loss = move_token_loss + lambda_action * action_token_loss + lambda_value * value_loss`.
6. Keep `policy_probs`, `selected_move`, and `value/value_target` for distillation and evaluation.

Prompt template:

```text
In: Play the strongest legal Gomoku move as black.
Out:
```

Do not use:

```text
In: place at row 7 column 7
```

## Auxiliary: Coordinate Execution

Coordinate-following pick/place can still be trained as a supporting skill, but it is not the project objective. Any dataset row that says `place at row X column Y` should be marked as execution-only and excluded from strategic VLA training. Strategic OpenVLA rows should use non-leaking prompts plus `<MOVE_k> <ACT_SO101_0000> ... <EOS>` targets, where the move token is the strategic label and the SO-101 joint tokens execute that move.

Safety layer는 모든 단계에서 VLA 외부에 유지한다.

## Feasibility

This is possible, but it should be treated as a staged research system rather than a one-shot fine-tune. The strongest path is:

- first prove move-token prediction from board images,
- then prove action-token generation after a teacher move token,
- then train the full `<MOVE_k> <ACT_...>` sequence,
- then evaluate strategy accuracy and physical placement accuracy separately.

The main risks are data quality, action representation, and sim-to-real transfer, not whether the architecture is conceptually valid.

Default simulation inputs for the SO-101 path are `board_top_before` and `wrist_cam_before`. `robot_full_before` should still be stored, but it is a QA/ablation view rather than the default VLA input.

## OpenVLA-OFT / Custom Multi-view Dataset

Use `scripts.export_openvla_oft_dataset` after MuJoCo collection. It converts the raw JSONL plus asset folder into a custom multi-view manifest designed for OpenVLA-OFT-style training:

```text
input:
  language_instruction: play the strongest legal Gomoku move as black
  images:
    board_top_before: strategic board input
    wrist_cam_before: SO-101 gripper-local input
  state:
    board_flat
    current_player_value
    robot_state

target:
  text: <MOVE_k> <ACT_SO101_0000> ... <EOS>
  move_token
  move
  action:
    names: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    sequence: SO-101 joint waypoints
```

This is intentionally not a single flat image folder. Each sample has its own `inputs/` and `qa/` directories, while `manifest.jsonl` defines the exact model input order. The default image order is:

```text
board_top_before, wrist_cam_before
```

`robot_full_before` is exported only under `qa.images`; it should not be passed as a default OpenVLA-OFT model input. If later ablation shows it helps, add it explicitly through a new manifest config rather than accidentally mixing it into the main dataset.

The included lightweight loader is `gomoku_ai.openvla_oft_dataset.OpenVLAOFTManifestDataset`. It is a bridge for custom training code; a full OpenVLA-OFT trainer still needs to map:

- `input.images` to the model's multi-image preprocessing path,
- `target.text` to the autoregressive tokenizer,
- `target.action.sequence` to the SO-101 action tokenizer or continuous action head.
