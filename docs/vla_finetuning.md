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

1. Generate strategic demonstrations with AlphaZero/MCTS or an external
   Piskvork engine such as Rapfi/Embryo:
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

For an external-engine smoke path, first collect MuJoCo records with
`scripts.generate_mujoco_policy_episodes --engine-command ...`, then export and
prepare the training package:

```bash
python -m scripts.export_openvla_oft_dataset \
  --input-jsonl data/rapfi_mujoco/episodes.jsonl \
  --output-dir data/rapfi_mujoco/openvla_oft

python -m scripts.split_openvla_manifest \
  --manifest data/rapfi_mujoco/openvla_oft/manifest.jsonl \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 20260719

python -m scripts.prepare_openvla_finetuning \
  --manifest data/rapfi_mujoco/openvla_oft/manifest_train.jsonl \
  --output-dir data/rapfi_mujoco/openvla_oft_prep \
  --base-model openvla/openvla-7b \
  --stage move_only
```

The split command groups by `source.game_id`, not by individual rows. This
keeps adjacent positions from the same game out of multiple splits and avoids
overstating validation or test accuracy. Use `manifest_train.jsonl` for
fine-tuning, `manifest_val.jsonl` for model selection, and keep
`manifest_test.jsonl` untouched for final reporting.

This repository still prepares a trainer-neutral OpenVLA/OFT package; it does
not download OpenVLA weights or launch a multi-GPU fine-tuning job locally.

## Local Smoke Fine-tuning

For a small local integration check, use the manifest trainer:

```bash
python -m scripts.train_openvla_manifest \
  --manifest data/rapfi_collection_2026-07-19_10-30-17_v2/openvla_oft/manifest_train.jsonl \
  --eval-manifest data/rapfi_collection_2026-07-19_10-30-17_v2/openvla_oft/manifest_val.jsonl \
  --prep-dir data/rapfi_collection_2026-07-19_10-30-17_v2/openvla_oft_prep \
  --output-dir data/rapfi_collection_2026-07-19_10-30-17_v2/openvla_move_only_smoke \
  --base-model openvla/openvla-7b \
  --stage move_only \
  --max-steps 1 \
  --batch-size 1 \
  --lora-rank 4
```

This uses the HuggingFace OpenVLA model with LoRA and trains only the
autoregressive move-token target from the custom manifest. It is a smoke test
for the custom data path, not a useful policy run; the Rapfi collection above
contains only a few samples.

If `--use-4bit` fails with a bitsandbytes `.to is not supported for 4-bit`
error, check the active Python environment before training:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
python -m bitsandbytes
```

4-bit LoRA requires both PyTorch CUDA and a CUDA-enabled bitsandbytes library.
If bitsandbytes loads `libbitsandbytes_cpu.so`, reinstall or fix that
environment before using `--use-4bit`; otherwise rerun the smoke path without
`--use-4bit` on a GPU with enough memory.

## GPU Training Preparation

This repository can prepare the files needed for a later GPU-side OpenVLA/OFT
run without importing OpenVLA or starting training:

```bash
python -m scripts.prepare_openvla_finetuning \
  --manifest data/<collection>/openvla_oft/manifest.jsonl \
  --output-dir data/<collection>/openvla_oft_prep \
  --base-model openvla/openvla-7b \
  --stage move_only
```

The preparation command writes:

- `special_tokens.json`: `<MOVE_000>` through `<MOVE_224>`, discovered
  `<ACT_SO101_...>` tokens, and tokenizer notes.
- `run_config.json`: trainer-neutral requirements for inputs, targets,
  tokenizer resizing, losses, and inference safety.
- `training_prompt.md`: a GPU-machine implementation prompt for connecting the
  manifest to an OpenVLA/OFT trainer.
- `dataset_preview.jsonl`: a small non-leaking input/target preview.

### Token and Embedding Requirements

Move tokens are semantically new, so they should not reuse OpenVLA action-bin
tokens. Add them as special tokens, resize the embedding matrix and language
model head, and keep the newly added embedding and `lm_head` rows trainable even
when the rest of the model is frozen or LoRA-tuned.

### Loss Recommendation

The first OpenVLA run should use discrete action tokens and autoregressive
cross-entropy for both move and action generation:

```text
loss = move_token_ce + lambda_action * action_token_ce
```

Avoid mixing move-token CE with continuous joint MSE in the first integration,
because the scales are different and make lambda tuning harder. Continuous
action heads can be added later after the token path works.

### Inference Safety

Training samples only contain legal moves, but inference must still mask illegal
move logits before selecting `<MOVE_k>`. Parse the board state, use
`board.gomoku.GomokuBoard` as the legality authority, mask occupied and
rule-illegal cells, then pass the selected move through
`robot_control.RobotSafetyController` before execution.

### Practical Staging

Use staged training to isolate failures:

```text
1. move_only:
   board_top_before + instruction -> <MOVE_k>

2. teacher_move_then_action:
   board_top_before + wrist_cam_before + teacher <MOVE_k> -> <ACT_SO101_...>

3. move_then_action_tokens:
   board_top_before + wrist_cam_before + instruction -> <MOVE_k> <ACT_SO101_...> <EOS>
```

For data efficiency, pretrain the move-token path with many simulated
`(board_state, board_top_before, selected_move)` samples before spending real
robot time. Use real robot data primarily to adapt wrist/action generation and
sim-to-real execution details.
