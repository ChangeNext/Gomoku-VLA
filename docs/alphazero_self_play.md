# AlphaZero-style Self-Play

## Goal

AlphaZero-style training is the learning-based path for making the Gomoku policy stronger without hand-written rule evaluation. The first stage is board-only:

```text
15x15 board state -> policy/value model -> MCTS -> row,col move
```

Camera recognition and robot control should feed into and consume this policy later, but they should not be part of the first strategy-learning loop. This keeps strategy learning fast, deterministic, and easy to debug.

## Current Implementation

The initial scaffold is in `gomoku_ai/`.

- `encoding.py`: converts `GomokuBoard` to current-player tensor input and maps `(row, col)` actions to flat indices.
- `model.py`: defines the `PolicyValueModel` interface plus `UniformPolicyValueModel` for plumbing tests.
- `mcts.py`: runs PUCT-style Monte Carlo Tree Search and returns an improved policy over legal moves.
- `self_play.py`: generates self-play samples containing `state`, `policy_target`, `value_target`, and `player`.
- `replay_buffer.py`: stores self-play samples and builds mini-batches.
- `torch_model.py`: defines a small PyTorch CNN with policy and value heads.
- `train.py`: runs self-play, trains the network, and writes checkpoints.

Run a smoke test game:

```bash
python -m scripts.generate_self_play --board-size 5 --win-length 4 --simulations 8
```

Run a small training smoke test:

```bash
python -m scripts.train_alphazero \
  --board-size 5 \
  --win-length 4 \
  --iterations 1 \
  --games 2 \
  --simulations 8 \
  --epochs 1
```

This writes `checkpoints/alphazero_latest.pt` by default. Use small boards and low simulation counts for pipeline checks; use larger boards, more self-play games, and more MCTS simulations only after the loop is stable.

Continue training from an existing checkpoint:

```bash
python -m scripts.train_alphazero \
  --board-size 9 \
  --win-length 5 \
  --iterations 12 \
  --games 30 \
  --simulations 96 \
  --epochs 8 \
  --batches-per-epoch 64 \
  --batch-size 256 \
  --device cuda \
  --initial-checkpoint checkpoints/alphazero_9x9_mid.pt \
  --checkpoint checkpoints/alphazero_9x9_long.pt
```

Training uses random rotation/flip augmentation by default so each sampled batch sees board-equivalent positions in different orientations. `--epochs` controls passes through the training loop, and `--batches-per-epoch` controls how many replay batches are sampled per epoch. Use `--no-augment` only for debugging exact sample contents.

Play against a trained checkpoint:

```bash
python -m scripts.play_ai_cli \
  --checkpoint checkpoints/alphazero_9x9_long.pt \
  --simulations 32
```

Use `--human white` to let the AI play the first move as black. The CLI reads the board size from the checkpoint and uses `--win-length 5` by default.

Run all tests:

```bash
python -m unittest discover -s tests
```

## Training Data

Each self-play turn stores:

```text
state:          [3, board_size, board_size]
policy_target: MCTS visit distribution over board_size * board_size actions
value_target:  +1 win, -1 loss, 0 draw from that player's perspective
```

The input is encoded from the current player's perspective:

- channel 0: current player's stones
- channel 1: opponent stones
- channel 2: current player color indicator

## Next Steps

1. Add evaluator matches between the latest model and the previous checkpoint.
2. Keep only checkpoints that beat the previous best by a target win-rate threshold.
3. Add replay buffer save/load so long runs can resume.
4. Increase model capacity with residual blocks once 9x9 training is stable.
5. Connect the selected `(row, col)` move to `simulation.GomokuMujocoEnv.step()`.

## Current Limits

The current trainer is intentionally small. It proves that self-play samples, policy/value training, and checkpoint saving work end to end. It is not expected to produce a strong 15x15 player yet. For serious training, start with `5x5` or `9x9`, increase `--games` and `--simulations`, then add evaluator-gated checkpoint promotion.

## Integration With Vision and Robot Control

The final system should remain modular:

```text
camera image -> vision board detector -> GomokuBoard state
GomokuBoard state -> AlphaZero policy -> row,col
row,col -> board_to_world -> robot controller or VLA action
```

This separation makes it possible to evaluate board recognition errors, policy mistakes, and robot placement failures independently.
