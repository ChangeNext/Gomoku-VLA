# AlphaZero-style Self-Play

Project goal: the VLA should play Gomoku well, not merely place a stone at a commanded coordinate. AlphaZero/MCTS outputs are strategic supervision labels for move choice, while robot targets remain execution labels.

## Goal

AlphaZero-style training is the learning-based path for making the Gomoku policy stronger without hand-written rule evaluation. The first stage is board-only:

```text
15x15 Renju board state -> policy/value model -> MCTS -> row,col move
```

Camera recognition and robot control should feed into and consume this policy later, but they should not be part of the first strategy-learning loop. This keeps strategy learning fast, deterministic, and easy to debug.

## Current Implementation

The initial scaffold is in `gomoku_ai/`.

- `encoding.py`: converts `GomokuBoard` to current-player tensor input and maps `(row, col)` actions to flat indices.
- `model.py`: defines the `PolicyValueModel` interface, including batched prediction, plus `UniformPolicyValueModel` for plumbing tests.
- `mcts.py`: runs PUCT-style Monte Carlo Tree Search and returns an improved policy over legal moves. It can batch leaf network evaluations across multiple simultaneous games.
- `self_play.py`: generates self-play samples containing `state`, `policy_target`, `value_target`, and `player`, with optional batched game generation.
- `replay_buffer.py`: stores self-play samples and builds mini-batches.
- `torch_model.py`: defines a ResNet-style PyTorch policy/value network plus legacy checkpoint loading.
- `train.py`: runs self-play, trains the network, evaluates promotion candidates, and writes structured run outputs.
- `tactics.py`: catches immediate wins and one-move defensive blocks for evaluation/play move selection.

The serious training target is 15x15 Renju: black opens at the center, black 3-3/4-4/overline moves are illegal, black wins with exactly five, and white wins with five or more. Small free-rule boards are still useful for smoke tests.

Run a smoke test game:

```bash
python -m scripts.generate_self_play --board-size 5 --win-length 4 --simulations 8
```

Run a small training smoke test:

```bash
python -m scripts.train_alphazero \
  --board-size 5 \
  --win-length 4 \
  --rule-set free \
  --no-center-opening \
  --iterations 1 \
  --games 2 \
  --simulations 8 \
  --epochs 1
```

This writes a timestamped run under `gomoku_ai/runs/` by default. Use small boards and low simulation counts for pipeline checks; use larger boards, more self-play games, and more MCTS simulations only after the loop is stable.

Start a 15x15 Renju run:

```bash
python -m scripts.train_alphazero \
  --iterations 80 \
  --games 80 \
  --self-play-batch-size 8 \
  --simulations 256 \
  --epochs 8 \
  --batches-per-epoch 128 \
  --batch-size 512 \
  --replay-capacity 500000 \
  --learning-rate 3e-4 \
  --device cuda \
  --amp \
  --run-name 2026-06-23_15x15_renju_resnet \
  --evaluation-games 20 \
  --evaluation-simulations 64
```

Training uses random rotation/flip augmentation by default so each sampled batch sees board-equivalent positions in different orientations. `--epochs` controls passes through the training loop, and `--batches-per-epoch` controls how many replay batches are sampled per epoch. `--self-play-batch-size` controls how many self-play games advance together; larger values improve GPU utilization by batching MCTS leaf inference without reducing MCTS simulations. Use `--no-augment` only for debugging exact sample contents. The default network is a deeper residual CNN with AdamW, optional CUDA AMP, and gradient clipping because Gomoku/Renju is a small 2D grid game with strong local pattern and translation structure; Transformer-style models are left for later comparison experiments after the AlphaZero loop and evaluator are stable.

Run outputs are grouped together:

```text
gomoku_ai/runs/<run-name>/config.json
gomoku_ai/runs/<run-name>/checkpoints/latest.pt
gomoku_ai/runs/<run-name>/metrics/history.csv
gomoku_ai/runs/<run-name>/plots/training.png
gomoku_ai/runs/<run-name>/replay/replay_buffer.pkl
```

Resume a run with `--resume-run gomoku_ai/runs/<run-name>`. This restores `latest.pt` and the replay buffer so long training keeps accumulated self-play samples.

Play against a trained checkpoint:

```bash
python -m scripts.play_ai_cli \
  --checkpoint gomoku_ai/runs/2026-06-23_15x15_renju_resnet/checkpoints/latest.pt \
  --simulations 32
```

Use `--human white` to let the AI play the first move as black. The CLI reads board size and rule metadata from the checkpoint and uses `--win-length 5` by default.

Use the shared inference API when connecting a checkpoint to simulation, robot control, or dataset generation:

```python
from gomoku_ai.inference import CheckpointPolicy

policy = CheckpointPolicy(
    "gomoku_ai/runs/2026-06-23_15x15_renju_resnet/checkpoints/best.pt",
    device="auto",
    simulations=64,
)
board = policy.new_board()
prediction = policy.predict(board)
row, col = prediction.move
```

`prediction.policy` is the MCTS-improved visit distribution over all board cells, `prediction.value` is the network value estimate for the current player before search, and `prediction.used_tactical_move` marks immediate win/block overrides. Downstream data collectors should store these fields with each selected move so later VLA datasets can distinguish the strategic teacher signal from robot execution results.

Run all tests:

```bash
python -m unittest discover -s tests
```

## Training Data

Each self-play turn stores:

```text
state:          [input_channels, board_size, board_size]
policy_target: MCTS visit distribution over board_size * board_size actions
value_target:  +1 win, -1 loss, 0 draw from that player's perspective
```

The input is encoded from the current player's perspective:

- channel 0: current player's stones
- channel 1: opponent stones
- channel 2: current player color indicator

## Next Steps

1. Run long 15x15 Renju training in `tmux`.
2. Increase evaluator games after the first stable checkpoint.
3. Tune `--self-play-batch-size` for the active GPU memory budget.
4. Connect the selected `(row, col)` move to `simulation.GomokuMujocoEnv.step()`.

## Current Limits

The trainer now targets 15x15 Renju, but strong play still depends on long self-play runs and enough evaluator games. Use `5x5` or `9x9` free-rule runs only for pipeline checks.

## Integration With Vision and Robot Control

The final system should remain modular:

```text
camera image -> vision board detector -> GomokuBoard state
GomokuBoard state -> AlphaZero policy -> row,col
row,col -> board_to_world -> robot controller or VLA action
```

This separation makes it possible to evaluate board recognition errors, policy mistakes, and robot placement failures independently.
