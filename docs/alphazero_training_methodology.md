# AlphaZero Training Methodology

Project goal: train strategic move choice for Gomoku. For VLA data, move coordinates and policy/value targets must remain labels, not text or state fields exposed as model input.

## Current Approach

The Gomoku AI uses an AlphaZero-style loop:

```text
board state -> ResNet policy/value network -> MCTS -> self-play move
self-play games -> replay buffer -> policy/value optimization
```

The default target is 15x15 Renju. Black must open at the center, black 3-3/4-4/overline moves are excluded from legal moves, black wins with exactly five, and white wins with five or more. Free-rule small boards remain available for fast smoke tests.

The default model is a residual CNN rather than a Transformer. Gomoku is a small 2D grid game with strong local tactical patterns, directionality, and translation symmetry. A ResNet CNN gives useful inductive bias and cheaper MCTS inference. Transformer-style models can be evaluated later, but they are not the default until the self-play, evaluator, and replay pipeline are stable.

## Run Outputs

Every training run is grouped under one folder:

```text
gomoku_ai/runs/<run-name>/
  config.json
  checkpoints/latest.pt
  checkpoints/best.pt
  metrics/history.csv
  metrics/evaluation.csv
  plots/training.png
  plots/policy_heatmap_empty.png
  replay/replay_buffer.pkl
```

`history.csv` records total loss, policy loss, value loss, target entropy, replay size, samples added, and optional evaluator promotion metrics. `policy_heatmap_empty.png` visualizes the network's current empty-board move prior, which is useful for spotting obvious policy collapse or center/corner bias. `replay_buffer.pkl` lets a run resume without throwing away self-play data.

The trainer displays nested `tqdm` progress bars for iterations, self-play games, optimization batches, and evaluator games. The top-level bar reports the latest loss, replay size, and evaluator score. Use `--no-progress` for redirected logs or non-interactive jobs.

## Interpreting Metrics

Training loss is not a fixed validation loss. The target distribution changes because every iteration creates new self-play data. Use it as a stability signal, not as the only strength measure.

- `policy_loss`: how well the model matches MCTS visit distributions.
- `value_loss`: how well the model predicts game outcome.
- `policy_target_entropy`: how sharp or diffuse the MCTS targets are.
- `replay_size`: whether the run is accumulating enough diverse positions.

Actual strength should be checked with evaluator-gated `best.pt` promotion, `scripts.evaluate_checkpoint`, and direct play via `scripts.play_ai_cli`.

## Small Boards vs 15x15

Small free-rule boards are used to debug the loop quickly. 15x15 Renju is the target board size, but it is much slower:

- action space grows from 81 to 225;
- games are longer;
- MCTS needs more simulations to resolve tactical threats;
- replay capacity must be larger.

For 15x15, use a larger model and more replay than smoke runs. The current stronger baseline uses 6 input planes, 256 channels, and 16 residual blocks. Increase `--games`, `--simulations`, and `--replay-capacity` before increasing model size further.

## Recommended 15x15 Command

```bash
python -m scripts.train_alphazero \
  --board-size 15 \
  --win-length 5 \
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
  --channels 256 \
  --res-blocks 16 \
  --input-channels 6 \
  --evaluation-games 20 \
  --evaluation-simulations 64
```

Run this in `tmux` for long training. Check progress with:

`--self-play-batch-size` keeps the same search depth and model size, but advances multiple self-play games together so MCTS leaf evaluations are sent through the policy/value network as batches. Start with 4 or 8 on CUDA and increase only if GPU memory remains comfortable.

```bash
tail -n 10 gomoku_ai/runs/<run-name>/metrics/history.csv
```

## Testing A Checkpoint

Direct human play:

```bash
python -m scripts.play_ai_cli \
  --checkpoint gomoku_ai/runs/<run-name>/checkpoints/latest.pt \
  --human black \
  --simulations 64
```

Checkpoint evaluation:

```bash
python -m scripts.evaluate_checkpoint \
  --candidate gomoku_ai/runs/<new-run>/checkpoints/latest.pt \
  --baseline gomoku_ai/runs/<old-run>/checkpoints/latest.pt \
  --games 50 \
  --simulations 64 \
  --rule-set renju \
  --center-opening \
  --output-csv gomoku_ai/runs/<new-run>/metrics/evaluation.csv \
  --promote-to gomoku_ai/runs/<new-run>/checkpoints/best.pt
```

## Tactical Guard

Self-play does not override MCTS moves with hand-written tactics. Evaluation and human-play move selection still checks simple tactics before MCTS:

1. play an immediate winning move if available;
2. block the opponent's immediate winning move if needed;
3. prefer open-four, closed-four plus open-three, and double-open-three threats;
4. block the opponent's comparable threat;
5. otherwise use MCTS.

This does not replace learning. It prevents obvious one-move blunders while the model is still weak.
