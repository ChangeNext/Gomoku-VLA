# Evaluation

## Metrics

- game win rate
- illegal move rate
- placement success rate
- placement error
- policy latency
- robot execution latency
- collision or safety intervention count

## Human Checkpoint Evaluation

Use the browser evaluation server when a `best.pt` checkpoint needs direct human playtesting.
The server fixes one checkpoint for the session, lets each player choose black or white, and
writes completed games to JSONL so the results can be aggregated later.

Install the web dependencies together with the learning stack:

```bash
pip install -e ".[learning,web]"
```

Run a local evaluation UI:

```bash
python -m scripts.human_eval_simple_server \
  --checkpoint gomoku_ai/runs/<run-name>/checkpoints/best.pt \
  --simulations 32 \
  --host 0.0.0.0 \
  --port 8000
```

Then share `http://<host-ip>:8000` with testers on the same network. Each completed game appends
one detailed JSON record and one CSV summary row with `player_id`, checkpoint path, human/AI colors,
result, move count, full move history, board settings, and timestamps.

By default, results are written under the training run that owns the checkpoint:

```text
gomoku_ai/runs/<run-name>/evaluation/best_human_eval.jsonl
gomoku_ai/runs/<run-name>/evaluation/best_human_eval.csv
```

Primary human-evaluation metrics:

- human win rate: `human_win / total_games`
- AI win rate: `ai_win / total_games`
- draw rate
- black/white split for the human player
- mean move count
- per-checkpoint results when comparing multiple `best.pt` versions
