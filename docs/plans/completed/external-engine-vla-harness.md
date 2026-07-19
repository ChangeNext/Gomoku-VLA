# External Engine VLA Harness

## Goal

Use Rapfi/Embryo-style Gomocup/Piskvork engines as strategic teachers for
board-only and MuJoCo VLA data collection without vendoring GPL engine code.

## Non-goals

- Do not bundle Rapfi, Embryo, or engine weights in this repository.
- Do not bypass repository Gomoku/Renju legality checks.
- Do not implement a full OpenVLA trainer in this repository.

## Acceptance Criteria

- A Piskvork-compatible engine command can produce `MovePrediction` records.
- Engine `x,y` coordinates are converted to repository `row,col` coordinates.
- Engine moves are checked with `GomokuBoard.is_legal_move()`.
- `scripts.generate_policy_episodes` accepts either `--checkpoint` or
  `--engine-command`.
- `scripts.generate_mujoco_policy_episodes` accepts either `--checkpoint` or
  `--engine-command`.
- VLA docs show collection, export, and preparation commands.

## Validation

```bash
python -m unittest discover -s tests -p "test_gomoku_ai.py"
python -m unittest discover -s tests -p "test_episode_recorder.py"
python -m unittest discover -s tests -p "test_mujoco_policy_collection.py"
```

Full-suite status should be recorded in the final implementation report.
