# Documentation Map

This directory is the project knowledge base. `AGENTS.md` is only the entry
map and operating contract; detailed behavior belongs here and in executable
tests.

## Start Here

- [Architecture](architecture.md): subsystem boundaries and end-to-end flows
- [Development and verification](development.md): setup, commands, and done criteria
- [Current roadmap](roadmap.md): implemented capabilities and remaining work

## Domain Documentation

- [Game rules](game_rules.md): `GomokuBoard`, free rules, Renju, and legality
- [AlphaZero self-play](alphazero_self_play.md): MCTS and self-play data flow
- [Training methodology](alphazero_training_methodology.md): training runs and promotion
- [MuJoCo environment](mujoco_environment.md): scene, coordinates, and robot models
- [Data collection](data_collection.md): policy episodes and observation/label separation
- [VLA fine-tuning](vla_finetuning.md): intended OpenVLA-style training contract
- [Robot safety](robot_safety.md): current enforced checks and known safety gaps
- [Vision](vision.md): calibrated grid detector and real-camera limitations
- [Evaluation](evaluation.md): strategy, human, perception, and execution metrics
- [Human-playable environment](human_playable_environment.md): CLI and viewer tools
- [Sim-to-real](sim2real.md): interfaces and transfer work
- [Study notes](study.md): short research framing

## Planning

Long-running cross-cutting work should use:

```text
docs/plans/active/       plans currently being executed
docs/plans/completed/    verified historical plans
```

Plan files should record the goal, non-goals, acceptance criteria, affected
modules, risks, implementation progress, decisions, and validation evidence.

## Which Source Wins?

- Game legality: `board/gomoku.py` plus `tests/test_board.py`
- Public behavior: implementation plus the matching tests
- Architectural intent and research constraints: this documentation
- Generated dataset truth: its manifest/metadata and exporter implementation

If code, tests, and documentation disagree, investigate rather than silently
choosing one. Correct the stale artifact in the same change.
