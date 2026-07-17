# OpenVLA Training/Data Audit

## Goal

Audit and tighten the AlphaZero training, MuJoCo/SO-101 collection, OpenVLA
export, and fine-tuning preparation contracts before long GPU or production
collection runs.

## Non-goals

- Do not run long GPU training on this CPU machine.
- Do not download OpenVLA models or run OpenVLA fine-tuning.
- Do not change board legality semantics.

## Acceptance Criteria

- Training checkpoint metadata preserves the board rule parameters required by
  downstream collection.
- MuJoCo collection rejects or labels non-training smoke settings accurately.
- Export and fine-tuning preparation continue to block target leakage.
- Focused tests and the full `unittest` suite pass.

## Risks

- Small-board smoke checkpoints can silently collect with a wrong win length if
  metadata is incomplete.
- Low-resolution smoke data can accidentally be treated as production VLA data.
- OpenVLA preparation must not reintroduce target coordinates into model input.

## Validation Commands

```bash
python -m unittest discover -s tests -p "test_gomoku_ai.py"
python -m unittest discover -s tests -p "test_mujoco_policy_collection.py"
python -m unittest discover -s tests -p "test_openvla*.py"
python -m unittest discover -s tests
```
