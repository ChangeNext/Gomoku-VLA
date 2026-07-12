# Development and Verification

## Environment

The package requires Python 3.10 or newer. Install the base project for board,
MuJoCo, rendering, and UI work:

```bash
python -m pip install -e .
```

Install learning or web evaluation dependencies when needed:

```bash
python -m pip install -e ".[learning]"
python -m pip install -e ".[learning,web]"
```

## Verification Ladder

Use the narrowest relevant check first, then widen verification.

```bash
python -m unittest discover -s tests -p "test_board.py"
python -m unittest discover -s tests -p "test_gomoku_ai.py"
python -m unittest discover -s tests -p "test_mujoco*.py"
python -m unittest discover -s tests -p "test_robot_control.py"
python -m unittest discover -s tests -p "test_vision.py"
python -m unittest discover -s tests
```

Useful manual smoke checks:

```bash
python -m scripts.play_cli
python -m scripts.interactive_play
python -m scripts.viewer_play --single-view
python -m scripts.render_snapshot
```

GUI checks require a display. Rendering and articulated-robot checks also
require MuJoCo and the vendored Menagerie assets.

## Change Requirements

A code change is done when:

- its observable acceptance criteria are met;
- a regression test covers new rules or a fixed defect;
- relevant focused tests pass;
- the full test suite passes, or an exact environment limitation is reported;
- the matching documentation reflects the resulting behavior;
- unrelated generated artifacts and training outputs are not included;
- no game-legality or robot-safety boundary has been weakened.

Visual changes additionally require inspection of the relevant viewer or a
rendered snapshot. Dataset changes require inspecting representative JSONL or
manifest samples and verifying that targets do not leak into model inputs.

## Repository Conventions

- Python uses type hints, four spaces, and explicit validation errors.
- Tests use standard-library `unittest` naming conventions.
- Board rules stay in `board/`; MuJoCo behavior stays in `simulation/`.
- Learned policies propose moves. Board and safety code authorize them.
- Training outputs belong under ignored run/data directories unless a small
  fixture is intentionally required by a test.

## Failure Handling

Classify a failure as an implementation defect, regression, dependency issue,
or environment limitation. Read the relevant error, fix the cause, and rerun
the failed command. Do not catch broad exceptions, relax assertions, or remove
checks solely to obtain a green run.
