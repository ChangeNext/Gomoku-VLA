# Repository Guidelines

Project goal: build a Gomoku-aware VLA that reads the board, chooses a strong legal move, and then executes that move. Do not treat coordinate-following placement as the main objective. Target row/column, target world pose, and robot trajectories are labels or metadata, not model-input instructions.

## Project Structure & Module Organization

This is a Python 3.10+ project for a MuJoCo-backed Gomoku simulation and future VLA robot work. Core game logic lives in `board/`, especially `board/gomoku.py`. MuJoCo environment code lives in `simulation/`, with `simulation/mujoco_gomoku_env.py` building and updating the scene. User-facing entry points are in `scripts/`: `play_cli.py`, `interactive_play.py`, `viewer_play.py`, and `render_snapshot.py`. Tests are in `tests/` and mirror the package areas they validate. Design notes and roadmap material belong in `docs/`. Top-level generated or reference assets include `gomoku_scene.xml`, `gomoku_snapshot.png`, and `gomoku_snapshot.ppm`.

## Build, Test, and Development Commands

- `python -m unittest discover -s tests`: run the full test suite.
- `python -m scripts.play_cli`: play Gomoku from the terminal.
- `python -m scripts.interactive_play`: run the clickable human-playable UI.
- `python -m scripts.viewer_play --single-view`: open the MuJoCo viewer with one camera.
- `python -m scripts.render_snapshot`: render a PNG snapshot of the scene.

Install the package dependencies from `pyproject.toml` in a virtual environment before running MuJoCo scripts.

## Coding Style & Naming Conventions

Use straightforward Python with type hints, as in the existing modules. Follow 4-space indentation, `snake_case` for functions, variables, and modules, and `PascalCase` for classes and enums. Keep board-state rules in `board/`; keep MuJoCo rendering, camera, robot, and coordinate conversion behavior in `simulation/`. Prefer small methods with explicit validation and clear `ValueError` messages for invalid moves or coordinates.

## Testing Guidelines

Tests use the standard library `unittest` framework. Name files `test_*.py`, test classes `*Test`, and methods `test_*`. Add focused tests for new board rules, coordinate conversions, robot target behavior, and scene updates. Run `python -m unittest discover -s tests` before submitting changes.

## Commit & Pull Request Guidelines

The current history uses short imperative commit subjects, for example `Improve Gomoku robot scene UI`. Keep subjects concise and describe the user-visible or architectural change. Pull requests should include a summary, test results, and any relevant screenshots or rendered snapshots for UI, viewer, or scene changes. Link related docs updates when changing roadmap, architecture, safety, data collection, or VLA behavior.

## Agent-Specific Instructions

When adding features, update the relevant `docs/` page alongside code. Preserve the separation between simulation MVP, rule/search baselines, learning policies, VLA inference, and safety controls. Never remove explicit safety checks such as invalid move rejection, workspace limits, collision checks, or emergency-stop planning when robot actions are introduced.
