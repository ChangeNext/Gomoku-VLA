from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from PIL import Image

from board import Player
from gomoku_ai.episode_recorder import EpisodeStepRecord, append_episode_record
from gomoku_ai.inference import MovePrediction

from .mujoco_gomoku_env import GomokuMujocoEnv
from .scripted_robot import build_pick_place_action


class MovePredictor(Protocol):
    def predict(self, board: object) -> MovePrediction:
        ...


def default_mujoco_episode_output_path(checkpoint: str | Path) -> Path:
    checkpoint_path = Path(checkpoint)
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    return run_dir / "data" / f"{checkpoint_path.stem}_mujoco_policy_episodes.jsonl"


def collect_mujoco_policy_episode(
    env: GomokuMujocoEnv,
    predictor: MovePredictor,
    output_jsonl: str | Path,
    assets_dir: str | Path,
    *,
    game_id: str | None = None,
    episode_index: int = 0,
    policy_source: str = "alphazero",
    checkpoint: str | None = None,
    max_moves: int | None = None,
    cameras: tuple[str, ...] = ("top", "iso", "robot_full"),
    image_width: int = 640,
    image_height: int = 640,
) -> list[EpisodeStepRecord]:
    if env.board.winner is not None:
        raise ValueError("cannot collect an episode from a finished environment")
    max_steps = max_moves or env.board.size * env.board.size
    if max_steps <= 0:
        raise ValueError("max_moves must be positive")

    output_path = Path(output_jsonl)
    asset_root = Path(assets_dir)
    episode_id = game_id or str(uuid.uuid4())
    episode_asset_dir = asset_root / episode_id
    episode_asset_dir.mkdir(parents=True, exist_ok=True)
    records: list[EpisodeStepRecord] = []

    while env.board.winner is None and env.board.move_count < max_steps:
        step = len(records)
        board_before = env.board.copy_state()
        player = env.board.current_player
        prediction = predictor.predict(env.board)
        row, col = prediction.move
        legal = env.board.is_legal_move(row, col)
        env.set_selection(row, col, update_robot_target=False)
        images_before = _render_images(env, episode_asset_dir, step, "before", cameras, image_width, image_height)
        robot_action = build_pick_place_action(env, (row, col), player)
        error = None
        if legal:
            env.step((row, col), update_robot_target=True)
        else:
            error = f"illegal move: row={row}, col={col}"
        images_after = _render_images(env, episode_asset_dir, step, "after", cameras, image_width, image_height)
        robot_action["execution_success"] = bool(
            legal
            and env.board.is_on_board(row, col)
            and env.board.grid[row][col] == int(player.value)
            and robot_action["placement_error_cell"] == 0.0
        )

        will_stop = env.board.winner is not None or not legal or env.board.move_count >= max_steps
        observation = _build_observation(
            env,
            player,
            (row, col),
            robot_action,
            board_state=board_before,
            images_before=images_before,
            images_after=images_after,
            episode_index=episode_index,
            frame_index=step,
            is_first=step == 0,
            is_last=will_stop,
            is_terminal=env.board.winner is not None or not legal,
        )
        record = EpisodeStepRecord(
            game_id=episode_id,
            step=step,
            timestamp=_now_iso(),
            board_before=board_before,
            board_after=env.board.copy_state(),
            board_size=env.board.size,
            win_length=env.board.win_length,
            rule_set=env.board.rule_set,
            enforce_center_opening=env.board.enforce_center_opening,
            current_player=player.name.lower(),
            current_player_value=int(player.value),
            selected_move=(row, col),
            action_index=int(prediction.action_index),
            policy_source=policy_source,
            policy_probs=[float(value) for value in prediction.policy.tolist()],
            value=float(prediction.value),
            legal=legal,
            used_tactical_move=prediction.used_tactical_move,
            winner=_winner_name(env.board.winner),
            winner_value=None if env.board.winner is None else int(env.board.winner.value),
            terminal=env.board.winner is not None or not legal,
            checkpoint=checkpoint,
            robot_action=robot_action,
            observation=observation,
            error=error,
        )
        records.append(record)
        append_episode_record(output_path, record)
        if not legal:
            break

    return records


def _render_images(
    env: GomokuMujocoEnv,
    episode_asset_dir: Path,
    step: int,
    timing: str,
    cameras: tuple[str, ...],
    width: int,
    height: int,
) -> dict[str, str]:
    images: dict[str, str] = {}
    for camera in cameras:
        image_path = episode_asset_dir / f"step_{step:03d}_{camera}_{timing}.png"
        Image.fromarray(env.render(width=width, height=height, camera=camera)).save(image_path)
        images[f"{camera}_{timing}"] = str(image_path)
    return images


def _build_observation(
    env: GomokuMujocoEnv,
    player: Player,
    target_cell: tuple[int, int],
    robot_action: dict[str, object],
    *,
    board_state: list[list[int]],
    images_before: dict[str, str],
    images_after: dict[str, str],
    episode_index: int,
    frame_index: int,
    is_first: bool,
    is_last: bool,
    is_terminal: bool,
) -> dict[str, object]:
    target_world = env.board_to_world(*target_cell)
    return {
        "language_instruction": (
            f"place the {player.name.lower()} stone at row {target_cell[0]} column {target_cell[1]}"
        ),
        "images": {**images_before, **images_after},
        "state": {
            "board_flat": [value for row in board_state for value in row],
            "current_player_value": int(player.value),
            "target_cell": [target_cell[0], target_cell[1]],
            "target_world_xyz": [target_world[0], target_world[1], target_world[2]],
            "robot_state": robot_action["action"][0] if robot_action.get("action") else None,
        },
        "episode_index": episode_index,
        "frame_index": frame_index,
        "timestamp_s": frame_index,
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _winner_name(winner: Player | None) -> str | None:
    if winner is None:
        return None
    return winner.name.lower()
