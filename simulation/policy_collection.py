from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from PIL import Image

from board import Player
from gomoku_ai.episode_recorder import EpisodeStepRecord, append_episode_record
from gomoku_ai.inference import MovePrediction
from robot_control import RobotSafetyController, SafetyReport

from .mujoco_gomoku_env import GomokuMujocoEnv
from .scripted_robot import build_pick_place_action

MIN_TRAINING_IMAGE_SIZE = 224


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
    safety = RobotSafetyController(env)

    while env.board.winner is None and env.board.move_count < max_steps:
        step = len(records)
        board_before = env.board.copy_state()
        player = env.board.current_player
        supply_before = _supply_counts(env)
        held_before = _held_stone_name(env)
        prediction = predictor.predict(env.board)
        row, col = prediction.move
        legal = env.board.is_legal_move(row, col)
        env.set_selection(row, col, update_robot_target=False)
        images_before = _render_images(env, episode_asset_dir, step, "before", cameras, image_width, image_height)
        robot_action = build_pick_place_action(env, (row, col), player)
        pick_report = safety.validate_pick(player)
        place_report = safety.validate_place_cell(row, col, player)
        trace_report = safety.validate_action_trace(robot_action)
        safe_to_execute = pick_report.ok and place_report.ok and trace_report.ok
        error = None
        if legal and safe_to_execute:
            pick_x, pick_y, pick_z = robot_action["pick_pose_xyz"]
            env.grasp_supply_stone(player, float(pick_x), float(pick_y), float(pick_z))
            env.commit_held_stone_to_cell(row, col, update_robot_target=True)
        else:
            error = _first_error(row, col, legal, pick_report, place_report, trace_report)
        images_after = _render_images(env, episode_asset_dir, step, "after", cameras, image_width, image_height)
        supply_after = _supply_counts(env)
        held_after = _held_stone_name(env)
        robot_action["safety"] = {
            "ok": safe_to_execute,
            "pick": _report_dict(pick_report),
            "place": _report_dict(place_report),
            "trace": _report_dict(trace_report),
        }
        robot_action["supply_before"] = supply_before
        robot_action["supply_after"] = supply_after
        robot_action["held_stone_before"] = held_before
        robot_action["held_stone_after"] = held_after
        robot_action["robot_model"] = env.robot_model
        robot_action["attachment_mode"] = "scripted_held_stone"
        robot_action["execution_success"] = bool(
            legal
            and safe_to_execute
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
            board_after=env.board.copy_state(),
            supply_before=supply_before,
            supply_after=supply_after,
            held_before=held_before,
            held_after=held_after,
            images_before=images_before,
            images_after=images_after,
            image_width=image_width,
            image_height=image_height,
            cameras=cameras,
            episode_index=episode_index,
            frame_index=step,
            is_first=step == 0,
            is_last=will_stop,
            is_terminal=env.board.winner is not None or not legal or not safe_to_execute,
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
            terminal=env.board.winner is not None or not legal or not safe_to_execute,
            checkpoint=checkpoint,
            robot_action=robot_action,
            observation=observation,
            error=error,
        )
        records.append(record)
        append_episode_record(output_path, record)
        if not legal or not safe_to_execute:
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
    board_after: list[list[int]],
    supply_before: dict[str, int],
    supply_after: dict[str, int],
    held_before: str | None,
    held_after: str | None,
    images_before: dict[str, str],
    images_after: dict[str, str],
    image_width: int,
    image_height: int,
    cameras: tuple[str, ...],
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
        "image_metadata": {
            "width": image_width,
            "height": image_height,
            "cameras": list(cameras),
            "training_usable": image_width >= MIN_TRAINING_IMAGE_SIZE and image_height >= MIN_TRAINING_IMAGE_SIZE,
            "minimum_training_size": MIN_TRAINING_IMAGE_SIZE,
        },
        "state": {
            "board_flat": [value for row in board_state for value in row],
            "board_after_flat": [value for row in board_after for value in row],
            "current_player_value": int(player.value),
            "target_cell": [target_cell[0], target_cell[1]],
            "target_world_xyz": [target_world[0], target_world[1], target_world[2]],
            "robot_state": robot_action["action"][0] if robot_action.get("action") else None,
            "robot_model": env.robot_model,
            "supply_before": supply_before,
            "supply_after": supply_after,
            "held_stone_before": held_before,
            "held_stone_after": held_after,
            "safety": robot_action.get("safety"),
        },
        "action": {
            "format": robot_action["action_names"],
            "sequence": robot_action["action"],
            "controller_type": robot_action["controller_type"],
            "attachment_mode": robot_action.get("attachment_mode"),
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


def _supply_counts(env: GomokuMujocoEnv) -> dict[str, int]:
    return {
        "black": int(env.supply_counts[Player.BLACK]),
        "white": int(env.supply_counts[Player.WHITE]),
    }


def _held_stone_name(env: GomokuMujocoEnv) -> str | None:
    return None if env.held_stone_player is None else env.held_stone_player.name.lower()


def _report_dict(report: SafetyReport) -> dict[str, object]:
    return {"ok": report.ok, "reason": report.reason}


def _first_error(
    row: int,
    col: int,
    legal: bool,
    pick_report: SafetyReport,
    place_report: SafetyReport,
    trace_report: SafetyReport,
) -> str:
    if not legal:
        return f"illegal move: row={row}, col={col}"
    for report in (pick_report, place_report, trace_report):
        if not report.ok:
            return report.reason or "unsafe robot action"
    return "unknown collection error"
