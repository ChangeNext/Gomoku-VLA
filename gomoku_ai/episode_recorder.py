from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from board import GomokuBoard, Player

from .inference import MovePrediction


class MovePredictor(Protocol):
    def predict(self, board: GomokuBoard) -> MovePrediction:
        ...


@dataclass(frozen=True)
class EpisodeStepRecord:
    game_id: str
    step: int
    timestamp: str
    board_before: list[list[int]]
    board_after: list[list[int]]
    board_size: int
    win_length: int
    rule_set: str
    enforce_center_opening: bool
    current_player: str
    current_player_value: int
    selected_move: tuple[int, int]
    action_index: int
    policy_source: str
    policy_probs: list[float]
    value: float
    legal: bool
    used_tactical_move: bool
    winner: str | None
    winner_value: int | None
    terminal: bool
    checkpoint: str | None = None
    robot_action: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    error: str | None = None


def play_and_record_episode(
    board: GomokuBoard,
    predictor: MovePredictor,
    output_jsonl: str | Path,
    *,
    game_id: str | None = None,
    policy_source: str = "alphazero",
    checkpoint: str | None = None,
    max_moves: int | None = None,
) -> list[EpisodeStepRecord]:
    if board.winner is not None:
        raise ValueError("cannot record an episode from a finished board")
    max_steps = max_moves or board.size * board.size
    if max_steps <= 0:
        raise ValueError("max_moves must be positive")

    path = Path(output_jsonl)
    path.parent.mkdir(parents=True, exist_ok=True)
    episode_id = game_id or str(uuid.uuid4())
    records: list[EpisodeStepRecord] = []

    while board.winner is None and board.move_count < max_steps:
        before = board.copy_state()
        player = board.current_player
        prediction = predictor.predict(board)
        row, col = prediction.move
        legal = board.is_legal_move(row, col)
        error = None
        if legal:
            board.place(row, col)
        else:
            error = f"illegal move: row={row}, col={col}"

        record = EpisodeStepRecord(
            game_id=episode_id,
            step=len(records),
            timestamp=_now_iso(),
            board_before=before,
            board_after=board.copy_state(),
            board_size=board.size,
            win_length=board.win_length,
            rule_set=board.rule_set,
            enforce_center_opening=board.enforce_center_opening,
            current_player=player.name.lower(),
            current_player_value=int(player.value),
            selected_move=(row, col),
            action_index=int(prediction.action_index),
            policy_source=policy_source,
            policy_probs=[float(value) for value in prediction.policy.tolist()],
            value=float(prediction.value),
            legal=legal,
            used_tactical_move=prediction.used_tactical_move,
            winner=_winner_name(board.winner),
            winner_value=None if board.winner is None else int(board.winner.value),
            terminal=board.winner is not None or not legal,
            checkpoint=checkpoint,
            robot_action=None,
            observation=None,
            error=error,
        )
        records.append(record)
        _append_jsonl(path, record)
        if not legal:
            break

    return records


def default_episode_output_path(checkpoint: str | Path) -> Path:
    checkpoint_path = Path(checkpoint)
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    return run_dir / "data" / f"{checkpoint_path.stem}_policy_episodes.jsonl"


def append_episode_record(path: str | Path, record: EpisodeStepRecord) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _append_jsonl(output_path, record)


def _append_jsonl(path: Path, record: EpisodeStepRecord) -> None:
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _winner_name(winner: Player | None) -> str | None:
    if winner is None:
        return None
    return winner.name.lower()
