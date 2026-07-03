from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from board import GomokuBoard, Player

from .encoding import action_to_index, clone_board, encode_board, index_to_action
from .mcts import MCTSConfig, run_mcts, run_mcts_batch
from .model import PolicyValueModel
from .tactics import select_tactical_move


@dataclass(frozen=True)
class SelfPlayConfig:
    board_size: int = 15
    win_length: int = 5
    rule_set: str = "free"
    enforce_center_opening: bool = False
    mcts: MCTSConfig = MCTSConfig()
    max_moves: int | None = None
    input_channels: int = 3
    temperature_moves: int = 12
    late_temperature: float = 0.1


@dataclass(frozen=True)
class SelfPlaySample:
    state: np.ndarray
    policy_target: np.ndarray
    value_target: float
    player: Player


def generate_self_play_game(model: PolicyValueModel, config: SelfPlayConfig | None = None) -> list[SelfPlaySample]:
    return generate_self_play_games(model, game_count=1, config=config)[0]


def generate_self_play_games(
    model: PolicyValueModel,
    game_count: int,
    config: SelfPlayConfig | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> list[list[SelfPlaySample]]:
    config = config or SelfPlayConfig()
    if game_count <= 0:
        raise ValueError("game_count must be positive")
    boards = [
        GomokuBoard(
            size=config.board_size,
            win_length=config.win_length,
            rule_set=config.rule_set,
            enforce_center_opening=config.enforce_center_opening,
        )
        for _ in range(game_count)
    ]
    max_moves = config.max_moves or config.board_size * config.board_size
    pending: list[list[tuple[np.ndarray, np.ndarray, Player]]] = [[] for _ in boards]

    while any(board.winner is None and board.move_count < max_moves for board in boards):
        active_indexes = [
            index
            for index, board in enumerate(boards)
            if board.winner is None and board.move_count < max_moves
        ]
        early_indexes = [
            index
            for index in active_indexes
            if boards[index].move_count < config.temperature_moves
        ]
        late_indexes = [index for index in active_indexes if index not in early_indexes]
        indexed_policies: dict[int, np.ndarray] = {}
        for indexes, temperature in (
            (early_indexes, config.mcts.temperature),
            (late_indexes, config.late_temperature),
        ):
            if not indexes:
                continue
            batch_policies = run_mcts_batch(
                [boards[index] for index in indexes],
                model,
                replace(config.mcts, temperature=temperature),
            )
            indexed_policies.update(zip(indexes, batch_policies))

        for index in active_indexes:
            board = boards[index]
            policy = indexed_policies[index]
            pending[index].append((encode_board(board, feature_planes=config.input_channels), policy, board.current_player))
            action_index = int(np.random.choice(board.size * board.size, p=policy))
            board.place(*index_to_action(action_index, board.size))
        if progress_callback is not None:
            completed_games = sum(
                board.winner is not None or board.move_count >= max_moves
                for board in boards
            )
            played_moves = sum(board.move_count for board in boards)
            progress_callback(played_moves, completed_games, len(active_indexes))

    return [
        [
            SelfPlaySample(
                state=state,
                policy_target=policy,
                value_target=_value_for_player(board.winner, player),
                player=player,
            )
            for state, policy, player in game_pending
        ]
        for board, game_pending in zip(boards, pending)
    ]


def select_greedy_move(board: GomokuBoard, model: PolicyValueModel, config: MCTSConfig | None = None) -> tuple[int, int]:
    tactical_move = select_tactical_move(board)
    if tactical_move is not None:
        return tactical_move
    eval_config = config or MCTSConfig(temperature=0.0)
    policy = run_mcts(clone_board(board), model, eval_config)
    return index_to_action(int(np.argmax(policy)), board.size)


def _value_for_player(winner: Player | None, player: Player) -> float:
    if winner is None or winner == Player.EMPTY:
        return 0.0
    return 1.0 if winner == player else -1.0
