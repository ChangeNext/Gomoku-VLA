from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from board import GomokuBoard, Player

from .encoding import action_to_index, clone_board, encode_board, index_to_action
from .mcts import MCTSConfig, run_mcts
from .model import PolicyValueModel


@dataclass(frozen=True)
class SelfPlayConfig:
    board_size: int = 15
    win_length: int = 5
    mcts: MCTSConfig = MCTSConfig()
    max_moves: int | None = None


@dataclass(frozen=True)
class SelfPlaySample:
    state: np.ndarray
    policy_target: np.ndarray
    value_target: float
    player: Player


def generate_self_play_game(model: PolicyValueModel, config: SelfPlayConfig | None = None) -> list[SelfPlaySample]:
    config = config or SelfPlayConfig()
    board = GomokuBoard(size=config.board_size, win_length=config.win_length)
    max_moves = config.max_moves or board.size * board.size
    pending: list[tuple[np.ndarray, np.ndarray, Player]] = []

    while board.winner is None and board.move_count < max_moves:
        policy = run_mcts(board, model, config.mcts)
        pending.append((encode_board(board), policy, board.current_player))
        action_index = int(np.random.choice(board.size * board.size, p=policy))
        board.place(*index_to_action(action_index, board.size))

    return [
        SelfPlaySample(
            state=state,
            policy_target=policy,
            value_target=_value_for_player(board.winner, player),
            player=player,
        )
        for state, policy, player in pending
    ]


def select_greedy_move(board: GomokuBoard, model: PolicyValueModel, config: MCTSConfig | None = None) -> tuple[int, int]:
    eval_config = config or MCTSConfig(temperature=0.0)
    policy = run_mcts(clone_board(board), model, eval_config)
    return index_to_action(int(np.argmax(policy)), board.size)


def _value_for_player(winner: Player | None, player: Player) -> float:
    if winner is None or winner == Player.EMPTY:
        return 0.0
    return 1.0 if winner == player else -1.0
