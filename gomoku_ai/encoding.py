from __future__ import annotations

import numpy as np

from board import GomokuBoard, Player


def action_to_index(row: int, col: int, board_size: int) -> int:
    return row * board_size + col


def index_to_action(index: int, board_size: int) -> tuple[int, int]:
    return divmod(index, board_size)


def legal_action_mask(board: GomokuBoard) -> np.ndarray:
    mask = np.zeros(board.size * board.size, dtype=bool)
    for row, col in board.legal_moves():
        mask[action_to_index(row, col, board.size)] = True
    return mask


def encode_board(board: GomokuBoard, feature_planes: int = 3) -> np.ndarray:
    """Encode board from the current player's perspective."""
    if feature_planes not in {3, 6}:
        raise ValueError(f"feature_planes must be 3 or 6, got {feature_planes}")
    current = board.current_player
    opponent = current.opponent if board.winner is None else Player.WHITE
    grid = np.asarray(board.grid, dtype=np.int8)
    encoded = np.zeros((feature_planes, board.size, board.size), dtype=np.float32)
    encoded[0] = grid == current.value
    encoded[1] = grid == opponent.value
    encoded[2].fill(1.0 if current == Player.BLACK else 0.0)
    if feature_planes == 6:
        legal = np.zeros((board.size, board.size), dtype=np.float32)
        for row, col in board.legal_moves():
            legal[row, col] = 1.0
        encoded[3] = legal
        encoded[4].fill(float(board.move_count) / float(board.size * board.size))
        encoded[5].fill(1.0)
    return encoded


def clone_board(board: GomokuBoard) -> GomokuBoard:
    copied = GomokuBoard(
        size=board.size,
        win_length=board.win_length,
        rule_set=board.rule_set,
        enforce_center_opening=board.enforce_center_opening,
    )
    copied.grid = [row[:] for row in board.grid]
    copied.current_player = board.current_player
    copied.move_count = board.move_count
    copied.winner = board.winner
    return copied
