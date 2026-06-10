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


def encode_board(board: GomokuBoard) -> np.ndarray:
    """Encode board from the current player's perspective as [3, size, size]."""
    current = board.current_player
    opponent = current.opponent if board.winner is None else Player.WHITE
    grid = np.asarray(board.grid, dtype=np.int8)
    encoded = np.zeros((3, board.size, board.size), dtype=np.float32)
    encoded[0] = grid == current.value
    encoded[1] = grid == opponent.value
    encoded[2].fill(1.0 if current == Player.BLACK else 0.0)
    return encoded


def clone_board(board: GomokuBoard) -> GomokuBoard:
    copied = GomokuBoard(size=board.size, win_length=board.win_length)
    copied.grid = [row[:] for row in board.grid]
    copied.current_player = board.current_player
    copied.move_count = board.move_count
    copied.winner = board.winner
    return copied
