from __future__ import annotations

from typing import Protocol

import numpy as np

from board import GomokuBoard

from .encoding import legal_action_mask


class PolicyValueModel(Protocol):
    def predict(self, board: GomokuBoard) -> tuple[np.ndarray, float]:
        """Return action priors and value from the current player's perspective."""

    def predict_batch(self, boards: list[GomokuBoard]) -> tuple[np.ndarray, np.ndarray]:
        """Return action priors and values for boards with the same board size."""


class UniformPolicyValueModel:
    """Dependency-free baseline used to validate MCTS and self-play plumbing."""

    def predict(self, board: GomokuBoard) -> tuple[np.ndarray, float]:
        policies, values = self.predict_batch([board])
        return policies[0], float(values[0])

    def predict_batch(self, boards: list[GomokuBoard]) -> tuple[np.ndarray, np.ndarray]:
        if not boards:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        board_size = boards[0].size
        if any(board.size != board_size for board in boards):
            raise ValueError("all boards in a batch must have the same size")
        policies = np.zeros((len(boards), board_size * board_size), dtype=np.float32)
        for index, board in enumerate(boards):
            mask = legal_action_mask(board)
            if mask.any():
                policies[index, mask] = 1.0 / float(mask.sum())
        return policies, np.zeros((len(boards),), dtype=np.float32)
