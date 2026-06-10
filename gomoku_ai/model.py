from __future__ import annotations

from typing import Protocol

import numpy as np

from board import GomokuBoard

from .encoding import legal_action_mask


class PolicyValueModel(Protocol):
    def predict(self, board: GomokuBoard) -> tuple[np.ndarray, float]:
        """Return action priors and value from the current player's perspective."""


class UniformPolicyValueModel:
    """Dependency-free baseline used to validate MCTS and self-play plumbing."""

    def predict(self, board: GomokuBoard) -> tuple[np.ndarray, float]:
        mask = legal_action_mask(board)
        policy = np.zeros(board.size * board.size, dtype=np.float32)
        if mask.any():
            policy[mask] = 1.0 / float(mask.sum())
        return policy, 0.0
