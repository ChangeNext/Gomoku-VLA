from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .self_play import SelfPlaySample


@dataclass(frozen=True)
class TrainingBatch:
    states: np.ndarray
    policy_targets: np.ndarray
    value_targets: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self._samples: deque[SelfPlaySample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._samples)

    def add_game(self, samples: Iterable[SelfPlaySample]) -> None:
        self._samples.extend(samples)

    def sample(self, batch_size: int, augment: bool = True) -> TrainingBatch:
        if not self._samples:
            raise ValueError("cannot sample from an empty replay buffer")
        batch = random.sample(list(self._samples), min(batch_size, len(self._samples)))
        states: list[np.ndarray] = []
        policy_targets: list[np.ndarray] = []
        for sample in batch:
            state = sample.state
            policy = sample.policy_target
            if augment:
                state, policy = augment_state_policy(state, policy)
            states.append(state)
            policy_targets.append(policy)
        return TrainingBatch(
            states=np.stack(states).astype(np.float32),
            policy_targets=np.stack(policy_targets).astype(np.float32),
            value_targets=np.asarray([sample.value_target for sample in batch], dtype=np.float32),
        )


def augment_state_policy(state: np.ndarray, policy_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply one random board symmetry to both state planes and policy targets."""
    if state.ndim != 3 or state.shape[1] != state.shape[2]:
        raise ValueError(f"state must have shape [channels, size, size], got {state.shape}")
    board_size = state.shape[1]
    if policy_target.shape != (board_size * board_size,):
        raise ValueError(f"policy_target shape must be {(board_size * board_size,)}, got {policy_target.shape}")

    rotations = random.randrange(4)
    flip = random.choice((False, True))
    augmented_state = np.rot90(state, k=rotations, axes=(1, 2))
    policy_board = np.rot90(policy_target.reshape(board_size, board_size), k=rotations)
    if flip:
        augmented_state = np.flip(augmented_state, axis=2)
        policy_board = np.fliplr(policy_board)
    return augmented_state.copy(), policy_board.reshape(-1).copy()
