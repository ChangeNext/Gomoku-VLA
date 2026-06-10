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

    def sample(self, batch_size: int) -> TrainingBatch:
        if not self._samples:
            raise ValueError("cannot sample from an empty replay buffer")
        batch = random.sample(list(self._samples), min(batch_size, len(self._samples)))
        return TrainingBatch(
            states=np.stack([sample.state for sample in batch]).astype(np.float32),
            policy_targets=np.stack([sample.policy_target for sample in batch]).astype(np.float32),
            value_targets=np.asarray([sample.value_target for sample in batch], dtype=np.float32),
        )
