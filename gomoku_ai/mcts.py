from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np

from board import GomokuBoard, Player

from .encoding import action_to_index, clone_board, index_to_action, legal_action_mask
from .model import PolicyValueModel


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 64
    c_puct: float = 1.5
    temperature: float = 1.0


@dataclass
class Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def run_mcts(board: GomokuBoard, model: PolicyValueModel, config: MCTSConfig | None = None) -> np.ndarray:
    config = config or MCTSConfig()
    root = Node(prior=1.0)
    root_board = clone_board(board)
    _expand(root, root_board, model)

    for _ in range(config.simulations):
        search_board = clone_board(board)
        _search(root, search_board, model, config)

    visits = np.zeros(board.size * board.size, dtype=np.float32)
    for action_index, child in root.children.items():
        visits[action_index] = child.visit_count
    return _visits_to_policy(visits, legal_action_mask(board), config.temperature)


def _search(node: Node, board: GomokuBoard, model: PolicyValueModel, config: MCTSConfig) -> float:
    if board.winner is not None:
        value = _terminal_value(board)
        node.visit_count += 1
        node.value_sum += value
        return value

    if not node.children:
        value = _expand(node, board, model)
        node.visit_count += 1
        node.value_sum += value
        return value

    action_index, child = _select_child(node, config)
    board.place(*index_to_action(action_index, board.size))
    child_value = _search(child, board, model, config)
    value = -child_value
    node.visit_count += 1
    node.value_sum += value
    return value


def _expand(node: Node, board: GomokuBoard, model: PolicyValueModel) -> float:
    policy, value = model.predict(board)
    policy = np.asarray(policy, dtype=np.float32).copy()
    if policy.shape != (board.size * board.size,):
        raise ValueError(f"policy shape must be {(board.size * board.size,)}, got {policy.shape}")

    mask = legal_action_mask(board)
    policy[~mask] = 0.0
    total = float(policy.sum())
    if total <= 0.0 and mask.any():
        policy[mask] = 1.0 / float(mask.sum())
    elif total > 0.0:
        policy /= total

    for action_index in np.flatnonzero(mask):
        node.children[int(action_index)] = Node(prior=float(policy[action_index]))
    return float(np.clip(value, -1.0, 1.0))


def _select_child(node: Node, config: MCTSConfig) -> tuple[int, Node]:
    total_visits = max(1, node.visit_count)

    def score(item: tuple[int, Node]) -> float:
        _, child = item
        q_value = -child.value
        exploration = config.c_puct * child.prior * sqrt(total_visits) / (1 + child.visit_count)
        return q_value + exploration

    return max(node.children.items(), key=score)


def _terminal_value(board: GomokuBoard) -> float:
    if board.winner == Player.EMPTY:
        return 0.0
    return -1.0


def _visits_to_policy(visits: np.ndarray, mask: np.ndarray, temperature: float) -> np.ndarray:
    policy = np.zeros_like(visits, dtype=np.float32)
    if not mask.any():
        return policy
    if visits.sum() <= 0:
        policy[mask] = 1.0 / float(mask.sum())
        return policy
    if temperature <= 1e-6:
        legal_visits = visits.copy()
        legal_visits[~mask] = -1.0
        policy[int(np.argmax(legal_visits))] = 1.0
        return policy
    scaled = np.power(visits, 1.0 / temperature)
    scaled[~mask] = 0.0
    policy = scaled / scaled.sum()
    return policy.astype(np.float32)
