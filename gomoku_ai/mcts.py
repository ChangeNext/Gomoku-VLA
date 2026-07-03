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
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    add_root_noise: bool = False


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
    return run_mcts_batch([board], model, config)[0]


def run_mcts_batch(
    boards: list[GomokuBoard],
    model: PolicyValueModel,
    config: MCTSConfig | None = None,
) -> list[np.ndarray]:
    config = config or MCTSConfig()
    if not boards:
        return []
    board_size = boards[0].size
    if any(board.size != board_size for board in boards):
        raise ValueError("all boards in a batch must have the same size")

    roots = [Node(prior=1.0) for _ in boards]
    root_boards = [clone_board(board) for board in boards]
    _expand_batch(roots, root_boards, model)
    if config.add_root_noise:
        for root in roots:
            _add_root_dirichlet_noise(root, config)

    for _ in range(config.simulations):
        searches = [
            _descend_to_leaf(root, clone_board(board), config)
            for root, board in zip(roots, boards)
        ]
        leaves = [search.leaf for search in searches if search.leaf is not None]
        leaf_boards = [search.board for search in searches if search.leaf is not None]
        if leaves:
            values = _expand_batch(leaves, leaf_boards, model)
            value_iter = iter(values)
        else:
            value_iter = iter(())
        for search in searches:
            leaf_value = search.terminal_value if search.leaf is None else next(value_iter)
            _backup(search.path, leaf_value)

    policies: list[np.ndarray] = []
    for root, board in zip(roots, boards):
        visits = np.zeros(board.size * board.size, dtype=np.float32)
        for action_index, child in root.children.items():
            visits[action_index] = child.visit_count
        policies.append(_visits_to_policy(visits, legal_action_mask(board), config.temperature))
    return policies


def _add_root_dirichlet_noise(root: Node, config: MCTSConfig) -> None:
    if not root.children:
        return
    actions = list(root.children)
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    for action_index, noise_value in zip(actions, noise):
        child = root.children[action_index]
        child.prior = (1.0 - config.root_exploration_fraction) * child.prior + config.root_exploration_fraction * float(noise_value)


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


@dataclass
class SearchPath:
    path: list[Node]
    board: GomokuBoard
    leaf: Node | None
    terminal_value: float = 0.0


def _descend_to_leaf(root: Node, board: GomokuBoard, config: MCTSConfig) -> SearchPath:
    node = root
    path = [node]
    while True:
        if board.winner is not None:
            return SearchPath(path=path, board=board, leaf=None, terminal_value=_terminal_value(board))
        if not node.children:
            return SearchPath(path=path, board=board, leaf=node)
        action_index, child = _select_child(node, config)
        board.place(*index_to_action(action_index, board.size))
        node = child
        path.append(node)


def _backup(path: list[Node], leaf_value: float) -> None:
    value = leaf_value
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def _expand(node: Node, board: GomokuBoard, model: PolicyValueModel) -> float:
    return float(_expand_batch([node], [board], model)[0])


def _expand_batch(nodes: list[Node], boards: list[GomokuBoard], model: PolicyValueModel) -> np.ndarray:
    if len(nodes) != len(boards):
        raise ValueError("nodes and boards must have the same length")
    if not nodes:
        return np.zeros((0,), dtype=np.float32)
    policies, values = model.predict_batch(boards)
    if policies.shape != (len(boards), boards[0].size * boards[0].size):
        raise ValueError(f"policy batch shape must be {(len(boards), boards[0].size * boards[0].size)}, got {policies.shape}")
    if values.shape != (len(boards),):
        raise ValueError(f"value batch shape must be {(len(boards),)}, got {values.shape}")
    for node, board, policy, value in zip(nodes, boards, policies, values):
        _expand_with_policy(node, board, policy, float(value))
    return np.clip(values.astype(np.float32), -1.0, 1.0)


def _expand_with_policy(node: Node, board: GomokuBoard, policy: np.ndarray, value: float) -> float:
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
