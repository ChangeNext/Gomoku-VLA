from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from board import GomokuBoard

from .encoding import clone_board, index_to_action
from .mcts import MCTSConfig, run_mcts
from .model import PolicyValueModel
from .tactics import select_tactical_move
from .torch_model import TorchPolicyValueModel, load_checkpoint


@dataclass(frozen=True)
class MovePrediction:
    row: int
    col: int
    action_index: int
    policy: np.ndarray
    value: float
    used_tactical_move: bool = False

    @property
    def move(self) -> tuple[int, int]:
        return self.row, self.col


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class CheckpointPolicy:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "auto",
        simulations: int = 64,
        c_puct: float = 1.5,
        use_tactics: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")

        self.device = resolve_device(device)
        self.network = load_checkpoint(self.checkpoint_path, device=self.device)
        self.model = TorchPolicyValueModel(self.network, device=self.device)
        self.mcts_config = MCTSConfig(simulations=simulations, c_puct=c_puct, temperature=0.0)
        self.use_tactics = use_tactics

    @property
    def board_size(self) -> int:
        return self.network.board_size

    @property
    def rule_set(self) -> str:
        return self.network.rule_set

    @property
    def enforce_center_opening(self) -> bool:
        return self.network.enforce_center_opening

    def new_board(
        self,
        *,
        win_length: int = 5,
        rule_set: str | None = None,
        enforce_center_opening: bool | None = None,
    ) -> GomokuBoard:
        return GomokuBoard(
            size=self.board_size,
            win_length=win_length,
            rule_set=rule_set or self.rule_set,
            enforce_center_opening=(
                self.enforce_center_opening if enforce_center_opening is None else enforce_center_opening
            ),
        )

    def predict(self, board: GomokuBoard) -> MovePrediction:
        return predict_move(
            board,
            self.model,
            self.mcts_config,
            use_tactics=self.use_tactics,
        )


def predict_move(
    board: GomokuBoard,
    model: PolicyValueModel,
    config: MCTSConfig | None = None,
    *,
    use_tactics: bool = True,
) -> MovePrediction:
    if board.winner is not None:
        raise ValueError("cannot predict a move for a finished game")
    if not board.legal_moves():
        raise ValueError("cannot predict a move when no legal moves are available")

    _, value = model.predict(board)
    if use_tactics:
        tactical_move = select_tactical_move(board)
        if tactical_move is not None:
            row, col = tactical_move
            action_index = row * board.size + col
            policy = np.zeros(board.size * board.size, dtype=np.float32)
            policy[action_index] = 1.0
            return MovePrediction(
                row=row,
                col=col,
                action_index=action_index,
                policy=policy,
                value=value,
                used_tactical_move=True,
            )

    eval_config = config or MCTSConfig(temperature=0.0)
    policy = run_mcts(clone_board(board), model, eval_config)
    action_index = int(np.argmax(policy))
    row, col = index_to_action(action_index, board.size)
    return MovePrediction(
        row=row,
        col=col,
        action_index=action_index,
        policy=policy,
        value=value,
        used_tactical_move=False,
    )
