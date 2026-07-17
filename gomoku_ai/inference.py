from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
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
        temperature: float = 0.0,
        temperature_moves: int = 0,
        late_temperature: float | None = None,
        sample_moves: bool = False,
        add_root_noise: bool = False,
        root_dirichlet_alpha: float = 0.3,
        root_exploration_fraction: float = 0.25,
        seed: int | None = None,
        use_tactics: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")

        self.device = resolve_device(device)
        self.network = load_checkpoint(self.checkpoint_path, device=self.device)
        self.model = TorchPolicyValueModel(self.network, device=self.device)
        self.mcts_config = MCTSConfig(
            simulations=simulations,
            c_puct=c_puct,
            temperature=temperature,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_exploration_fraction=root_exploration_fraction,
            add_root_noise=add_root_noise,
        )
        self.temperature_moves = temperature_moves
        self.late_temperature = temperature if late_temperature is None else late_temperature
        self.sample_moves = sample_moves
        self.rng = np.random.default_rng(seed)
        self.use_tactics = use_tactics

    @property
    def board_size(self) -> int:
        return self.network.board_size

    @property
    def win_length(self) -> int:
        return self.network.win_length

    @property
    def rule_set(self) -> str:
        return self.network.rule_set

    @property
    def enforce_center_opening(self) -> bool:
        return self.network.enforce_center_opening

    def new_board(
        self,
        *,
        win_length: int | None = None,
        rule_set: str | None = None,
        enforce_center_opening: bool | None = None,
    ) -> GomokuBoard:
        return GomokuBoard(
            size=self.board_size,
            win_length=self.win_length if win_length is None else win_length,
            rule_set=rule_set or self.rule_set,
            enforce_center_opening=(
                self.enforce_center_opening if enforce_center_opening is None else enforce_center_opening
            ),
        )

    def predict(self, board: GomokuBoard) -> MovePrediction:
        mcts_config = self.mcts_config
        if self.temperature_moves > 0 and board.move_count >= self.temperature_moves:
            mcts_config = replace(mcts_config, temperature=self.late_temperature)
        return predict_move(
            board,
            self.model,
            mcts_config,
            use_tactics=self.use_tactics,
            sample_move=self.sample_moves,
            rng=self.rng,
        )


def predict_move(
    board: GomokuBoard,
    model: PolicyValueModel,
    config: MCTSConfig | None = None,
    *,
    use_tactics: bool = True,
    sample_move: bool = False,
    rng: np.random.Generator | None = None,
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
    action_index = _select_action_index(policy, sample_move=sample_move, rng=rng)
    row, col = index_to_action(action_index, board.size)
    return MovePrediction(
        row=row,
        col=col,
        action_index=action_index,
        policy=policy,
        value=value,
        used_tactical_move=False,
    )


def _select_action_index(
    policy: np.ndarray,
    *,
    sample_move: bool,
    rng: np.random.Generator | None,
) -> int:
    if not sample_move:
        return int(np.argmax(policy))
    probabilities = np.asarray(policy, dtype=np.float64)
    if probabilities.ndim != 1 or not np.all(np.isfinite(probabilities)):
        raise ValueError("cannot sample a move from a non-finite policy")
    if np.any(probabilities < 0.0):
        raise ValueError("cannot sample a move from a policy with negative probabilities")
    total = float(probabilities.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("cannot sample a move from an empty policy")
    probabilities /= total
    # NumPy's Generator.choice checks the sum more strictly than float32 MCTS
    # policies can guarantee. Correct the final floating-point residual without
    # changing the relative policy distribution in any meaningful way.
    residual = 1.0 - float(probabilities.sum(dtype=np.float64))
    probabilities[int(np.argmax(probabilities))] += residual
    generator = rng or np.random.default_rng()
    return int(generator.choice(len(probabilities), p=probabilities))
