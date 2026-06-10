from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from board import GomokuBoard

from .encoding import encode_board, legal_action_mask


class GomokuPolicyValueNet(nn.Module):
    def __init__(self, board_size: int = 15, channels: int = 64) -> None:
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.trunk = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.action_size, self.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.action_size, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(states)
        policy_logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return policy_logits, values


class TorchPolicyValueModel:
    def __init__(self, network: GomokuPolicyValueNet, device: str | torch.device = "cpu") -> None:
        self.network = network.to(device)
        self.device = torch.device(device)

    def predict(self, board: GomokuBoard) -> tuple[np.ndarray, float]:
        if board.size != self.network.board_size:
            raise ValueError(f"model board_size={self.network.board_size}, got board size={board.size}")
        state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        self.network.eval()
        with torch.no_grad():
            logits, value = self.network(state)
            logits = logits.squeeze(0).detach().cpu().numpy().astype(np.float32)

        mask = legal_action_mask(board)
        logits[~mask] = -1.0e9
        policy = _softmax(logits)
        policy[~mask] = 0.0
        total = float(policy.sum())
        if total <= 0.0 and mask.any():
            policy[mask] = 1.0 / float(mask.sum())
        elif total > 0.0:
            policy /= total
        return policy.astype(np.float32), float(value.item())


def save_checkpoint(network: GomokuPolicyValueNet, path: str | Path) -> None:
    checkpoint = {
        "board_size": network.board_size,
        "state_dict": network.state_dict(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> GomokuPolicyValueNet:
    checkpoint = torch.load(path, map_location=device)
    network = GomokuPolicyValueNet(board_size=int(checkpoint["board_size"]))
    network.load_state_dict(checkpoint["state_dict"])
    return network.to(device)


def policy_value_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
) -> torch.Tensor:
    policy_loss = -(policy_targets * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(values, value_targets)
    return policy_loss + value_loss


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / exp.sum()
