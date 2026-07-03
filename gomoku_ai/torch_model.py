from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from board import GomokuBoard

from .encoding import encode_board, legal_action_mask


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.activation(states + self.layers(states))


class GomokuPolicyValueNet(nn.Module):
    def __init__(
        self,
        board_size: int = 15,
        channels: int = 256,
        res_blocks: int = 16,
        input_channels: int = 6,
        architecture: str = "resnet",
        rule_set: str = "free",
        enforce_center_opening: bool = False,
    ) -> None:
        super().__init__()
        if architecture not in {"resnet", "legacy_cnn"}:
            raise ValueError(f"unsupported architecture: {architecture}")
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.channels = channels
        self.res_blocks = res_blocks
        self.input_channels = input_channels
        self.architecture = architecture
        self.rule_set = rule_set
        self.enforce_center_opening = enforce_center_opening
        if architecture == "legacy_cnn":
            self.trunk = nn.Sequential(
                nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            )
        else:
            self.trunk = nn.Sequential(
                nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                *[ResidualBlock(channels) for _ in range(res_blocks)],
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
        policies, values = self.predict_batch([board])
        return policies[0], float(values[0])

    def predict_batch(self, boards: list[GomokuBoard]) -> tuple[np.ndarray, np.ndarray]:
        if not boards:
            return np.zeros((0, self.network.action_size), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        if any(board.size != self.network.board_size for board in boards):
            bad_size = next(board.size for board in boards if board.size != self.network.board_size)
            raise ValueError(f"model board_size={self.network.board_size}, got board size={bad_size}")
        states = np.stack(
            [encode_board(board, feature_planes=self.network.input_channels) for board in boards]
        ).astype(np.float32)
        state_tensor = torch.from_numpy(states).to(self.device)
        self.network.eval()
        with torch.no_grad():
            logits_tensor, value_tensor = self.network(state_tensor)
            logits_batch = logits_tensor.detach().cpu().numpy().astype(np.float32)
            values = value_tensor.detach().cpu().numpy().astype(np.float32)

        policies = np.zeros_like(logits_batch, dtype=np.float32)
        for index, board in enumerate(boards):
            mask = legal_action_mask(board)
            logits = logits_batch[index]
            logits[~mask] = -1.0e9
            policy = _softmax(logits)
            policy[~mask] = 0.0
            total = float(policy.sum())
            if total <= 0.0 and mask.any():
                policy[mask] = 1.0 / float(mask.sum())
            elif total > 0.0:
                policy /= total
            policies[index] = policy.astype(np.float32)
        return policies, values

def save_checkpoint(network: GomokuPolicyValueNet, path: str | Path, metadata: dict | None = None) -> None:
    checkpoint = {
        "board_size": network.board_size,
        "architecture": network.architecture,
        "channels": network.channels,
        "res_blocks": network.res_blocks,
        "input_channels": network.input_channels,
        "rule_set": network.rule_set,
        "enforce_center_opening": network.enforce_center_opening,
        "state_dict": network.state_dict(),
    }
    if metadata:
        checkpoint.update(metadata)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> GomokuPolicyValueNet:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["state_dict"]
    channels = int(checkpoint.get("channels", state_dict["trunk.0.weight"].shape[0]))
    input_channels = int(checkpoint.get("input_channels", state_dict["trunk.0.weight"].shape[1]))
    network = GomokuPolicyValueNet(
        board_size=int(checkpoint["board_size"]),
        channels=channels,
        res_blocks=int(checkpoint.get("res_blocks", 0)),
        input_channels=input_channels,
        architecture=str(checkpoint.get("architecture", "legacy_cnn")),
        rule_set=str(checkpoint.get("rule_set", checkpoint.get("training_config", {}).get("rule_set", "free"))),
        enforce_center_opening=bool(
            checkpoint.get(
                "enforce_center_opening",
                checkpoint.get("training_config", {}).get("enforce_center_opening", False),
            )
        ),
    )
    network.load_state_dict(state_dict)
    return network.to(device)


def policy_value_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
) -> torch.Tensor:
    policy_loss, value_loss = policy_value_loss_components(policy_logits, values, policy_targets, value_targets)
    return policy_loss + value_loss


def policy_value_loss_components(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    policy_loss = -(policy_targets * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(values, value_targets)
    return policy_loss, value_loss


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / exp.sum()
