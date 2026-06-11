from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .mcts import MCTSConfig
from .replay_buffer import ReplayBuffer
from .self_play import SelfPlayConfig, generate_self_play_game
from .torch_model import (
    GomokuPolicyValueNet,
    TorchPolicyValueModel,
    load_checkpoint,
    policy_value_loss,
    save_checkpoint,
)


@dataclass(frozen=True)
class AlphaZeroTrainingConfig:
    board_size: int = 9
    win_length: int = 5
    iterations: int = 1
    games_per_iteration: int = 2
    mcts_simulations: int = 16
    epochs: int = 1
    batches_per_epoch: int = 1
    batch_size: int = 32
    replay_capacity: int = 20_000
    learning_rate: float = 1e-3
    checkpoint_path: str = "checkpoints/alphazero_latest.pt"
    initial_checkpoint_path: str | None = None
    device: str = "cpu"
    augment_batches: bool = True


def run_training(config: AlphaZeroTrainingConfig) -> list[dict[str, float]]:
    device = torch.device(config.device)
    if config.initial_checkpoint_path:
        network = load_checkpoint(config.initial_checkpoint_path, device=device)
        if network.board_size != config.board_size:
            raise ValueError(
                f"initial checkpoint board_size={network.board_size}, got config board_size={config.board_size}"
            )
    else:
        network = GomokuPolicyValueNet(board_size=config.board_size).to(device)
    model = TorchPolicyValueModel(network, device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    replay = ReplayBuffer(capacity=config.replay_capacity)
    history: list[dict[str, float]] = []

    for iteration in range(1, config.iterations + 1):
        samples_added = 0
        for _ in range(config.games_per_iteration):
            samples = generate_self_play_game(
                model,
                SelfPlayConfig(
                    board_size=config.board_size,
                    win_length=config.win_length,
                    mcts=MCTSConfig(simulations=config.mcts_simulations),
                ),
            )
            replay.add_game(samples)
            samples_added += len(samples)

        mean_loss = train_epochs(
            network,
            optimizer,
            replay,
            config.epochs,
            config.batches_per_epoch,
            config.batch_size,
            device,
            augment=config.augment_batches,
        )
        save_checkpoint(network, Path(config.checkpoint_path))
        history.append(
            {
                "iteration": float(iteration),
                "samples_added": float(samples_added),
                "replay_size": float(len(replay)),
                "train_steps": float(config.epochs * config.batches_per_epoch),
                "loss": mean_loss,
            }
        )

    return history


def train_epochs(
    network: GomokuPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    device: torch.device,
    augment: bool = True,
) -> float:
    if len(replay) == 0:
        raise ValueError("replay buffer is empty")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batches_per_epoch <= 0:
        raise ValueError("batches_per_epoch must be positive")
    losses: list[float] = []
    network.train()
    for _ in range(epochs):
        for _ in range(batches_per_epoch):
            batch = replay.sample(batch_size, augment=augment)
            states = torch.from_numpy(batch.states).to(device)
            policy_targets = torch.from_numpy(batch.policy_targets).to(device)
            value_targets = torch.from_numpy(batch.value_targets).to(device)
            optimizer.zero_grad()
            policy_logits, values = network(states)
            loss = policy_value_loss(policy_logits, values, policy_targets, value_targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    return sum(losses) / len(losses)
