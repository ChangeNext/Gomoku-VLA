from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

from board import GomokuBoard, Player

from .mcts import MCTSConfig
from .replay_buffer import ReplayBuffer
from .self_play import SelfPlayConfig, generate_self_play_game, select_greedy_move
from .torch_model import (
    GomokuPolicyValueNet,
    TorchPolicyValueModel,
    load_checkpoint,
    policy_value_loss_components,
    save_checkpoint,
)


@dataclass(frozen=True)
class AlphaZeroTrainingConfig:
    board_size: int = 15
    win_length: int = 5
    rule_set: str = "renju"
    enforce_center_opening: bool = True
    iterations: int = 1
    games_per_iteration: int = 2
    mcts_simulations: int = 64
    epochs: int = 1
    batches_per_epoch: int = 1
    batch_size: int = 32
    replay_capacity: int = 500_000
    learning_rate: float = 3e-4
    checkpoint_path: str | None = None
    initial_checkpoint_path: str | None = None
    history_csv_path: str | None = None
    plot_path: str | None = None
    runs_dir: str = "gomoku_ai/runs"
    run_name: str | None = None
    resume_run: str | None = None
    architecture: str = "resnet"
    channels: int = 256
    res_blocks: int = 16
    input_channels: int = 6
    temperature_moves: int = 16
    late_temperature: float = 0.1
    root_dirichlet_alpha: float = 0.03
    root_exploration_fraction: float = 0.25
    add_root_noise: bool = True
    device: str = "cpu"
    augment_batches: bool = True
    gradient_clip_norm: float = 5.0
    use_amp: bool = False
    evaluation_games: int = 0
    evaluation_simulations: int = 64
    promotion_threshold: float = 0.55
    evaluation_seed: int = 0


@dataclass(frozen=True)
class TrainingRunPaths:
    run_dir: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    history_csv_path: Path
    plot_path: Path
    policy_heatmap_path: Path
    replay_path: Path
    config_path: Path


def run_training(config: AlphaZeroTrainingConfig) -> list[dict[str, float]]:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device was requested, but torch.cuda.is_available() is false")
    run_paths = resolve_training_run_paths(config)
    write_training_config(config, run_paths)

    initial_checkpoint_path = config.initial_checkpoint_path
    if config.resume_run and initial_checkpoint_path is None and run_paths.checkpoint_path.exists():
        initial_checkpoint_path = str(run_paths.checkpoint_path)

    if initial_checkpoint_path:
        network = load_checkpoint(initial_checkpoint_path, device=device)
        if network.board_size != config.board_size:
            raise ValueError(
                f"initial checkpoint board_size={network.board_size}, got config board_size={config.board_size}"
            )
        if network.rule_set != config.rule_set:
            raise ValueError(f"initial checkpoint rule_set={network.rule_set}, got config rule_set={config.rule_set}")
        if network.enforce_center_opening != config.enforce_center_opening:
            raise ValueError(
                "initial checkpoint enforce_center_opening="
                f"{network.enforce_center_opening}, got config enforce_center_opening={config.enforce_center_opening}"
            )
    else:
        network = GomokuPolicyValueNet(
            board_size=config.board_size,
            channels=config.channels,
            res_blocks=config.res_blocks,
            input_channels=config.input_channels,
            architecture=config.architecture,
            rule_set=config.rule_set,
            enforce_center_opening=config.enforce_center_opening,
        ).to(device)
    model = TorchPolicyValueModel(network, device=device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    replay = ReplayBuffer.load(run_paths.replay_path) if config.resume_run and run_paths.replay_path.exists() else ReplayBuffer(capacity=config.replay_capacity)
    history = read_training_history_csv(run_paths.history_csv_path) if config.resume_run else []
    start_iteration = int(history[-1]["iteration"]) if history else 0

    for iteration in range(1, config.iterations + 1):
        global_iteration = start_iteration + iteration
        samples_added = 0
        for _ in range(config.games_per_iteration):
            samples = generate_self_play_game(
                model,
                SelfPlayConfig(
                    board_size=config.board_size,
                    win_length=config.win_length,
                    rule_set=config.rule_set,
                    enforce_center_opening=config.enforce_center_opening,
                    mcts=MCTSConfig(
                        simulations=config.mcts_simulations,
                        root_dirichlet_alpha=config.root_dirichlet_alpha,
                        root_exploration_fraction=config.root_exploration_fraction,
                        add_root_noise=config.add_root_noise,
                    ),
                    input_channels=network.input_channels,
                    temperature_moves=config.temperature_moves,
                    late_temperature=config.late_temperature,
                ),
            )
            replay.add_game(samples)
            samples_added += len(samples)

        metrics = train_epoch_metrics(
            network,
            optimizer,
            replay,
            config.epochs,
            config.batches_per_epoch,
            config.batch_size,
            device,
            augment=config.augment_batches,
            gradient_clip_norm=config.gradient_clip_norm,
            use_amp=config.use_amp,
        )
        save_checkpoint(
            network,
            run_paths.checkpoint_path,
            metadata={
                "rule_set": config.rule_set,
                "enforce_center_opening": config.enforce_center_opening,
                "training_config": asdict(config),
                "run_dir": str(run_paths.run_dir),
            },
        )
        evaluation_metrics = evaluate_for_promotion(
            network,
            run_paths,
            config,
            device,
        )
        metrics.update(evaluation_metrics)
        should_promote = should_promote_checkpoint(run_paths, metrics, history, config)
        if should_promote:
            save_checkpoint(
                network,
                run_paths.best_checkpoint_path,
                metadata={
                    "rule_set": config.rule_set,
                    "enforce_center_opening": config.enforce_center_opening,
                    "training_config": asdict(config),
                    "run_dir": str(run_paths.run_dir),
                    "selection_metric": metrics.get("selection_metric", "loss"),
                    "selection_value": metrics.get("selection_value", metrics["loss"]),
                },
            )
        replay.save(run_paths.replay_path)
        numeric_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        history.append(
            {
                "iteration": float(global_iteration),
                "samples_added": float(samples_added),
                "replay_size": float(len(replay)),
                "train_steps": float(config.epochs * config.batches_per_epoch),
                **numeric_metrics,
            }
        )
        write_training_history_csv(history, run_paths.history_csv_path)
        plot_training_history(history, run_paths.plot_path)
        plot_policy_heatmap(
            network,
            run_paths.policy_heatmap_path,
            device,
            config.win_length,
            config.rule_set,
            config.enforce_center_opening,
        )

    return history


def resolve_training_run_paths(config: AlphaZeroTrainingConfig) -> TrainingRunPaths:
    if config.resume_run:
        run_dir = Path(config.resume_run)
    else:
        run_name = config.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(config.runs_dir) / run_name
    return TrainingRunPaths(
        run_dir=run_dir,
        checkpoint_path=Path(config.checkpoint_path) if config.checkpoint_path else run_dir / "checkpoints" / "latest.pt",
        best_checkpoint_path=run_dir / "checkpoints" / "best.pt",
        history_csv_path=Path(config.history_csv_path) if config.history_csv_path else run_dir / "metrics" / "history.csv",
        plot_path=Path(config.plot_path) if config.plot_path else run_dir / "plots" / "training.png",
        policy_heatmap_path=run_dir / "plots" / "policy_heatmap_empty.png",
        replay_path=run_dir / "replay" / "replay_buffer.pkl",
        config_path=run_dir / "config.json",
    )


def evaluate_for_promotion(
    network: GomokuPolicyValueNet,
    paths: TrainingRunPaths,
    config: AlphaZeroTrainingConfig,
    device: torch.device,
) -> dict[str, float | str]:
    if config.evaluation_games <= 0 or not paths.best_checkpoint_path.exists():
        return {}

    candidate = TorchPolicyValueModel(network, device=device)
    baseline_network = load_checkpoint(paths.best_checkpoint_path, device=device)
    baseline = TorchPolicyValueModel(baseline_network, device=device)
    rng = random.Random(config.evaluation_seed)
    score = {"candidate": 0, "baseline": 0, "draw": 0}
    total_moves = 0
    for game in range(1, config.evaluation_games + 1):
        winner, moves = play_evaluation_game(
            candidate,
            baseline,
            board_size=config.board_size,
            win_length=config.win_length,
            rule_set=config.rule_set,
            enforce_center_opening=config.enforce_center_opening,
            simulations=config.evaluation_simulations,
            candidate_is_black=game % 2 == 1,
            rng=rng,
        )
        score[winner] += 1
        total_moves += moves

    candidate_score = score["candidate"] + 0.5 * score["draw"]
    evaluation_score = candidate_score / float(config.evaluation_games)
    return {
        "evaluation_candidate_wins": float(score["candidate"]),
        "evaluation_baseline_wins": float(score["baseline"]),
        "evaluation_draws": float(score["draw"]),
        "evaluation_candidate_score": evaluation_score,
        "evaluation_mean_moves": total_moves / float(config.evaluation_games),
        "selection_metric": "evaluation_candidate_score",
        "selection_value": evaluation_score,
    }


def play_evaluation_game(
    candidate: TorchPolicyValueModel,
    baseline: TorchPolicyValueModel,
    board_size: int,
    win_length: int,
    rule_set: str,
    enforce_center_opening: bool,
    simulations: int,
    candidate_is_black: bool,
    rng: random.Random,
) -> tuple[str, int]:
    board = GomokuBoard(
        size=board_size,
        win_length=win_length,
        rule_set=rule_set,
        enforce_center_opening=enforce_center_opening,
    )
    config = MCTSConfig(simulations=simulations, temperature=0.0, add_root_noise=False)
    while board.winner is None and board.move_count < board.size * board.size:
        use_candidate = (board.current_player == Player.BLACK and candidate_is_black) or (
            board.current_player == Player.WHITE and not candidate_is_black
        )
        row, col = select_greedy_move(board, candidate if use_candidate else baseline, config)
        if (row, col) not in board.legal_moves():
            legal_moves = board.legal_moves()
            if not legal_moves:
                break
            row, col = rng.choice(legal_moves)
        board.place(row, col)

    if board.winner is None or board.winner == Player.EMPTY:
        return "draw", board.move_count
    candidate_won = (board.winner == Player.BLACK and candidate_is_black) or (
        board.winner == Player.WHITE and not candidate_is_black
    )
    return ("candidate" if candidate_won else "baseline"), board.move_count


def should_promote_checkpoint(
    paths: TrainingRunPaths,
    metrics: dict[str, float | str],
    history: list[dict[str, float]],
    config: AlphaZeroTrainingConfig,
) -> bool:
    if not paths.best_checkpoint_path.exists():
        metrics["selection_metric"] = "initial"
        metrics["selection_value"] = 1.0
        return True
    if config.evaluation_games > 0:
        score = float(metrics.get("evaluation_candidate_score", 0.0))
        return score >= config.promotion_threshold

    metrics["selection_metric"] = "loss"
    metrics["selection_value"] = float(metrics["loss"])
    return float(metrics["loss"]) <= min((item["loss"] for item in history), default=float("inf"))


def write_training_config(config: AlphaZeroTrainingConfig, paths: TrainingRunPaths) -> None:
    paths.config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["resolved_paths"] = {
        "run_dir": str(paths.run_dir),
        "checkpoint_path": str(paths.checkpoint_path),
        "history_csv_path": str(paths.history_csv_path),
        "plot_path": str(paths.plot_path),
        "policy_heatmap_path": str(paths.policy_heatmap_path),
        "replay_path": str(paths.replay_path),
    }
    paths.config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_training_history_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open(newline="") as csv_file:
        return [{key: float(value) for key, value in row.items() if value != ""} for row in csv.DictReader(csv_file)]


def write_training_history_csv(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "samples_added",
        "replay_size",
        "train_steps",
        "loss",
        "policy_loss",
        "value_loss",
        "policy_target_entropy",
        "evaluation_candidate_wins",
        "evaluation_baseline_wins",
        "evaluation_draws",
        "evaluation_candidate_score",
        "evaluation_mean_moves",
        "selection_value",
    ]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_training_history(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gomoku-vla")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [item["iteration"] for item in history]
    losses = [item["loss"] for item in history]
    policy_losses = [item.get("policy_loss", 0.0) for item in history]
    value_losses = [item.get("value_loss", 0.0) for item in history]
    replay_sizes = [item["replay_size"] for item in history]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, loss_ax = plt.subplots(figsize=(8, 4.5))
    replay_ax = loss_ax.twinx()

    loss_ax.plot(iterations, losses, color="tab:red", marker="o", linewidth=2, label="loss")
    if any(policy_losses):
        loss_ax.plot(iterations, policy_losses, color="tab:orange", linewidth=1.5, label="policy")
    if any(value_losses):
        loss_ax.plot(iterations, value_losses, color="tab:green", linewidth=1.5, label="value")
    replay_ax.plot(iterations, replay_sizes, color="tab:blue", marker="s", linewidth=1.5, label="replay size")

    loss_ax.set_xlabel("Iteration")
    loss_ax.set_ylabel("Mean training loss", color="tab:red")
    replay_ax.set_ylabel("Replay samples", color="tab:blue")
    loss_ax.tick_params(axis="y", labelcolor="tab:red")
    replay_ax.tick_params(axis="y", labelcolor="tab:blue")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.set_title("AlphaZero Self-Play Training")
    loss_ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_policy_heatmap(
    network: GomokuPolicyValueNet,
    path: Path,
    device: torch.device,
    win_length: int = 5,
    rule_set: str = "free",
    enforce_center_opening: bool = False,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gomoku-vla")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    board = GomokuBoard(
        size=network.board_size,
        win_length=win_length,
        rule_set=rule_set,
        enforce_center_opening=enforce_center_opening,
    )
    model = TorchPolicyValueModel(network, device=device)
    policy, value = model.predict(board)
    policy_board = policy.reshape(network.board_size, network.board_size)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    image = ax.imshow(policy_board, cmap="viridis")
    ax.set_title(f"Empty-board policy heatmap | value={value:.3f}")
    ax.set_xlabel("Col")
    ax.set_ylabel("Row")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_epochs(
    network: GomokuPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    device: torch.device,
    augment: bool = True,
    gradient_clip_norm: float = 5.0,
    use_amp: bool = False,
) -> float:
    return train_epoch_metrics(
        network,
        optimizer,
        replay,
        epochs,
        batches_per_epoch,
        batch_size,
        device,
        augment=augment,
        gradient_clip_norm=gradient_clip_norm,
        use_amp=use_amp,
    )["loss"]


def train_epoch_metrics(
    network: GomokuPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    device: torch.device,
    augment: bool = True,
    gradient_clip_norm: float = 5.0,
    use_amp: bool = False,
) -> dict[str, float]:
    if len(replay) == 0:
        raise ValueError("replay buffer is empty")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batches_per_epoch <= 0:
        raise ValueError("batches_per_epoch must be positive")
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    policy_target_entropies: list[float] = []
    network.train()
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    for _ in range(epochs):
        for _ in range(batches_per_epoch):
            batch = replay.sample(batch_size, augment=augment)
            states = torch.from_numpy(batch.states).to(device)
            policy_targets = torch.from_numpy(batch.policy_targets).to(device)
            value_targets = torch.from_numpy(batch.value_targets).to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                policy_logits, values = network(states)
                policy_loss, value_loss = policy_value_loss_components(policy_logits, values, policy_targets, value_targets)
                loss = policy_loss + value_loss
            scaler.scale(loss).backward()
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            target_entropy = -(policy_targets * torch.log(policy_targets.clamp_min(1e-8))).sum(dim=1).mean()
            policy_target_entropies.append(float(target_entropy.item()))
    return {
        "loss": sum(losses) / len(losses),
        "policy_loss": sum(policy_losses) / len(policy_losses),
        "value_loss": sum(value_losses) / len(value_losses),
        "policy_target_entropy": sum(policy_target_entropies) / len(policy_target_entropies),
    }
