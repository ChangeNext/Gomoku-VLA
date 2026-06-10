from __future__ import annotations

import argparse

from gomoku_ai.train import AlphaZeroTrainingConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small AlphaZero-style training loop.")
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--win-length", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="checkpoints/alphazero_latest.pt")
    args = parser.parse_args()

    history = run_training(
        AlphaZeroTrainingConfig(
            board_size=args.board_size,
            win_length=args.win_length,
            iterations=args.iterations,
            games_per_iteration=args.games,
            mcts_simulations=args.simulations,
            epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    )
    for item in history:
        print(
            "iteration={iteration:.0f} samples_added={samples_added:.0f} "
            "replay_size={replay_size:.0f} loss={loss:.4f}".format(**item)
        )


if __name__ == "__main__":
    main()
