from __future__ import annotations

import argparse

from gomoku_ai.train import AlphaZeroTrainingConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small AlphaZero-style training loop.")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--win-length", type=int, default=5)
    parser.add_argument("--rule-set", choices=("free", "renju"), default="renju")
    parser.add_argument("--no-center-opening", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--self-play-batch-size", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batches-per-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs-dir", default="gomoku_ai/runs")
    parser.add_argument("--run-name")
    parser.add_argument("--resume-run")
    parser.add_argument("--architecture", choices=("resnet", "legacy_cnn"), default="resnet")
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--res-blocks", type=int, default=16)
    parser.add_argument("--input-channels", type=int, default=6)
    parser.add_argument("--temperature-moves", type=int, default=16)
    parser.add_argument("--late-temperature", type=float, default=0.1)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.03)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--no-root-noise", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--history-csv", help="Write per-iteration metrics to this CSV file.")
    parser.add_argument("--plot", help="Write a PNG training plot after each iteration.")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--evaluation-games", type=int, default=0)
    parser.add_argument("--evaluation-simulations", type=int, default=64)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    history = run_training(
        AlphaZeroTrainingConfig(
            board_size=args.board_size,
            win_length=args.win_length,
            rule_set=args.rule_set,
            enforce_center_opening=args.rule_set == "renju" and not args.no_center_opening,
            iterations=args.iterations,
            games_per_iteration=args.games,
            self_play_batch_size=args.self_play_batch_size,
            mcts_simulations=args.simulations,
            epochs=args.epochs,
            batches_per_epoch=args.batches_per_epoch,
            batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
            learning_rate=args.learning_rate,
            checkpoint_path=args.checkpoint,
            initial_checkpoint_path=args.initial_checkpoint,
            history_csv_path=args.history_csv,
            plot_path=args.plot,
            runs_dir=args.runs_dir,
            run_name=args.run_name,
            resume_run=args.resume_run,
            architecture=args.architecture,
            channels=args.channels,
            res_blocks=args.res_blocks,
            input_channels=args.input_channels,
            temperature_moves=args.temperature_moves,
            late_temperature=args.late_temperature,
            root_dirichlet_alpha=args.root_dirichlet_alpha,
            root_exploration_fraction=args.root_exploration_fraction,
            add_root_noise=not args.no_root_noise,
            device=args.device,
            augment_batches=not args.no_augment,
            gradient_clip_norm=args.gradient_clip_norm,
            use_amp=args.amp,
            evaluation_games=args.evaluation_games,
            evaluation_simulations=args.evaluation_simulations,
            promotion_threshold=args.promotion_threshold,
            evaluation_seed=args.evaluation_seed,
            show_progress=not args.no_progress,
        )
    )
    for item in history:
        print(
            "iteration={iteration:.0f} samples_added={samples_added:.0f} "
            "replay_size={replay_size:.0f} train_steps={train_steps:.0f} "
            "loss={loss:.4f} policy_loss={policy_loss:.4f} "
            "value_loss={value_loss:.4f}".format(**item),
            flush=True,
        )


if __name__ == "__main__":
    main()
