from __future__ import annotations

import argparse
from pathlib import Path

from gomoku_ai.episode_recorder import default_episode_output_path, play_and_record_episode
from gomoku_ai.inference import CheckpointPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate move-level JSONL episodes from a trained Gomoku checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--sample-moves",
        action="store_true",
        help="Sample teacher moves from the MCTS policy instead of always taking argmax.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--temperature-moves", type=int, default=0)
    parser.add_argument("--late-temperature", type=float, default=0.0)
    parser.add_argument("--root-noise", action="store_true")
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--win-length", type=int, help="Override checkpoint win length.")
    parser.add_argument("--rule-set", choices=("free", "renju"))
    parser.add_argument("--center-opening", action="store_true")
    parser.add_argument("--no-center-opening", action="store_true")
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--policy-source", default="alphazero")
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be positive")
    if args.temperature < 0.0:
        parser.error("--temperature must be non-negative")
    if args.late_temperature < 0.0:
        parser.error("--late-temperature must be non-negative")
    if args.temperature_moves < 0:
        parser.error("--temperature-moves must be non-negative")
    if args.root_dirichlet_alpha <= 0.0:
        parser.error("--root-dirichlet-alpha must be positive")
    if not 0.0 <= args.root_exploration_fraction <= 1.0:
        parser.error("--root-exploration-fraction must be between 0 and 1")
    if args.center_opening and args.no_center_opening:
        parser.error("--center-opening and --no-center-opening cannot be used together")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        parser.error(f"checkpoint not found: {checkpoint}")

    enforce_center_opening = None
    if args.center_opening:
        enforce_center_opening = True
    if args.no_center_opening:
        enforce_center_opening = False

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else default_episode_output_path(checkpoint)
    policy = CheckpointPolicy(
        checkpoint,
        device=args.device,
        simulations=args.simulations,
        temperature=args.temperature,
        temperature_moves=args.temperature_moves,
        late_temperature=args.late_temperature,
        sample_moves=args.sample_moves,
        add_root_noise=args.root_noise,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        seed=args.seed,
    )
    total_records = 0
    for game_index in range(args.games):
        board = policy.new_board(
            win_length=args.win_length,
            rule_set=args.rule_set,
            enforce_center_opening=enforce_center_opening,
        )
        records = play_and_record_episode(
            board,
            policy,
            output_jsonl,
            game_id=f"{checkpoint.stem}-{game_index + 1}",
            policy_source=args.policy_source,
            checkpoint=str(checkpoint),
            max_moves=args.max_moves,
        )
        total_records += len(records)

    print(f"wrote {total_records} move records from {args.games} games to {output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
