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
    parser.add_argument("--win-length", type=int, default=5)
    parser.add_argument("--rule-set", choices=("free", "renju"))
    parser.add_argument("--center-opening", action="store_true")
    parser.add_argument("--no-center-opening", action="store_true")
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--policy-source", default="alphazero")
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be positive")
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
    policy = CheckpointPolicy(checkpoint, device=args.device, simulations=args.simulations)
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
