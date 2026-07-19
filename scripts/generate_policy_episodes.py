from __future__ import annotations

import argparse
from pathlib import Path

from gomoku_ai.episode_recorder import default_episode_output_path, play_and_record_episode
from gomoku_ai.external_engine import build_piskvork_policy
from gomoku_ai.inference import CheckpointPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate move-level JSONL episodes from a trained Gomoku checkpoint.")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--engine-command",
        help="Piskvork/Gomocup engine command, for example a Rapfi executable path. Mutually exclusive with --checkpoint.",
    )
    parser.add_argument("--engine-timeout-turn-ms", type=int, default=1000)
    parser.add_argument("--engine-protocol-timeout-s", type=float, default=5.0)
    parser.add_argument("--board-size", type=int, default=15)
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

    if bool(args.checkpoint) == bool(args.engine_command):
        parser.error("provide exactly one of --checkpoint or --engine-command")
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.board_size <= 0:
        parser.error("--board-size must be positive")
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
    if args.engine_timeout_turn_ms <= 0:
        parser.error("--engine-timeout-turn-ms must be positive")
    if args.engine_protocol_timeout_s <= 0.0:
        parser.error("--engine-protocol-timeout-s must be positive")

    enforce_center_opening = None
    if args.center_opening:
        enforce_center_opening = True
    if args.no_center_opening:
        enforce_center_opening = False

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            parser.error(f"checkpoint not found: {checkpoint}")
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
        policy_source = args.policy_source
        checkpoint_label = str(checkpoint)
        game_id_prefix = checkpoint.stem
    else:
        rule_set = args.rule_set or "renju"
        center_opening = True if enforce_center_opening is None else enforce_center_opening
        output_jsonl = Path(args.output_jsonl) if args.output_jsonl else Path("data/external_engine_policy_episodes.jsonl")
        policy = build_piskvork_policy(
            args.engine_command,
            board_size=args.board_size,
            win_length=args.win_length or 5,
            rule_set=rule_set,
            enforce_center_opening=center_opening,
            timeout_turn_ms=args.engine_timeout_turn_ms,
            protocol_timeout_s=args.engine_protocol_timeout_s,
        )
        policy_source = args.policy_source if args.policy_source != "alphazero" else f"piskvork:{policy.name}"
        checkpoint_label = None
        game_id_prefix = policy.name

    total_records = 0
    try:
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
                game_id=f"{game_id_prefix}-{game_index + 1}",
                policy_source=policy_source,
                checkpoint=checkpoint_label,
                max_moves=args.max_moves,
            )
            total_records += len(records)
    finally:
        close = getattr(policy, "close", None)
        if close is not None:
            close()

    print(f"wrote {total_records} move records from {args.games} games to {output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
