from __future__ import annotations

import argparse

from gomoku_ai import MCTSConfig, SelfPlayConfig, UniformPolicyValueModel, generate_self_play_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one AlphaZero-style self-play game.")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--win-length", type=int, default=5)
    parser.add_argument("--rule-set", choices=("free", "renju"), default="free")
    parser.add_argument("--center-opening", action="store_true")
    parser.add_argument("--simulations", type=int, default=32)
    args = parser.parse_args()

    config = SelfPlayConfig(
        board_size=args.board_size,
        win_length=args.win_length,
        rule_set=args.rule_set,
        enforce_center_opening=args.center_opening,
        mcts=MCTSConfig(simulations=args.simulations),
    )
    samples = generate_self_play_game(UniformPolicyValueModel(), config)
    outcome = samples[-1].value_target if samples else 0.0
    print(f"generated_samples={len(samples)} final_first_player_value={outcome:+.1f}")


if __name__ == "__main__":
    main()
