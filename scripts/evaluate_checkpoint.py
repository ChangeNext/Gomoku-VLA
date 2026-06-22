from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from board import GomokuBoard, Player
from gomoku_ai.mcts import MCTSConfig
from gomoku_ai.self_play import select_greedy_move
from gomoku_ai.torch_model import TorchPolicyValueModel, load_checkpoint


def play_match_game(
    candidate: TorchPolicyValueModel,
    baseline: TorchPolicyValueModel,
    board_size: int,
    win_length: int,
    rule_set: str,
    enforce_center_opening: bool,
    simulations: int,
    candidate_is_black: bool,
    opening_random_moves: int,
    rng: random.Random,
) -> tuple[str, int]:
    board = GomokuBoard(
        size=board_size,
        win_length=win_length,
        rule_set=rule_set,
        enforce_center_opening=enforce_center_opening,
    )
    for _ in range(opening_random_moves):
        legal_moves = board.legal_moves()
        if not legal_moves:
            break
        board.place(*rng.choice(legal_moves))

    config = MCTSConfig(simulations=simulations, temperature=0.0)
    while board.winner is None and board.move_count < board.size * board.size:
        use_candidate = (board.current_player == Player.BLACK and candidate_is_black) or (
            board.current_player == Player.WHITE and not candidate_is_black
        )
        row, col = select_greedy_move(board, candidate if use_candidate else baseline, config)
        board.place(row, col)

    if board.winner is None or board.winner == Player.EMPTY:
        return "draw", board.move_count
    candidate_won = (board.winner == Player.BLACK and candidate_is_black) or (
        board.winner == Player.WHITE and not candidate_is_black
    )
    return ("candidate" if candidate_won else "baseline"), board.move_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two Gomoku policy/value checkpoints.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--win-length", type=int, default=5)
    parser.add_argument("--rule-set", choices=("free", "renju"), help="Defaults to the candidate checkpoint rule_set metadata.")
    parser.add_argument("--center-opening", action="store_true")
    parser.add_argument("--no-center-opening", action="store_true")
    parser.add_argument("--opening-random-moves", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default="gomoku_ai/runs/adhoc/metrics/evaluation.csv")
    parser.add_argument("--promote-to", help="Copy candidate here if it reaches --promotion-threshold.")
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    args = parser.parse_args()

    candidate_network = load_checkpoint(args.candidate, device="cpu")
    baseline_network = load_checkpoint(args.baseline, device="cpu")
    if candidate_network.board_size != baseline_network.board_size:
        parser.error("candidate and baseline board sizes must match")
    rule_set = args.rule_set or candidate_network.rule_set
    enforce_center_opening = candidate_network.enforce_center_opening
    if args.center_opening:
        enforce_center_opening = True
    if args.no_center_opening:
        enforce_center_opening = False

    candidate = TorchPolicyValueModel(candidate_network, device="cpu")
    baseline = TorchPolicyValueModel(baseline_network, device="cpu")
    rng = random.Random(args.seed)
    rows: list[dict[str, str | int]] = []
    score = {"candidate": 0, "baseline": 0, "draw": 0}
    for game in range(1, args.games + 1):
        candidate_is_black = game % 2 == 1
        winner, moves = play_match_game(
            candidate,
            baseline,
            board_size=candidate_network.board_size,
            win_length=args.win_length,
            rule_set=rule_set,
            enforce_center_opening=enforce_center_opening,
            simulations=args.simulations,
            candidate_is_black=candidate_is_black,
            opening_random_moves=args.opening_random_moves,
            rng=rng,
        )
        score[winner] += 1
        rows.append(
            {
                "game": game,
                "candidate_color": "black" if candidate_is_black else "white",
                "winner": winner,
                "moves": moves,
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["game", "candidate_color", "winner", "moves"])
        writer.writeheader()
        writer.writerows(rows)

    candidate_score = score["candidate"] + 0.5 * score["draw"]
    win_rate = candidate_score / args.games
    if args.promote_to and win_rate >= args.promotion_threshold:
        Path(args.promote_to).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.candidate, args.promote_to)
    print(
        "candidate={candidate} baseline={baseline} draw={draw} "
        "candidate_score={win_rate:.3f}".format(win_rate=win_rate, **score),
        flush=True,
    )


if __name__ == "__main__":
    main()
