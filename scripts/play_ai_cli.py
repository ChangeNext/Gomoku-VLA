from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch

from board import GomokuBoard, Player
from gomoku_ai.mcts import MCTSConfig
from gomoku_ai.self_play import select_greedy_move
from gomoku_ai.torch_model import TorchPolicyValueModel, load_checkpoint


OutputFn = Callable[[str], None]


def print_board(board: GomokuBoard, output: OutputFn = print) -> None:
    symbols = {
        Player.EMPTY.value: ".",
        Player.BLACK.value: "B",
        Player.WHITE.value: "W",
    }
    header = "   " + " ".join(f"{col:02d}" for col in range(board.size))
    output(header)
    for row, values in enumerate(board.grid):
        output(f"{row:02d} " + "  ".join(symbols[value] for value in values))


def parse_human_move(raw: str) -> tuple[int, int] | None:
    if raw.lower() in {"q", "quit", "exit"}:
        return None
    row_text, col_text = raw.split()
    return int(row_text), int(col_text)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def play_game(
    board: GomokuBoard,
    model: TorchPolicyValueModel,
    human_player: Player,
    simulations: int,
    input_fn: Callable[[str], str] = input,
    output: OutputFn = print,
) -> None:
    if human_player not in {Player.BLACK, Player.WHITE}:
        raise ValueError("human_player must be BLACK or WHITE")

    ai_player = human_player.opponent
    mcts_config = MCTSConfig(simulations=simulations, temperature=0.0)

    output(f"Human: {human_player.name} | AI: {ai_player.name}")
    while board.winner is None:
        print_board(board, output)
        if board.current_player == human_player:
            prompt = f"{human_player.name} move as 'row col' or 'q': "
            try:
                raw = input_fn(prompt).strip()
                move = parse_human_move(raw)
                if move is None:
                    output("Game stopped")
                    return
                board.place(*move)
            except EOFError:
                output("Game stopped")
                return
            except ValueError as exc:
                output(f"Invalid move: {exc}")
        else:
            row, col = select_greedy_move(board, model, mcts_config)
            output(f"AI {ai_player.name}: {row} {col}")
            board.place(row, col)

    print_board(board, output)
    if board.winner == Player.EMPTY:
        output("Draw")
    else:
        output(f"Winner: {board.winner.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against a trained Gomoku policy/value checkpoint.")
    parser.add_argument("--checkpoint", default="gomoku_ai/runs/alphazero_9x9_long/checkpoints/latest.pt")
    parser.add_argument("--win-length", type=int, default=5)
    parser.add_argument("--rule-set", choices=("free", "renju"), help="Defaults to the checkpoint rule_set metadata.")
    parser.add_argument("--center-opening", action="store_true", help="Force the first black move to the center.")
    parser.add_argument("--no-center-opening", action="store_true", help="Disable checkpoint center-opening metadata.")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    parser.add_argument("--human", choices=("black", "white"), default="black")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        parser.error(f"checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    network = load_checkpoint(checkpoint_path, device=device)
    model = TorchPolicyValueModel(network, device=device)
    rule_set = args.rule_set or network.rule_set
    enforce_center_opening = network.enforce_center_opening
    if args.center_opening:
        enforce_center_opening = True
    if args.no_center_opening:
        enforce_center_opening = False
    board = GomokuBoard(
        size=network.board_size,
        win_length=args.win_length,
        rule_set=rule_set,
        enforce_center_opening=enforce_center_opening,
    )
    human_player = Player.BLACK if args.human == "black" else Player.WHITE
    print(f"Loaded {checkpoint_path} on {device}")
    play_game(board, model, human_player, simulations=args.simulations)


if __name__ == "__main__":
    main()
