from __future__ import annotations

from board import Player
from simulation import GomokuMujocoEnv


def print_board(env: GomokuMujocoEnv) -> None:
    symbols = {
        Player.EMPTY.value: ".",
        Player.BLACK.value: "B",
        Player.WHITE.value: "W",
    }
    header = "   " + " ".join(f"{col:02d}" for col in range(env.board.size))
    print(header)
    for row, values in enumerate(env.board.grid):
        print(f"{row:02d} " + "  ".join(symbols[value] for value in values))


def main() -> None:
    env = GomokuMujocoEnv()
    while env.board.winner is None:
        print_board(env)
        player = "BLACK" if env.board.current_player == Player.BLACK else "WHITE"
        raw = input(f"{player} move as 'row col' or 'q': ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            break
        try:
            row_text, col_text = raw.split()
            env.step((int(row_text), int(col_text)))
        except ValueError as exc:
            print(f"Invalid move: {exc}")

    print_board(env)
    if env.board.winner == Player.EMPTY:
        print("Draw")
    elif env.board.winner is not None:
        print(f"Winner: {env.board.winner.name}")


if __name__ == "__main__":
    main()
