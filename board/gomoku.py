from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Player(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> "Player":
        if self == Player.BLACK:
            return Player.WHITE
        if self == Player.WHITE:
            return Player.BLACK
        raise ValueError("EMPTY has no opponent")


BoardState = list[list[int]]


@dataclass
class GomokuBoard:
    size: int = 15
    win_length: int = 5
    grid: BoardState = field(init=False)
    current_player: Player = field(default=Player.BLACK, init=False)
    move_count: int = field(default=0, init=False)
    winner: Player | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.size < self.win_length:
            raise ValueError("board size must be at least win_length")
        self.grid = [[Player.EMPTY.value for _ in range(self.size)] for _ in range(self.size)]

    def reset(self) -> None:
        self.grid = [[Player.EMPTY.value for _ in range(self.size)] for _ in range(self.size)]
        self.current_player = Player.BLACK
        self.move_count = 0
        self.winner = None

    def is_on_board(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def is_legal_move(self, row: int, col: int) -> bool:
        return self.winner is None and self.is_on_board(row, col) and self.grid[row][col] == Player.EMPTY

    def place(self, row: int, col: int) -> Player | None:
        if not self.is_legal_move(row, col):
            raise ValueError(f"illegal move: row={row}, col={col}")

        player = self.current_player
        self.grid[row][col] = player.value
        self.move_count += 1

        if self._has_five_from(row, col, player):
            self.winner = player
        elif self.move_count == self.size * self.size:
            self.winner = Player.EMPTY
        else:
            self.current_player = player.opponent

        return self.winner

    def legal_moves(self) -> list[tuple[int, int]]:
        if self.winner is not None:
            return []
        return [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if self.grid[row][col] == Player.EMPTY
        ]

    def copy_state(self) -> BoardState:
        return [row[:] for row in self.grid]

    def _has_five_from(self, row: int, col: int, player: Player) -> bool:
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        return any(
            1 + self._count(row, col, dr, dc, player) + self._count(row, col, -dr, -dc, player)
            >= self.win_length
            for dr, dc in directions
        )

    def _count(self, row: int, col: int, dr: int, dc: int, player: Player) -> int:
        total = 0
        row += dr
        col += dc
        while self.is_on_board(row, col) and self.grid[row][col] == player.value:
            total += 1
            row += dr
            col += dc
        return total
