from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal


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
RuleSet = Literal["free", "renju"]


@dataclass
class GomokuBoard:
    size: int = 15
    win_length: int = 5
    rule_set: RuleSet = "free"
    enforce_center_opening: bool = False
    grid: BoardState = field(init=False)
    current_player: Player = field(default=Player.BLACK, init=False)
    move_count: int = field(default=0, init=False)
    winner: Player | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.rule_set not in {"free", "renju"}:
            raise ValueError(f"unsupported rule_set: {self.rule_set}")
        if self.size < self.win_length:
            raise ValueError("board size must be at least win_length")
        if self.rule_set == "renju" and self.win_length != 5:
            raise ValueError("renju rule_set requires win_length=5")
        self.grid = [[Player.EMPTY.value for _ in range(self.size)] for _ in range(self.size)]

    def reset(self) -> None:
        self.grid = [[Player.EMPTY.value for _ in range(self.size)] for _ in range(self.size)]
        self.current_player = Player.BLACK
        self.move_count = 0
        self.winner = None

    def is_on_board(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def is_legal_move(self, row: int, col: int) -> bool:
        if self.winner is not None or not self.is_on_board(row, col) or self.grid[row][col] != Player.EMPTY:
            return False
        if self.enforce_center_opening and self.move_count == 0:
            center = self.size // 2
            return row == center and col == center
        return not self.is_forbidden_move(row, col)

    def is_forbidden_move(self, row: int, col: int) -> bool:
        if self.rule_set != "renju" or self.current_player != Player.BLACK:
            return False
        if not self.is_on_board(row, col) or self.grid[row][col] != Player.EMPTY:
            return True

        self.grid[row][col] = Player.BLACK.value
        try:
            if any(self._line_length(row, col, dr, dc, Player.BLACK) > self.win_length for dr, dc in _DIRECTIONS):
                return True
            return self._open_four_count(row, col, Player.BLACK) >= 2 or self._open_three_count(row, col, Player.BLACK) >= 2
        finally:
            self.grid[row][col] = Player.EMPTY.value

    def place(self, row: int, col: int) -> Player | None:
        if not self.is_legal_move(row, col):
            raise ValueError(f"illegal move: row={row}, col={col}")

        player = self.current_player
        self.grid[row][col] = player.value
        self.move_count += 1

        if self._is_winning_move(row, col, player):
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
            and self.is_legal_move(row, col)
        ]

    def copy_state(self) -> BoardState:
        return [row[:] for row in self.grid]

    def _has_five_from(self, row: int, col: int, player: Player) -> bool:
        return any(
            self._line_length(row, col, dr, dc, player) >= self.win_length
            for dr, dc in _DIRECTIONS
        )

    def _is_winning_move(self, row: int, col: int, player: Player) -> bool:
        lengths = [self._line_length(row, col, dr, dc, player) for dr, dc in _DIRECTIONS]
        if self.rule_set == "renju" and player == Player.BLACK:
            return any(length == self.win_length for length in lengths)
        return any(length >= self.win_length for length in lengths)

    def _line_length(self, row: int, col: int, dr: int, dc: int, player: Player) -> int:
        return 1 + self._count(row, col, dr, dc, player) + self._count(row, col, -dr, -dc, player)

    def _count(self, row: int, col: int, dr: int, dc: int, player: Player) -> int:
        total = 0
        row += dr
        col += dc
        while self.is_on_board(row, col) and self.grid[row][col] == player.value:
            total += 1
            row += dr
            col += dc
        return total

    def _open_four_count(self, row: int, col: int, player: Player) -> int:
        return sum(
            1
            for dr, dc in _DIRECTIONS
            if self._winning_completions_in_direction(row, col, dr, dc, player) > 0
        )

    def _open_three_count(self, row: int, col: int, player: Player) -> int:
        total = 0
        for dr, dc in _DIRECTIONS:
            if self._creates_open_four_in_direction(row, col, dr, dc, player):
                total += 1
        return total

    def _creates_open_four_in_direction(
        self,
        row: int,
        col: int,
        dr: int,
        dc: int,
        player: Player,
    ) -> bool:
        for candidate_row, candidate_col in self._line_window(row, col, dr, dc):
            if self.grid[candidate_row][candidate_col] != Player.EMPTY.value:
                continue
            self.grid[candidate_row][candidate_col] = player.value
            try:
                line_length = self._line_length(candidate_row, candidate_col, dr, dc, player)
                if line_length < self.win_length - 1:
                    continue
                if line_length >= self.win_length:
                    continue
                if self._winning_completions_in_direction(row, col, dr, dc, player) >= 2:
                    return True
            finally:
                self.grid[candidate_row][candidate_col] = Player.EMPTY.value
        return False

    def _winning_completions_in_direction(
        self,
        row: int,
        col: int,
        dr: int,
        dc: int,
        player: Player,
    ) -> int:
        completions = 0
        for candidate_row, candidate_col in self._line_window(row, col, dr, dc):
            if self.grid[candidate_row][candidate_col] != Player.EMPTY.value:
                continue
            self.grid[candidate_row][candidate_col] = player.value
            try:
                if self._line_length(candidate_row, candidate_col, dr, dc, player) == self.win_length:
                    completions += 1
            finally:
                self.grid[candidate_row][candidate_col] = Player.EMPTY.value
        return completions

    def _line_window(self, row: int, col: int, dr: int, dc: int) -> list[tuple[int, int]]:
        cells = []
        for offset in range(-self.win_length, self.win_length + 1):
            candidate_row = row + offset * dr
            candidate_col = col + offset * dc
            if self.is_on_board(candidate_row, candidate_col):
                cells.append((candidate_row, candidate_col))
        return cells


_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
