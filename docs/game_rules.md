# Game Rules and Board State

`board.GomokuBoard` is the authoritative game-state implementation. Other
packages must call its legality APIs instead of duplicating Gomoku or Renju
rules.

## State Model

The board stores a square integer grid using `Player.EMPTY`, `Player.BLACK`,
and `Player.WHITE`. It also owns the current player, move count, and terminal
winner. `Player.EMPTY` as the winner represents a full-board draw.

Coordinates are zero-based `(row, col)`. `(0, 0)` is the upper-left
intersection in rendered board views. `place()` rejects out-of-range,
occupied, forbidden, post-game, and invalid opening moves with `ValueError`.

## Supported Rule Sets

`free` allows both players to win with a line of at least `win_length`.

`renju` requires `win_length=5` and applies restrictions only to Black:

- an overline is forbidden;
- double-four and double-three moves are forbidden by the current pattern
  detector;
- Black wins with exactly five;
- White wins with five or more.

`enforce_center_opening=True` independently requires the first move to be the
center cell. Training checkpoints store the rule set and opening flag so
inference can construct a matching board.

The Renju detector is a compact project implementation, not a formally
certified implementation of every edge case in the complete Renju rulebook.
Extend it with focused regression positions before relying on a new pattern.

## Public Contract

- `is_on_board(row, col)`: bounds check
- `is_legal_move(row, col)`: complete current-state authorization
- `is_forbidden_move(row, col)`: current Black Renju restriction check
- `place(row, col)`: mutate state and update winner/turn
- `legal_moves()`: all currently authorized moves
- `copy_state()`: detached grid copy for observations and records

Policy, UI, simulation, and robot code should use `is_legal_move()` or
`place()` at the final boundary. A policy probability or predicted coordinate
is never sufficient authorization by itself.

## Tests

`tests/test_board.py` covers occupied-cell rejection, horizontal wins, center
opening, Black overline/double-three/double-four restrictions, White overline,
and Black exact-five victory. Add a regression test for every new rule pattern
or bug.
