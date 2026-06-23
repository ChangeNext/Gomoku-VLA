from __future__ import annotations

from board import GomokuBoard, Player

from .encoding import clone_board


def find_immediate_win(board: GomokuBoard, player: Player) -> tuple[int, int] | None:
    if board.winner is not None:
        return None
    for row, col in _legal_moves_for_player(board, player):
        candidate = clone_board(board)
        candidate.current_player = player
        if candidate.place(row, col) == player:
            return row, col
    return None


def find_forced_block(board: GomokuBoard) -> tuple[int, int] | None:
    if board.winner is not None:
        return None
    return find_immediate_win(board, board.current_player.opponent)


def select_tactical_move(board: GomokuBoard) -> tuple[int, int] | None:
    winning_move = find_immediate_win(board, board.current_player)
    if winning_move is not None:
        return winning_move
    forced_block = find_forced_block(board)
    if forced_block is not None:
        return forced_block

    own_threat = find_best_threat_move(board, board.current_player)
    if own_threat is not None:
        return own_threat
    return find_best_threat_move(board, board.current_player.opponent)


def find_best_threat_move(board: GomokuBoard, player: Player) -> tuple[int, int] | None:
    best_move: tuple[int, int] | None = None
    best_score = 0
    for row, col in _legal_moves_for_player(board, player):
        score = _threat_score_after_move(board, row, col, player)
        if score > best_score:
            best_score = score
            best_move = (row, col)
    return best_move if best_score >= 100 else None


def _threat_score_after_move(board: GomokuBoard, row: int, col: int, player: Player) -> int:
    candidate = clone_board(board)
    candidate.current_player = player
    try:
        candidate.place(row, col)
    except ValueError:
        return 0
    open_fours = 0
    closed_fours = 0
    open_threes = 0
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        length, open_ends = _line_shape(candidate, row, col, dr, dc, player)
        if length >= candidate.win_length - 1:
            if open_ends >= 2:
                open_fours += 1
            elif open_ends == 1:
                closed_fours += 1
        elif length == candidate.win_length - 2 and open_ends >= 2:
            open_threes += 1
    if open_fours:
        return 400 + 50 * open_fours
    if closed_fours >= 2 or (closed_fours and open_threes):
        return 300 + 25 * closed_fours + 10 * open_threes
    if open_threes >= 2:
        return 200 + 10 * open_threes
    if closed_fours:
        return 150 + 10 * closed_fours
    if open_threes:
        return 100 + open_threes
    return 0


def _line_shape(
    board: GomokuBoard,
    row: int,
    col: int,
    dr: int,
    dc: int,
    player: Player,
) -> tuple[int, int]:
    forward_count, forward_open = _count_side(board, row, col, dr, dc, player)
    backward_count, backward_open = _count_side(board, row, col, -dr, -dc, player)
    return 1 + forward_count + backward_count, int(forward_open) + int(backward_open)


def _count_side(
    board: GomokuBoard,
    row: int,
    col: int,
    dr: int,
    dc: int,
    player: Player,
) -> tuple[int, bool]:
    count = 0
    row += dr
    col += dc
    while board.is_on_board(row, col) and board.grid[row][col] == player.value:
        count += 1
        row += dr
        col += dc
    return count, board.is_on_board(row, col) and board.grid[row][col] == Player.EMPTY.value


def _legal_moves_for_player(board: GomokuBoard, player: Player) -> list[tuple[int, int]]:
    candidate = clone_board(board)
    candidate.current_player = player
    return candidate.legal_moves()
