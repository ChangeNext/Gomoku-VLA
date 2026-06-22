import unittest

from board import GomokuBoard, Player


class GomokuBoardTest(unittest.TestCase):
    def test_rejects_occupied_cell(self) -> None:
        board = GomokuBoard()
        board.place(7, 7)
        with self.assertRaises(ValueError):
            board.place(7, 7)

    def test_detects_horizontal_win(self) -> None:
        board = GomokuBoard()
        moves = [(7, 3), (8, 3), (7, 4), (8, 4), (7, 5), (8, 5), (7, 6), (8, 6), (7, 7)]
        winner = None
        for move in moves:
            winner = board.place(*move)
        self.assertEqual(winner, Player.BLACK)
        self.assertEqual(board.winner, Player.BLACK)

    def test_renju_center_opening_requires_first_move_at_center(self) -> None:
        board = GomokuBoard(rule_set="renju", enforce_center_opening=True)
        self.assertFalse(board.is_legal_move(0, 0))
        self.assertTrue(board.is_legal_move(7, 7))

    def test_renju_rejects_black_overline(self) -> None:
        board = GomokuBoard(rule_set="renju")
        for col in range(3, 8):
            board.grid[7][col] = Player.BLACK.value
            board.move_count += 1

        with self.assertRaisesRegex(ValueError, "illegal move"):
            board.place(7, 8)

    def test_renju_allows_white_overline_to_win(self) -> None:
        board = GomokuBoard(rule_set="renju")
        board.current_player = Player.WHITE
        for col in range(3, 8):
            board.grid[7][col] = Player.WHITE.value
            board.move_count += 1

        self.assertEqual(board.place(7, 8), Player.WHITE)

    def test_renju_rejects_black_double_three(self) -> None:
        board = GomokuBoard(rule_set="renju")
        for row, col in ((7, 6), (7, 8), (6, 7), (8, 7)):
            board.grid[row][col] = Player.BLACK.value
            board.move_count += 1

        self.assertTrue(board.is_forbidden_move(7, 7))
        self.assertFalse(board.is_legal_move(7, 7))

    def test_renju_rejects_black_double_four(self) -> None:
        board = GomokuBoard(rule_set="renju")
        for row, col in ((7, 5), (7, 6), (7, 8), (5, 7), (6, 7), (8, 7)):
            board.grid[row][col] = Player.BLACK.value
            board.move_count += 1

        self.assertTrue(board.is_forbidden_move(7, 7))
        self.assertFalse(board.is_legal_move(7, 7))

    def test_renju_black_exact_five_wins(self) -> None:
        board = GomokuBoard(rule_set="renju")
        for col in range(3, 7):
            board.grid[7][col] = Player.BLACK.value
            board.move_count += 1

        self.assertEqual(board.place(7, 7), Player.BLACK)


if __name__ == "__main__":
    unittest.main()
