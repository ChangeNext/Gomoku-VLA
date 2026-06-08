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


if __name__ == "__main__":
    unittest.main()
