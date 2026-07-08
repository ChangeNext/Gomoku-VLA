import unittest

import numpy as np

from board import Player
from vision import BoardCalibration, GridBoardDetector


class GridBoardDetectorTest(unittest.TestCase):
    def test_detects_black_and_white_stones_from_calibrated_grid(self) -> None:
        image = np.full((80, 80, 3), [190, 135, 70], dtype=np.uint8)
        calibration = BoardCalibration(
            board_size=3,
            top_left=(20, 20),
            top_right=(60, 20),
            bottom_left=(20, 60),
        )
        image[16:25, 16:25] = [20, 20, 20]
        image[56:65, 56:65] = [235, 235, 225]

        board = GridBoardDetector(calibration, sample_radius=3).detect(image)

        self.assertEqual(board[0][0], Player.BLACK.value)
        self.assertEqual(board[2][2], Player.WHITE.value)
        self.assertEqual(board[1][1], Player.EMPTY.value)

    def test_rejects_non_rgb_images(self) -> None:
        calibration = BoardCalibration(board_size=3, top_left=(0, 0), top_right=(2, 0), bottom_left=(0, 2))

        with self.assertRaises(ValueError):
            GridBoardDetector(calibration).detect(np.zeros((10, 10), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
