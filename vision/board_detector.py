from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from board import Player


@dataclass(frozen=True)
class BoardCalibration:
    board_size: int
    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_left: tuple[float, float]

    def pixel_for_cell(self, row: int, col: int) -> tuple[float, float]:
        if not 0 <= row < self.board_size or not 0 <= col < self.board_size:
            raise ValueError(f"cell out of range: row={row}, col={col}")
        denom = max(1, self.board_size - 1)
        u = col / denom
        v = row / denom
        x = self.top_left[0] + u * (self.top_right[0] - self.top_left[0]) + v * (self.bottom_left[0] - self.top_left[0])
        y = self.top_left[1] + u * (self.top_right[1] - self.top_left[1]) + v * (self.bottom_left[1] - self.top_left[1])
        return x, y


@dataclass(frozen=True)
class StoneClassifier:
    empty_min_brightness: float = 80.0
    black_max_brightness: float = 75.0
    white_min_brightness: float = 165.0
    min_contrast_from_board: float = 28.0

    def classify_patch(self, patch: np.ndarray, board_reference: np.ndarray) -> Player:
        rgb = _as_rgb_float(patch)
        reference = _as_rgb_float(board_reference)
        brightness = float(rgb.mean())
        reference_brightness = float(reference.mean())
        delta = brightness - reference_brightness
        if brightness <= self.black_max_brightness and abs(delta) >= self.min_contrast_from_board:
            return Player.BLACK
        if brightness >= self.white_min_brightness and delta >= self.min_contrast_from_board:
            return Player.WHITE
        if reference_brightness >= self.empty_min_brightness:
            return Player.EMPTY
        return Player.EMPTY


class GridBoardDetector:
    def __init__(
        self,
        calibration: BoardCalibration,
        classifier: StoneClassifier | None = None,
        *,
        sample_radius: int = 4,
        reference_offset: int = 10,
    ) -> None:
        if sample_radius < 1:
            raise ValueError("sample_radius must be positive")
        self.calibration = calibration
        self.classifier = classifier or StoneClassifier()
        self.sample_radius = sample_radius
        self.reference_offset = reference_offset

    def detect(self, image: np.ndarray) -> list[list[int]]:
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("image must be an RGB/RGBA array")
        board: list[list[int]] = []
        for row in range(self.calibration.board_size):
            board_row: list[int] = []
            for col in range(self.calibration.board_size):
                x, y = self.calibration.pixel_for_cell(row, col)
                patch = _sample_patch(image, x, y, self.sample_radius)
                reference = _sample_patch(image, x + self.reference_offset, y + self.reference_offset, self.sample_radius)
                board_row.append(int(self.classifier.classify_patch(patch, reference).value))
            board.append(board_row)
        return board


def _sample_patch(image: np.ndarray, x: float, y: float, radius: int) -> np.ndarray:
    height, width = image.shape[:2]
    cx = int(round(x))
    cy = int(round(y))
    x0 = max(0, cx - radius)
    x1 = min(width, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(height, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"sample point outside image: x={x}, y={y}")
    return image[y0:y1, x0:x1, :3]


def _as_rgb_float(patch: np.ndarray) -> np.ndarray:
    if patch.size == 0:
        raise ValueError("empty image patch")
    return patch[:, :, :3].astype(float).reshape(-1, 3).mean(axis=0)
