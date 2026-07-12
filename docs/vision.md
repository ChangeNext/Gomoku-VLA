# Board Vision

`vision.board_detector` is a deterministic calibrated-grid baseline that
converts a top-down RGB/RGBA image into a board matrix.

## Current Pipeline

`BoardCalibration` maps each zero-based board cell to image pixels using the
top-left, top-right, and bottom-left board corners. The mapping is affine and
assumes a rectified or nearly top-down view.

`GridBoardDetector` samples a small patch at every intersection and a nearby
reference patch. `StoneClassifier` compares brightness and local contrast to
classify the intersection as empty, Black, or White. The output uses the same
integer values as `board.Player`.

This module detects appearance only. It does not infer whose turn it is,
validate move history, or repair an impossible board. Consumers must validate
the observed state before policy inference.

## Limitations

- no automatic corner or grid detection;
- no perspective or lens-distortion correction;
- hand-tuned brightness thresholds;
- no confidence score or unknown class;
- no temporal filtering or occlusion handling;
- no validation against the previous board state;
- reference sampling can be contaminated by nearby stones or board features.

The baseline is suitable for synthetic and controlled images. Real-camera
integration needs calibration persistence, confidence/error handling,
lighting robustness, and transition validation.

## Verification

`tests/test_vision.py` checks synthetic Black/White detection and invalid image
shape rejection. New calibration models and classifiers should include
fixtures for perspective, lighting, board color, occlusion, and false-stone
cases without changing the board-rule package.
