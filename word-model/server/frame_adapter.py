from __future__ import annotations

import numpy as np

from .messages import DetectedHand, WordLandmarkFrameRequest


FRAME_SHAPE = (2, 21, 2)
LEFT_SLOT = 0
RIGHT_SLOT = 1


def request_to_model_frame(
    request: WordLandmarkFrameRequest,
) -> np.ndarray:
    frame = np.zeros(FRAME_SHAPE, dtype=np.float32)
    _write_hand(
        frame,
        LEFT_SLOT,
        request.left_hand,
        request.frame_width,
        request.frame_height,
    )
    _write_hand(
        frame,
        RIGHT_SLOT,
        request.right_hand,
        request.frame_width,
        request.frame_height,
    )
    return frame


def empty_model_frame() -> np.ndarray:
    return np.zeros(FRAME_SHAPE, dtype=np.float32)


def _write_hand(
    frame: np.ndarray,
    slot: int,
    hand: DetectedHand | None,
    width: int,
    height: int,
) -> None:
    if hand is None:
        return
    frame[slot] = np.asarray(
        [
            (point.x * width, point.y * height)
            for point in hand.landmarks
        ],
        dtype=np.float32,
    )
