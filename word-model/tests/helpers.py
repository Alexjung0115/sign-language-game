from __future__ import annotations

import json

import numpy as np

from wordmodel.adapter import WordModelContract


def hand_payload(
    handedness: str,
    x_offset: float = 0.0,
) -> dict[str, object]:
    return {
        "handedness": handedness,
        "score": 0.95,
        "landmarks": [
            {
                "x": 0.1 + x_offset + index * 0.01,
                "y": 0.2 + index * 0.005,
                "z": -0.01 * index,
            }
            for index in range(21)
        ],
    }


def landmark_frame(
    frame_id: int = 1,
    captured_at: int = 1000,
    width: int = 1280,
    height: int = 720,
    left: dict[str, object] | None = None,
    right: dict[str, object] | None = None,
) -> str:
    return json.dumps(
        {
            "type": "LANDMARK_FRAME",
            "frameId": frame_id,
            "capturedAt": captured_at,
            "frameWidth": width,
            "frameHeight": height,
            "hands": {
                "left": (
                    hand_payload("LEFT")
                    if left is None and right is None
                    else left
                ),
                "right": right,
            },
        },
    )


class FakePredictor:
    def __init__(
        self,
        outputs: list[np.ndarray],
        *,
        labels: tuple[str, ...] = ("airplane", "moon"),
        sequence_length: int = 60,
        minimum_frames: int = 20,
    ) -> None:
        self.contract = WordModelContract(
            labels=labels,
            sequence_length=sequence_length,
            minimum_frames=minimum_frames,
            output_size=len(labels),
            model_version="mock-word-v1",
        )
        self._outputs = [
            np.asarray(output, dtype=np.float32)
            for output in outputs
        ]
        self.calls: list[np.ndarray] = []

    def predict_sequence(
        self,
        raw_keypoints: np.ndarray,
    ) -> np.ndarray:
        self.calls.append(raw_keypoints.copy())
        if not self._outputs:
            raise AssertionError("Fake prediction queue is empty")
        return self._outputs.pop(0)


def model_frame(offset: float = 0.0) -> np.ndarray:
    frame = np.zeros((2, 21, 2), dtype=np.float32)
    frame[0] = np.asarray(
        [
            (300.0 + offset + index * 4.0, 220.0 + index * 3.0)
            for index in range(21)
        ],
        dtype=np.float32,
    )
    return frame
