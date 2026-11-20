from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .frame_adapter import empty_model_frame
from .messages import none_prediction, prediction_message
from wordmodel.adapter import WordModelContract


class WordPredictor(Protocol):
    contract: WordModelContract

    def predict_sequence(
        self,
        raw_keypoints: np.ndarray,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class RecognitionConfig:
    inference_stride: int = 3
    smoothing_windows: int = 5
    confidence_threshold: float = 0.55
    minimum_detection_ratio: float = 0.35
    hand_lost_reset_ms: int = 250
    top_candidates: int = 3
    recommended_fps: int = 18

    def __post_init__(self) -> None:
        if self.inference_stride <= 0:
            raise ValueError("inference_stride must be positive")
        if self.smoothing_windows <= 0:
            raise ValueError("smoothing_windows must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1",
            )
        if not 0.0 <= self.minimum_detection_ratio <= 1.0:
            raise ValueError(
                "minimum_detection_ratio must be between 0 and 1",
            )


class RecognitionSession:
    def __init__(
        self,
        predictor: WordPredictor,
        config: RecognitionConfig = RecognitionConfig(),
    ) -> None:
        self._predictor = predictor
        self._config = config
        self._sequence: deque[np.ndarray] = deque(
            maxlen=predictor.contract.sequence_length,
        )
        self._probability_history: deque[np.ndarray] = deque(
            maxlen=config.smoothing_windows,
        )
        self._sample_index = 0
        self._missing_since: int | None = None
        self._missing_frames_appended = 0
        self._waiting_for_hand_after_reset = False

    @property
    def contract(self) -> WordModelContract:
        return self._predictor.contract

    @property
    def config(self) -> RecognitionConfig:
        return self._config

    @property
    def buffered_frames(self) -> int:
        return len(self._sequence)

    def process_model_frame(
        self,
        frame_id: int,
        captured_at: int,
        frame: np.ndarray,
    ) -> list[dict[str, object]]:
        if frame.shape != (2, 21, 2):
            raise ValueError(
                f"Expected model frame (2,21,2), got {frame.shape}",
            )
        if not np.isfinite(frame).all():
            raise ValueError("Model frame contains NaN or Inf")
        self._missing_since = None
        self._missing_frames_appended = 0
        self._waiting_for_hand_after_reset = False
        self._sequence.append(
            np.asarray(frame, dtype=np.float32).copy(),
        )
        self._sample_index += 1
        if (
            len(self._sequence) < self.contract.minimum_frames
            or self._sample_index % self._config.inference_stride != 0
        ):
            return [none_prediction(frame_id, captured_at)]

        sequence = np.asarray(self._sequence, dtype=np.float32)
        detected_frames = np.any(
            sequence != 0.0,
            axis=(2, 3),
        ).any(axis=1)
        detection_ratio = float(detected_frames.mean())
        if detection_ratio < self._config.minimum_detection_ratio:
            self._probability_history.clear()
            return [none_prediction(frame_id, captured_at)]

        probabilities = self._predictor.predict_sequence(sequence)
        self._validate_probabilities(probabilities)
        self._probability_history.append(probabilities)
        smoothed = np.mean(
            np.stack(self._probability_history),
            axis=0,
        )
        indexes = np.argsort(-smoothed, kind="stable")[
            : self._config.top_candidates
        ]
        candidates = [
            {
                "symbol": self.contract.labels[int(index)],
                "confidence": float(smoothed[int(index)]),
            }
            for index in indexes
        ]
        top = candidates[0]
        return [
            prediction_message(
                frame_id=frame_id,
                symbol=str(top["symbol"]),
                confidence=float(top["confidence"]),
                predicted_at=captured_at,
                top_candidates=candidates,
            ),
        ]

    def process_hand_not_detected(
        self,
        captured_at: int,
    ) -> list[dict[str, object]]:
        if self._waiting_for_hand_after_reset:
            return []
        if self._missing_since is None:
            self._missing_since = captured_at
        if self._missing_frames_appended < 3:
            self._sequence.append(empty_model_frame())
            self._sample_index += 1
            self._missing_frames_appended += 1
        if (
            captured_at - self._missing_since
            >= self._config.hand_lost_reset_ms
        ):
            self._clear_inference_state()
            self._waiting_for_hand_after_reset = True
        return []

    def reset(self) -> None:
        self._clear_inference_state()
        self._missing_since = None
        self._missing_frames_appended = 0
        self._waiting_for_hand_after_reset = False

    def _clear_inference_state(self) -> None:
        self._sequence.clear()
        self._probability_history.clear()
        self._sample_index = 0

    def _validate_probabilities(
        self,
        probabilities: np.ndarray,
    ) -> None:
        expected = (self.contract.output_size,)
        if probabilities.shape != expected:
            raise ValueError(
                f"Expected {expected} probabilities, "
                f"got {probabilities.shape}",
            )
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise ValueError(
                "Model probabilities must be finite and between 0 and 1",
            )
