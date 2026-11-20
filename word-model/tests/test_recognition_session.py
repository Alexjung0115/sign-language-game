from __future__ import annotations

import unittest

import numpy as np

from server.recognition_session import (
    RecognitionConfig,
    RecognitionSession,
)
from tests.helpers import FakePredictor, model_frame


class RecognitionSessionTests(unittest.TestCase):
    def test_warmup_and_stride_always_return_a_response(self) -> None:
        predictor = FakePredictor(
            [np.array([0.8, 0.2])],
            sequence_length=60,
            minimum_frames=2,
        )
        session = RecognitionSession(
            predictor,
            RecognitionConfig(
                inference_stride=3,
                smoothing_windows=5,
            ),
        )
        first = session.process_model_frame(1, 100, model_frame())
        second = session.process_model_frame(2, 150, model_frame(1))
        third = session.process_model_frame(3, 200, model_frame(2))
        self.assertEqual(first[0]["symbol"], "none")
        self.assertEqual(second[0]["symbol"], "none")
        self.assertEqual(third[0]["symbol"], "airplane")
        self.assertEqual(third[0]["frameId"], 3)
        self.assertEqual(len(predictor.calls), 1)

    def test_probability_smoothing_matches_live_settings(self) -> None:
        predictor = FakePredictor(
            [
                np.array([0.9, 0.1]),
                np.array([0.3, 0.7]),
            ],
            sequence_length=4,
            minimum_frames=1,
        )
        session = RecognitionSession(
            predictor,
            RecognitionConfig(
                inference_stride=1,
                smoothing_windows=5,
            ),
        )
        first = session.process_model_frame(1, 100, model_frame())
        second = session.process_model_frame(2, 150, model_frame(1))
        self.assertAlmostEqual(first[0]["confidence"], 0.9, places=6)
        self.assertEqual(second[0]["symbol"], "airplane")
        self.assertAlmostEqual(second[0]["confidence"], 0.6, places=6)

    def test_short_missing_gap_is_kept_for_interpolation(self) -> None:
        predictor = FakePredictor(
            [],
            sequence_length=60,
            minimum_frames=20,
        )
        session = RecognitionSession(predictor)
        session.process_model_frame(1, 1000, model_frame())
        session.process_hand_not_detected(1080)
        session.process_hand_not_detected(1160)
        self.assertEqual(session.buffered_frames, 3)
        session.process_model_frame(2, 1210, model_frame(2))
        self.assertEqual(session.buffered_frames, 4)

    def test_short_missing_gap_never_appends_more_than_three_zeros(self) -> None:
        predictor = FakePredictor(
            [],
            sequence_length=60,
            minimum_frames=20,
        )
        session = RecognitionSession(predictor)
        session.process_model_frame(1, 1000, model_frame())
        session.process_hand_not_detected(1010)
        session.process_hand_not_detected(1080)
        session.process_hand_not_detected(1160)
        session.process_hand_not_detected(1240)
        self.assertEqual(session.buffered_frames, 4)

    def test_continuous_missing_resets_after_250ms(self) -> None:
        predictor = FakePredictor(
            [],
            sequence_length=60,
            minimum_frames=20,
        )
        session = RecognitionSession(predictor)
        session.process_model_frame(1, 1000, model_frame())
        session.process_hand_not_detected(1080)
        session.process_hand_not_detected(1160)
        session.process_hand_not_detected(1330)
        self.assertEqual(session.buffered_frames, 0)
        session.process_hand_not_detected(1410)
        session.process_hand_not_detected(1490)
        self.assertEqual(session.buffered_frames, 0)
        session.process_model_frame(2, 1550, model_frame(2))
        self.assertEqual(session.buffered_frames, 1)

    def test_low_detection_ratio_returns_none_without_inference(self) -> None:
        predictor = FakePredictor(
            [],
            sequence_length=4,
            minimum_frames=1,
        )
        session = RecognitionSession(
            predictor,
            RecognitionConfig(
                inference_stride=1,
                minimum_detection_ratio=0.75,
            ),
        )
        session.process_hand_not_detected(100)
        session.process_hand_not_detected(180)
        response = session.process_model_frame(
            1,
            220,
            model_frame(),
        )
        self.assertEqual(response[0]["symbol"], "none")
        self.assertEqual(predictor.calls, [])


if __name__ == "__main__":
    unittest.main()
