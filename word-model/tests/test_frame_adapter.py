from __future__ import annotations

import unittest

import numpy as np

from server.frame_adapter import request_to_model_frame
from server.messages import WordLandmarkFrameRequest, parse_request
from tests.helpers import hand_payload, landmark_frame


class FrameAdapterTests(unittest.TestCase):
    def test_restores_pixel_coordinates_in_fixed_slots(self) -> None:
        request = parse_request(
            landmark_frame(
                width=1000,
                height=500,
                left=hand_payload("LEFT"),
                right=hand_payload("RIGHT", 0.4),
            ),
        )
        assert isinstance(request, WordLandmarkFrameRequest)
        frame = request_to_model_frame(request)
        self.assertEqual(frame.shape, (2, 21, 2))
        self.assertEqual(frame.dtype, np.float32)
        np.testing.assert_allclose(frame[0, 0], [100.0, 100.0])
        np.testing.assert_allclose(frame[1, 0], [500.0, 100.0])

    def test_missing_hand_stays_zero(self) -> None:
        request = parse_request(landmark_frame())
        assert isinstance(request, WordLandmarkFrameRequest)
        frame = request_to_model_frame(request)
        self.assertTrue(np.any(frame[0] != 0.0))
        self.assertTrue(np.all(frame[1] == 0.0))


if __name__ == "__main__":
    unittest.main()
