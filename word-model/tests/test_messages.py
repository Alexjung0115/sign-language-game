from __future__ import annotations

import json
import unittest

from server.messages import (
    ProtocolError,
    WordLandmarkFrameRequest,
    capabilities_message,
    parse_request,
)
from tests.helpers import hand_payload, landmark_frame


class MessageParsingTests(unittest.TestCase):
    def test_parses_two_hand_word_frame(self) -> None:
        request = parse_request(
            landmark_frame(
                left=hand_payload("LEFT"),
                right=hand_payload("RIGHT", 0.4),
            ),
        )
        self.assertIsInstance(request, WordLandmarkFrameRequest)
        assert isinstance(request, WordLandmarkFrameRequest)
        self.assertEqual(request.frame_width, 1280)
        self.assertEqual(request.frame_height, 720)
        self.assertEqual(len(request.left_hand.landmarks), 21)
        self.assertEqual(len(request.right_hand.landmarks), 21)

    def test_requires_both_slots_even_when_one_is_null(self) -> None:
        payload = json.loads(landmark_frame())
        del payload["hands"]["right"]
        with self.assertRaises(ProtocolError) as raised:
            parse_request(json.dumps(payload))
        self.assertEqual(raised.exception.code, "INVALID_HANDS")

    def test_rejects_empty_two_hand_frame(self) -> None:
        payload = json.loads(landmark_frame())
        payload["hands"] = {"left": None, "right": None}
        with self.assertRaises(ProtocolError) as raised:
            parse_request(json.dumps(payload))
        self.assertEqual(raised.exception.code, "NO_HANDS")

    def test_rejects_wrong_handedness_slot(self) -> None:
        payload = json.loads(landmark_frame())
        payload["hands"]["left"]["handedness"] = "RIGHT"
        with self.assertRaises(ProtocolError) as raised:
            parse_request(json.dumps(payload))
        self.assertEqual(
            raised.exception.code,
            "INVALID_HANDEDNESS",
        )

    def test_rejects_wrong_landmark_count(self) -> None:
        payload = json.loads(landmark_frame())
        payload["hands"]["left"]["landmarks"].pop()
        with self.assertRaises(ProtocolError) as raised:
            parse_request(json.dumps(payload))
        self.assertEqual(
            raised.exception.code,
            "INVALID_LANDMARK_COUNT",
        )

    def test_rejects_invalid_frame_size(self) -> None:
        payload = json.loads(landmark_frame())
        payload["frameWidth"] = 0
        with self.assertRaises(ProtocolError) as raised:
            parse_request(json.dumps(payload))
        self.assertEqual(
            raised.exception.code,
            "INVALID_FRAME_SIZE",
        )

    def test_capabilities_match_frontend_parser_contract(self) -> None:
        message = capabilities_message(
            "word-13-v1",
            ("airplane", "moon"),
            60,
            20,
            18,
            0.55,
        )
        self.assertEqual(message["type"], "CAPABILITIES")
        self.assertEqual(
            message["supportedSymbols"],
            ["airplane", "moon", "none"],
        )
        self.assertEqual(message["sequenceLength"], 60)
        self.assertEqual(
            message["confirmationAuthority"],
            "FRONTEND_TEMPORAL_DECODER",
        )
        self.assertEqual(
            message["confidenceThresholds"],
            {"airplane": 0.55, "moon": 0.55},
        )


if __name__ == "__main__":
    unittest.main()
