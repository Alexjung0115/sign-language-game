from __future__ import annotations

import json
import unittest

import numpy as np

from server import main as server_main
from server.messages import ProtocolError
from server.recognition_session import RecognitionConfig, RecognitionSession
from tests.helpers import FakePredictor, landmark_frame


class ServerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = FakePredictor(
            [np.array([0.8, 0.2])],
            sequence_length=60,
            minimum_frames=1,
        )
        self.session = RecognitionSession(
            self.predictor,
            RecognitionConfig(inference_stride=1),
        )

    def test_landmark_frame_returns_matching_prediction(self) -> None:
        response = server_main.process_request(
            self.session,
            landmark_frame(frame_id=7, captured_at=1234),
        )[0]
        self.assertEqual(response["type"], "PREDICTION")
        self.assertEqual(response["frameId"], 7)
        self.assertEqual(response["predictedAt"], 1234)
        self.assertFalse(response["isStable"])
        self.assertEqual(response["symbol"], "airplane")

    def test_capabilities_include_word_contract(self) -> None:
        response = server_main.process_request(
            self.session,
            '{"type":"GET_CAPABILITIES"}',
        )[0]
        self.assertEqual(response["modelVersion"], "mock-word-v1")
        self.assertEqual(response["sequenceLength"], 60)
        self.assertEqual(response["minimumFrames"], 1)
        self.assertEqual(
            response["inputMode"],
            "TWO_HAND_LANDMARK_SEQUENCE",
        )

    def test_invalid_payload_does_not_mutate_session(self) -> None:
        payload = json.loads(landmark_frame())
        payload["hands"]["left"]["landmarks"].pop()
        with self.assertRaises(ProtocolError):
            server_main.process_request(
                self.session,
                json.dumps(payload),
            )
        self.assertEqual(self.session.buffered_frames, 0)

    def test_default_path_is_word(self) -> None:
        self.assertEqual(server_main.PATH, "/word")
        self.assertEqual(server_main.PORT, 8767)

    def test_only_word_path_is_accepted(self) -> None:
        self.assertIsNone(
            server_main.check_path(
                _Connection(),
                _Request("/word?room=1"),
            ),
        )
        response = server_main.check_path(
            _Connection(),
            _Request("/number"),
        )
        self.assertIsNotNone(response)
        self.assertEqual(response.status, 404)


class _Request:
    def __init__(self, path: str) -> None:
        self.path = path


class _Response:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body


class _Connection:
    @staticmethod
    def respond(status, body):
        return _Response(int(status), body)


if __name__ == "__main__":
    unittest.main()
