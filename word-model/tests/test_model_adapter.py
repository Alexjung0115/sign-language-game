from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from wordmodel.adapter import LABELS, WordModelAdapter
from wordmodel.model import MultiEncoderSignClassifier


def write_tiny_checkpoint(
    directory: Path,
    labels: tuple[str, ...] = LABELS,
) -> Path:
    config = {
        "num_classes": len(labels),
        "feature_dims": 6,
        "d_model": 8,
        "num_layers": 1,
        "global_heads": 2,
        "hand_heads": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "max_length": 100,
    }
    model = MultiEncoderSignClassifier(**config)
    path = directory / "best_model.pt"
    torch.save(
        {
            "model_config": config,
            "model_state_dict": model.state_dict(),
            "id_to_class": {
                str(index): label
                for index, label in enumerate(labels)
            },
        },
        path,
    )
    return path


def sequence(length: int = 20) -> np.ndarray:
    frames = np.zeros((length, 2, 21, 2), dtype=np.float32)
    for frame_index in range(length):
        for point_index in range(21):
            frames[frame_index, 0, point_index] = (
                300 + point_index * 4 + frame_index,
                220 + point_index * 3 + frame_index * 0.5,
            )
    return frames


class WordModelAdapterTests(unittest.TestCase):
    def test_loads_checkpoint_and_returns_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            checkpoint = write_tiny_checkpoint(Path(raw_directory))
            adapter = WordModelAdapter(
                checkpoint,
                torch_threads=1,
            )
            probabilities = adapter.predict_sequence(sequence())
        self.assertEqual(probabilities.shape, (len(LABELS),))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertAlmostEqual(
            float(probabilities.sum()),
            1.0,
            places=5,
        )

    def test_rejects_different_class_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            wrong = tuple(reversed(LABELS))
            checkpoint = write_tiny_checkpoint(
                Path(raw_directory),
                wrong,
            )
            with self.assertRaisesRegex(
                ValueError,
                "class order",
            ):
                WordModelAdapter(checkpoint, torch_threads=1)


if __name__ == "__main__":
    unittest.main()
