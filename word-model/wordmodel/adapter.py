from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading

import numpy as np
import torch

from .model import create_model_from_checkpoint
from .preprocessing import preprocess_sample


MODEL_VERSION = "word-19-v1"
LABELS = (
    "airplane",
    "bicycle",
    "moon",
    "motorcycle",
    "rain",
    "run",
    "ship",
    "star",
    "subway",
    "swim",
    "train",
    "walk",
    "wind",
    "bad",
    "bus",
    "car",
    "good",
    "helicopter",
    "sun",
)
MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = (
    MODEL_ROOT / "models" / MODEL_VERSION / "best_model.pt"
)


@dataclass(frozen=True)
class WordModelContract:
    labels: tuple[str, ...] = LABELS
    sequence_length: int = 60
    minimum_frames: int = 20
    feature_dims: int = 6
    output_size: int = len(LABELS)
    model_version: str = MODEL_VERSION

    def __post_init__(self) -> None:
        if self.output_size != len(self.labels):
            raise ValueError(
                "Model output size does not match label count",
            )
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Model labels must be unique")


class WordModelAdapter:
    def __init__(
        self,
        checkpoint_path: Path | None = None,
        torch_threads: int | None = None,
    ) -> None:
        configured_path = os.getenv(
            "HANDPRACTICE_WORD_CHECKPOINT",
        )
        self.checkpoint_path = (
            checkpoint_path
            or (Path(configured_path) if configured_path else None)
            or DEFAULT_CHECKPOINT_PATH
        ).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Word checkpoint not found: {self.checkpoint_path}",
            )
        self._validate_checksum()

        selected_threads = (
            torch_threads
            if torch_threads is not None
            else int(os.getenv("HANDPRACTICE_WORD_TORCH_THREADS", "2"))
        )
        if selected_threads <= 0:
            raise ValueError(
                "HANDPRACTICE_WORD_TORCH_THREADS must be positive",
            )
        torch.set_num_threads(selected_threads)
        self.device = torch.device("cpu")
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
            )
        self.contract = WordModelContract()
        self._validate_checkpoint(checkpoint)
        self._model = create_model_from_checkpoint(
            checkpoint,
            self.device,
        )
        self._model.eval()
        self._lock = threading.Lock()

    def predict_sequence(
        self,
        raw_keypoints: np.ndarray,
    ) -> np.ndarray:
        processed = preprocess_sample(
            np.asarray(raw_keypoints, dtype=np.float32),
            max_interpolation_gap=3,
            swap_ratio=0.65,
            outlier_threshold=3.0,
        )
        features = torch.from_numpy(
            np.asarray(processed["features"], dtype=np.float32),
        ).unsqueeze(0)
        padding_mask = torch.zeros(
            (1, features.shape[1]),
            dtype=torch.bool,
        )
        with self._lock:
            with torch.inference_mode():
                logits = self._model(features, padding_mask)
                probabilities = torch.softmax(
                    logits.float(),
                    dim=1,
                )
        output = probabilities[0].cpu().numpy().astype(
            np.float32,
            copy=False,
        )
        if output.shape != (self.contract.output_size,):
            raise ValueError(
                f"Expected {(self.contract.output_size,)} probabilities, "
                f"got {output.shape}",
            )
        if (
            not np.isfinite(output).all()
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise ValueError(
                "Model probabilities must be finite and between 0 and 1",
            )
        return output

    def _validate_checkpoint(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, dict):
            raise ValueError("Word checkpoint must contain a mapping")
        raw_mapping = checkpoint.get("id_to_class")
        if not isinstance(raw_mapping, dict):
            raise ValueError(
                "Word checkpoint does not contain id_to_class",
            )
        ordered_labels = tuple(
            str(raw_mapping[str(index)])
            if str(index) in raw_mapping
            else str(raw_mapping[index])
            for index in range(len(LABELS))
        )
        if ordered_labels != LABELS:
            raise ValueError(
                "Word checkpoint class order does not match word-19-v1",
            )
        config = checkpoint.get("model_config")
        if not isinstance(config, dict):
            raise ValueError(
                "Word checkpoint does not contain model_config",
            )
        if config.get("num_classes") != len(LABELS):
            raise ValueError(
                "Word checkpoint num_classes must be 19",
            )
        if config.get("feature_dims") != self.contract.feature_dims:
            raise ValueError(
                "Word checkpoint feature_dims must be 6",
            )

    def _validate_checksum(self) -> None:
        expected = os.getenv(
            "HANDPRACTICE_WORD_CHECKPOINT_SHA256",
            "",
        ).strip().lower()
        if not expected:
            return
        actual = _sha256(self.checkpoint_path)
        if actual != expected:
            raise ValueError(
                "Word checkpoint SHA256 does not match "
                "HANDPRACTICE_WORD_CHECKPOINT_SHA256",
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()
