from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_length: int = 2000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(
            "positional_encoding",
            encoding.unsqueeze(0),
            persistent=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.shape[1]
        if sequence_length > self.positional_encoding.shape[1]:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"max_length {self.positional_encoding.shape[1]}.",
            )
        return self.dropout(
            inputs + self.positional_encoding[:, :sequence_length],
        )


class TransformerBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_length: int = 2000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_length=max_length,
            dropout=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    @staticmethod
    def masked_mean_pooling(
        encoded: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
        summed = (encoded * valid_mask).sum(dim=1)
        valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
        return summed / valid_count

    def forward(
        self,
        inputs: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.input_projection(inputs)
        encoded = encoded * math.sqrt(self.d_model)
        encoded = self.positional_encoding(encoded)
        encoded = self.transformer_encoder(
            encoded,
            src_key_padding_mask=padding_mask,
        )
        pooled = self.masked_mean_pooling(encoded, padding_mask)
        return self.classifier(self.output_norm(pooled))


class MultiEncoderSignClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dims: int = 6,
        d_model: int = 256,
        num_layers: int = 4,
        global_heads: int = 8,
        hand_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_length: int = 2000,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dims = feature_dims
        common = {
            "num_classes": num_classes,
            "d_model": d_model,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "max_length": max_length,
        }
        self.global_branch = TransformerBranch(
            input_dim=42 * feature_dims,
            num_heads=global_heads,
            **common,
        )
        self.left_branch = TransformerBranch(
            input_dim=21 * feature_dims,
            num_heads=hand_heads,
            **common,
        )
        self.right_branch = TransformerBranch(
            input_dim=21 * feature_dims,
            num_heads=hand_heads,
            **common,
        )

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
        return_branch_logits: bool = False,
    ):
        if features.ndim != 4:
            raise ValueError(
                f"Expected features (B,T,42,{self.feature_dims}), "
                f"got {tuple(features.shape)}.",
            )
        if features.shape[2:] != (42, self.feature_dims):
            raise ValueError(
                f"Unexpected landmark feature shape: {tuple(features.shape)}.",
            )

        batch_size, sequence_length = features.shape[:2]
        if padding_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                f"Expected padding mask {(batch_size, sequence_length)}, "
                f"got {tuple(padding_mask.shape)}.",
            )

        global_features = features.reshape(
            batch_size,
            sequence_length,
            42 * self.feature_dims,
        )
        left_features = features[:, :, :21].reshape(
            batch_size,
            sequence_length,
            21 * self.feature_dims,
        )
        right_features = features[:, :, 21:].reshape(
            batch_size,
            sequence_length,
            21 * self.feature_dims,
        )
        global_logits = self.global_branch(global_features, padding_mask)
        left_logits = self.left_branch(left_features, padding_mask)
        right_logits = self.right_branch(right_features, padding_mask)
        logits = global_logits + left_logits + right_logits
        if return_branch_logits:
            return {
                "logits": logits,
                "global_logits": global_logits,
                "left_logits": left_logits,
                "right_logits": right_logits,
            }
        return logits


def create_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device | str = "cpu",
) -> MultiEncoderSignClassifier:
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise KeyError("Checkpoint does not contain model_config.")
    model = MultiEncoderSignClassifier(**config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device)
