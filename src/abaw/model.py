from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class LandmarkEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.coordinate_projection = nn.Linear(2, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 68, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        tokens = self.coordinate_projection(landmarks) + self.position_embedding
        return self.norm(self.encoder(tokens))


class AffectFusionModel(nn.Module):
    def __init__(
        self,
        backbone: str = "google/vit-base-patch16-224-in21k",
        hidden_dim: int = 256,
        landmark_layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.2,
        freeze_image_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.image_encoder = AutoModel.from_pretrained(backbone)
        image_dim = self.image_encoder.config.hidden_size
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.landmark_encoder = LandmarkEncoder(
            hidden_dim=hidden_dim,
            layers=landmark_layers,
            heads=attention_heads,
            dropout=dropout,
        )
        self.landmark_to_image = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.image_to_landmark = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.landmark_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )
        self.set_image_encoder_trainable(not freeze_image_encoder)

    def set_image_encoder_trainable(self, trainable: bool) -> None:
        self.image_encoder_trainable = trainable
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = trainable

    def train(self, mode: bool = True) -> AffectFusionModel:
        super().train(mode)
        if not self.image_encoder_trainable:
            self.image_encoder.eval()
        return self

    def forward(
        self,
        pixel_values: torch.Tensor,
        landmarks: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        image_tokens = self.image_projection(
            self.image_encoder(pixel_values=pixel_values).last_hidden_state
        )
        landmark_tokens = self.landmark_encoder(landmarks)

        landmark_context, _ = self.landmark_to_image(
            landmark_tokens, image_tokens, image_tokens, need_weights=False
        )
        landmark_feature = self.landmark_norm(landmark_tokens + landmark_context).mean(dim=1)

        image_query = image_tokens[:, :1]
        image_context, _ = self.image_to_landmark(
            image_query, landmark_tokens, landmark_tokens, need_weights=False
        )
        image_feature = self.image_norm(image_query + image_context).squeeze(1)

        gate = self.modality_gate(torch.cat([image_feature, landmark_feature], dim=-1))
        fused = gate * image_feature + (1.0 - gate) * landmark_feature
        prediction = self.regression_head(fused)
        if return_features:
            return {
                "prediction": prediction,
                "image_feature": image_feature,
                "landmark_feature": landmark_feature,
                "image_gate": gate,
            }
        return prediction
